
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
import numpy as np
import os
import torch.nn.functional as F
import shutil
from accelerate.logging import get_logger
from eval.conditional_pipeline import KukaPipeline
from utils.helpers import apply_kuka_2d

logger = get_logger(__name__, log_level="INFO")

def train_kuka(config, model, text_encoder, tokenizer, noise_scheduler, optimizer, train_dataloader, val_dataloader, evaluator, lr_scheduler, ema_model=None):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    try:
        out_channels = model.config.out_channels
    except:
        out_channels = model.module.config.out_channels

    text_encoder.to(accelerator.device)
    text_encoder.requires_grad_(False)

    if ema_model is not None:
        ema_model.to(accelerator.device)

    # whether to use classifier-free guidance
    guidance_scale = config.guidance_scale

    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            trajectories = batch.trajectories
            # Sample noise to add to the images
            noise = torch.randn(trajectories.shape).to(trajectories.device)
            # cause we condition on t=0, we don't want to add noise to the conditioning
            bs = trajectories.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=trajectories.device
            ).long()
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_trajectories = noise_scheduler.add_noise(trajectories, noise, timesteps)
            noisy_trajectories = apply_kuka_2d(noisy_trajectories, batch.conditions)
            # print(trajectories.shape, batch.conditions[0].shape, noisy_trajectories.shape, noise.shape)

            # For input_ids
            # input_ids = batch.input_ids.unsqueeze(1).float()
            # encoder_hidden_states = input_ids.unsqueeze(2).expand(-1,-1,64) / 10

            # For instructions
            input_ids = tokenizer(
                batch.instructions,
                padding="longest",
                return_tensors="pt",
            )
            encoder_hidden_states = text_encoder(input_ids["input_ids"].to(accelerator.device)).last_hidden_state


            if guidance_scale > 0:
                noisy_trajectories = torch.cat([noisy_trajectories] * 2, dim=0)
                timesteps = torch.cat([timesteps, timesteps], dim=0)
                unconditional_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                encoder_hidden_states = torch.cat([encoder_hidden_states, unconditional_encoder_hidden_states], dim=0)

            with accelerator.accumulate(model):
                # b x 2 x h x w
                noise_pred = model(noisy_trajectories, timesteps, encoder_hidden_states, return_dict=False)[0]
                if guidance_scale > 0:
                    noise_cond, noise_uncond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                # full version
                loss = F.mse_loss(noise_pred, noise[:, 1:])
                # path only
                loss = F.mse_loss(noise_pred[:, -1], noise[:, -1])
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if ema_model is not None:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % config.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if config.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(config.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= config.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                        if accelerator.is_main_process:
                            save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if ema_model is not None:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        # evaluation
        if accelerator.is_main_process:
            if epoch % config.eval_per_epoch == 0 or epoch == config.num_epochs - 1:
                progress_bar = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process)
                progress_bar.set_description(f"Eval Epoch {epoch}")
                unet = accelerator.unwrap_model(model)

                if ema_model is not None:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = KukaPipeline(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=noise_scheduler)
                generator = torch.Generator(device=pipeline.device).manual_seed(0)

                evaluator.reset()
                for step, eval_batch in enumerate(val_dataloader):
                    
                    pred_trajectories = pipeline(
                        generator=generator,
                        num_inference_steps=config.num_inference_steps,
                        conditions=eval_batch.conditions,
                        input_ids=eval_batch.input_ids,
                        instructions=eval_batch.instructions,
                        guidance_scale=guidance_scale
                    )
                    # print(np.round(pred_trajectories[0, :, :4], 2))
                    # print(eval_batch.trajectories[0, :, :4])
                    # pred_trajectories = np.round(pred_trajectories)

                    # Calculate score
                    evaluator.update(
                        pred_trajectories, 
                        eval_batch.trajectories.cpu().numpy(), 
                    )
                    
                    progress_bar.update(1)

                progress_bar.close()
                score_log = evaluator.summarize()
                accelerator.log(score_log, step=global_step)

            if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if ema_model is not None:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = KukaPipeline(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=noise_scheduler)

                pipeline.save_pretrained(config.output_dir)

                if ema_model is not None:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()


