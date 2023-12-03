from diffusers import DiffusionPipeline
import torch
from typing import List, Optional, Union
from utils.helpers import randn_tensor, apply_conditioning_2d, apply_representation_2d, apply_kuka_2d
import numpy as np
import os
import matplotlib.pyplot as plt

class Conditional2DPipeline(DiffusionPipeline):
    def __init__(self, unet, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.register_modules(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        conditions: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        instructions: Optional[List[str]] = None,
        guidance_scale: float = 7.5,
    ):
        batch_size = conditions[0].shape[0]
        trajectory_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            trajectory = randn_tensor(trajectory_shape, generator=generator)
            trajectory = trajectory.to(self.device)
        else:
            trajectory = randn_tensor(trajectory_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        out_channels = self.unet.config.out_channels
        if out_channels == 1:
            trajectory = apply_conditioning_2d(trajectory, conditions)
        elif out_channels == 2:
            trajectory = apply_representation_2d(trajectory, conditions)

        # if input_ids.dim() == 1:
        #     input_ids = input_ids.unsqueeze(1)
        # input_ids = input_ids.float().to(self.device)
        # # encoder_hidden_states = self.text_encoder(input_ids)
        # encoder_hidden_states = input_ids.unsqueeze(2).expand(-1,-1,64) / 10

        input_ids = self.tokenizer(instructions, return_tensors="pt", padding="longest").input_ids
        encoder_hidden_states = self.text_encoder(input_ids.to(self.device)).last_hidden_state


        if guidance_scale > 0:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.cat([encoder_hidden_states, uncond_encoder_hidden_states], dim=0)

        for t in (self.scheduler.timesteps):
            trajectory_input = torch.cat([trajectory]*2, dim=0) if guidance_scale > 0 else trajectory
            # 1. predict noise model_output
            # print(t)
            # print(np.round(trajectory[0, :, :4].cpu().numpy(), 2))
            model_output = self.unet(trajectory_input, t, encoder_hidden_states).sample

            if guidance_scale > 0:
                cond_noise, uncond_noise = model_output.chunk(2)
                model_output = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

            # 2. compute previous image: x_t -> x_t-1
            updated_sample = self.scheduler.step(model_output, t, trajectory[:, -out_channels:], generator=generator).prev_sample
            # 3. update trajectory
            trajectory[:, -out_channels:] = updated_sample
            
            # output_dir = "output/visualization/process/"
            # img_fn = output_dir + "img_{}.png".format(t)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # if not os.path.exists(img_fn):
            #     plt.imshow(trajectory[0, 1].cpu().numpy())
            #     plt.colorbar()
            #     plt.savefig(img_fn)
            #     plt.close()

            # trajectory = apply_representation_2d(trajectory, conditions)
        
        # img_fn = output_dir + "path.png"
        # if not os.path.exists(img_fn):
        #     # plt.imshow(np.round(trajectory[0, -1].cpu().numpy()) + np.round(trajectory[0, 0].cpu().numpy()))
        #     plt.imshow(np.round(trajectory[0, -1].cpu().numpy()))
        #     plt.colorbar()
        #     plt.savefig(img_fn)
        #     plt.close()

        trajectory = trajectory.cpu().numpy()

        return trajectory

class KukaPipeline(DiffusionPipeline):
    def __init__(self, unet, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.register_modules(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        conditions: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        instructions: Optional[List[str]] = None,
        guidance_scale: float = 7.5,
    ):
        batch_size = conditions[0].shape[0]
        trajectory_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            trajectory = randn_tensor(trajectory_shape, generator=generator)
            trajectory = trajectory.to(self.device)
        else:
            trajectory = randn_tensor(trajectory_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        out_channels = self.unet.config.out_channels
        trajectory = apply_kuka_2d(trajectory, conditions)
        input_ids = self.tokenizer(instructions, return_tensors="pt", padding="longest").input_ids
        encoder_hidden_states = self.text_encoder(input_ids.to(self.device)).last_hidden_state

        if guidance_scale > 0:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.cat([encoder_hidden_states, uncond_encoder_hidden_states], dim=0)

        for t in (self.scheduler.timesteps):
            trajectory_input = torch.cat([trajectory]*2, dim=0) if guidance_scale > 0 else trajectory
            # 1. predict noise model_output
            model_output = self.unet(trajectory_input, t, encoder_hidden_states).sample

            if guidance_scale > 0:
                cond_noise, uncond_noise = model_output.chunk(2)
                model_output = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

            # 2. compute previous image: x_t -> x_t-1
            updated_sample = self.scheduler.step(model_output, t, trajectory[:, -out_channels:], generator=generator).prev_sample
            # 3. update trajectory
            trajectory[:, -out_channels:] = updated_sample
            

        trajectory = trajectory.cpu().numpy()

        return trajectory

class AlfredPipeline(DiffusionPipeline):
    def __init__(self, unet, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.register_modules(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        conditions: Optional[dict] = None,
        instructions: Optional[List[str]] = None,
        guidance_scale: float = 7.5,
    ):
        batch_size = conditions[0].shape[0]
        trajectory_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            trajectory = randn_tensor(trajectory_shape, generator=generator)
            trajectory = trajectory.to(self.device)
        else:
            trajectory = randn_tensor(trajectory_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        out_channels = self.unet.config.out_channels
        trajectory = apply_representation_2d(trajectory, conditions)

        input_ids = self.tokenizer(instructions, return_tensors="pt", padding=True, truncation=True, max_length=77).input_ids
        encoder_hidden_states = self.text_encoder(input_ids.to(self.device)).last_hidden_state

        if guidance_scale > 0:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.cat([encoder_hidden_states, uncond_encoder_hidden_states], dim=0)

        for t in (self.scheduler.timesteps):
            trajectory_input = torch.cat([trajectory]*2, dim=0) if guidance_scale > 0 else trajectory
            # 1. predict noise model_output
            model_output = self.unet(trajectory_input, t, encoder_hidden_states).sample

            if guidance_scale > 0:
                cond_noise, uncond_noise = model_output.chunk(2)
                model_output = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

            # 2. compute previous image: x_t -> x_t-1
            updated_sample = self.scheduler.step(model_output, t, trajectory[:, -out_channels:], generator=generator).prev_sample
            # 3. update trajectory
            trajectory[:, -out_channels:] = updated_sample
            

        trajectory = trajectory.cpu().numpy()

        return trajectory
