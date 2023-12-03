from tqdm.auto import tqdm
import torch
import numpy as np
import os
import torch.nn.functional as F
import shutil
from datasets.pomdp import POMDPAgent
from datasets.kuka import KukaStackAgent, KukaRearrangeAgent

def eval_unconditional_1d(config, pipeline, val_dataloader, evaluator):
    progress_bar = tqdm(total=len(val_dataloader))
    progress_bar.set_description(f"Evaluating")

    evaluator.reset()
    for step, eval_batch in enumerate(val_dataloader):

        pred_trajectories = pipeline(
            num_inference_steps=config.num_inference_steps,
            conditions=eval_batch.conditions
        )
        # print(np.round(pred_trajectories[0, :, :4], 2))
        # print(eval_batch.trajectories[0, :, :4])
        # pred_trajectories = np.round(pred_trajectories)

        # Calculate score
        evaluator.update(pred_trajectories, eval_batch.trajectories.cpu().numpy(), config.action_dim)
        for b in range(pred_trajectories.shape[0]):
            if b == 2:
                print("="*20)
                print("Condition")
                gt_pos = eval_batch.conditions[0][b]
                gt_pos = np.round(gt_pos * 4.5 + 4.5)
                pred_pos = pred_trajectories[b, config.action_dim:]
                pred_pos = np.round(pred_pos * 4.5 + 4.5)
                gt_traj = (eval_batch.trajectories[b, config.action_dim:].transpose(1,0)).cpu().numpy()
                gt_traj = np.round(gt_traj * 4.5 + 4.5)
                print(gt_traj)
                print(gt_pos)
                for t in range(pred_pos.shape[1]):
                    print(pred_pos[:, t])
        progress_bar.update(1)

    progress_bar.close()
    score_log = evaluator.summarize()

    return score_log

def eval_unconditional_2d(config, pipeline, val_dataloader, evaluator):
    progress_bar = tqdm(total=len(val_dataloader))
    progress_bar.set_description(f"Evaluating")

    evaluator.reset()
    for step, eval_batch in enumerate(val_dataloader):

        pred_trajectories = pipeline(
            num_inference_steps=config.num_inference_steps,
            conditions=eval_batch.conditions
        )
        # print(np.round(pred_trajectories[0, :, :4], 2))
        # print(eval_batch.trajectories[0, :, :4])
        # pred_trajectories = np.round(pred_trajectories)

        # Calculate score
        evaluator.update(pred_trajectories, eval_batch.trajectories.cpu().numpy())
        progress_bar.update(1)

    progress_bar.close()
    score_log = evaluator.summarize()

    return score_log

def eval_conditional_2d(config, pipeline, val_dataloader, evaluator):
    progress_bar = tqdm(total=len(val_dataloader))
    progress_bar.set_description(f"Evaluating")

    evaluator.reset()
    for step, eval_batch in enumerate(val_dataloader):

        pred_trajectories = pipeline(
            num_inference_steps=config.num_inference_steps,
            conditions=eval_batch.conditions,
            input_ids=eval_batch.input_ids,
            instructions=eval_batch.instructions,
            guidance_scale=config.guidance_scale
        )
        # print(np.round(pred_trajectories[0, :, :4], 2))
        # print(eval_batch.trajectories[0, :, :4])
        # pred_trajectories = np.round(pred_trajectories)

        # Calculate score
        evaluator.update(pred_trajectories, eval_batch.trajectories.cpu().numpy())
        progress_bar.update(1)

    progress_bar.close()
    score_log = evaluator.summarize()

    return score_log


def eval_sequence_2d(config, pipeline, val_dataloader, evaluator):
    progress_bar = tqdm(total=len(val_dataloader))
    progress_bar.set_description(f"Evaluating")

    agents = POMDPAgent(val_dataloader.dataset.config)

    evaluator.reset()
    for step, eval_batch in enumerate(val_dataloader):
        agents.initialize(eval_batch.init_states.cpu().numpy(), eval_batch.input_ids.cpu().numpy())

        for i in range(10):

            if i == 0:
                conditions = eval_batch.conditions

            pred_trajectories = pipeline(
                num_inference_steps=config.num_inference_steps,
                conditions=conditions,
                input_ids=eval_batch.input_ids,
                instructions=eval_batch.instructions,
                guidance_scale=config.guidance_scale
            )
            next_states, finished = agents.step_entire(pred_trajectories[:, -1])
            next_states = torch.tensor(next_states).to(eval_batch.conditions[0].device)
            conditions = {0: next_states}

            if finished.all():
                break

        # pred_trajectories = pipeline(
        #     num_inference_steps=config.num_inference_steps,
        #     conditions=eval_batch.conditions,
        #     input_ids=eval_batch.input_ids,
        #     instructions=eval_batch.instructions,
        #     guidance_scale=config.guidance_scale
        # )
        
        # pred_trajectories[:, -1] = agents.trajectories
        evaluator.update(pred_trajectories, eval_batch.trajectories.cpu().numpy(), eval_batch.init_states.cpu().numpy())
        progress_bar.update(1)

    progress_bar.close()
    score_log = evaluator.summarize()

    return score_log


def eval_kuka(config, pipeline, val_dataloader, evaluator):
    progress_bar = tqdm(total=len(val_dataloader))
    progress_bar.set_description(f"Evaluating")

    if val_dataloader.dataset.config.type == "stacking":
        agents = KukaStackAgent(val_dataloader.dataset.config)
        total_step = 3
    elif val_dataloader.dataset.config.type == "rearrangement":
        agents = KukaRearrangeAgent(val_dataloader.dataset.config)
        total_step = 2
    else:
        raise NotImplementedError("Only stacking and rearrangement are supported for Kuka dataset.")

    evaluator.reset()
    for step, eval_batch in enumerate(val_dataloader):
        agents.initialize(eval_batch.trajectories[:, 0].cpu().numpy(), eval_batch.input_ids.cpu().numpy())
        for i in range(total_step):

            if i == 0:
                conditions = eval_batch.conditions

            pred_trajectories = pipeline(
                num_inference_steps=config.num_inference_steps,
                conditions=conditions,
                input_ids=eval_batch.input_ids,
                instructions=eval_batch.instructions[i],
                guidance_scale=config.guidance_scale
            )
            next_states = agents.step(pred_trajectories[:, -1])
            next_states = torch.tensor(next_states).to(eval_batch.conditions[0].device)
            conditions = {0: next_states}

        correct, approx_correct, goal_dist, total_stack = agents.get_result()

        # Calculate score
        evaluator.update_eval(correct, approx_correct, goal_dist, total_stack)
        progress_bar.update(1)

    progress_bar.close()
    score_log = evaluator.summarize()

    return score_log

def eval_alfred(config, pipeline, val_dataloader, evaluator):
    progress_bar = tqdm(total=len(val_dataloader))
    progress_bar.set_description(f"Evaluating")

    evaluator.reset()
    for step, eval_batch in enumerate(val_dataloader):

        pred_trajectories = pipeline(
            num_inference_steps=config.num_inference_steps,
            conditions=eval_batch.conditions,
            instructions=eval_batch.instructions,
            guidance_scale=config.guidance_scale
        )

        # Calculate score
        evaluator.update(pred_trajectories, eval_batch.trajectories.cpu().numpy())
        progress_bar.update(1)

    progress_bar.close()
    score_log = evaluator.summarize()

    return score_log
