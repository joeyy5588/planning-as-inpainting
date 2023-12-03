from diffusers import DDPMScheduler, DDIMScheduler
from .train_unconditional import train_unconditional_1d, train_unconditional_2d
from .train_conditional import train_conditional_2d
from .train_represntation import train_represntation_2d
from .train_kuka import train_kuka
from .train_alfred import train_alfred
from .evaluation import eval_unconditional_1d, eval_unconditional_2d, eval_conditional_2d, eval_sequence_2d, eval_kuka, eval_alfred

def build_scheduler(config):
    if config.scheduler_type == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=config.n_diffusion_steps, 
            beta_schedule=config.beta_schedule
        )

    elif config.scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=config.n_diffusion_steps, 
            beta_schedule=config.beta_schedule
        )

    return scheduler


def build_train_loop(config):
    dataset_name = config.name
    if dataset_name == "GridDataset":
        train_loop = train_unconditional_1d
    elif dataset_name == "SimpleDataset":
        train_loop = train_unconditional_1d
    elif dataset_name == "Simple2DDataset":
        train_loop = train_unconditional_2d
    elif dataset_name == "Grid2DDataset":
        train_loop = train_unconditional_2d
    elif dataset_name == "GridConditionDataset":
        train_loop = train_conditional_2d
    elif dataset_name == "GridRepresentationDataset":
        train_loop = train_represntation_2d
    elif dataset_name == "GridHeatMapDataset":
        train_loop = train_represntation_2d
    elif dataset_name == "GridSequenceDataset":
        train_loop = train_represntation_2d
    elif dataset_name == "KukaDataset":
        train_loop = train_kuka
    elif "Alfred" in dataset_name:
        train_loop = train_alfred

    return train_loop

def build_eval_loop(config):
    dataset_name = config.name
    if dataset_name == "GridDataset":
        eval_loop = eval_unconditional_1d
    elif dataset_name == "SimpleDataset":
        eval_loop = eval_unconditional_1d
    elif dataset_name == "Simple2DDataset":
        eval_loop = eval_unconditional_2d
    elif dataset_name == "Grid2DDataset":
        eval_loop = eval_unconditional_2d
    elif dataset_name == "GridConditionDataset":
        eval_loop = eval_conditional_2d
    elif dataset_name == "GridRepresentationDataset":
        eval_loop = eval_conditional_2d
    elif dataset_name == "GridHeatMapDataset":
        eval_loop = eval_conditional_2d
    elif dataset_name == "GridSequenceDataset":
        eval_loop = eval_sequence_2d
    elif dataset_name == "KukaDataset":
        eval_loop = eval_kuka
    elif "Alfred" in dataset_name:
        eval_loop = eval_alfred

    return eval_loop


