from diffusers import DiffusionPipeline
import torch
from typing import List, Optional, Union
from utils.helpers import randn_tensor, apply_conditioning_1d, apply_conditioning_2d
import numpy as np


class PlanningPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet1DModel`]):
            A `UNet1DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        conditions: Optional[dict] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        batch_size = conditions[0].shape[0]
        if isinstance(self.unet.config.sample_size, int):
            trajectory_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
            )
        else:
            trajectory_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            trajectory = randn_tensor(trajectory_shape, generator=generator)
            trajectory = trajectory.to(self.device)
        else:
            trajectory = randn_tensor(trajectory_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        if len(trajectory_shape) == 4:
            trajectory = apply_conditioning_2d(trajectory, conditions)
        elif len(trajectory_shape) == 3:
            action_dim = self.unet.config.in_channels - conditions[0].shape[-1]
            trajectory = apply_conditioning_1d(trajectory, conditions, action_dim)

        for t in (self.scheduler.timesteps):
            # 1. predict noise model_output
            # print(t)
            # print(np.round(trajectory[0, :, :4].cpu().numpy(), 2))
            model_output = self.unet(trajectory, t).sample

            # 2. compute previous image: x_t -> x_t-1
            trajectory = self.scheduler.step(model_output, t, trajectory, generator=generator).prev_sample

            if len(trajectory_shape) == 4:
                trajectory = apply_conditioning_2d(trajectory, conditions)
            elif len(trajectory_shape) == 3:
                action_dim = self.unet.config.in_channels - conditions[0].shape[-1]
                trajectory = apply_conditioning_1d(trajectory, conditions, action_dim)


        trajectory = trajectory.cpu().numpy()

        return trajectory
