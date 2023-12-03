from diffusers import UNet1DModel
from utils.helpers import apply_conditioning

class UNet1DConditionModel(UNet1DModel):
    def __init__(self, config):
        super().__init__(
            sample_size=config.sequence_length,
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            use_timestep_embedding=config.use_timestep_embedding,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            act_fn=config.act_fn
        )
        
