from diffusers import UNet1DModel, UNet2DModel, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from .encoder import NumberEncoder
from transformers import CLIPTextModel, AutoTokenizer


def build_model(config):
    model_name = config.name
    model, ema_model = None, None
    if model_name == "UNet1DModel":
        model = UNet1DModel(
            sample_size=config.sequence_length,
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            use_timestep_embedding=config.use_timestep_embedding,
            time_embedding_type=config.time_embedding_type,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            out_block_type=config.out_block_type,
            block_out_channels=config.block_out_channels,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
        )

        if config.use_ema:
            ema_model = EMAModel(
                model.parameters(),
                decay=config.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=config.ema_inv_gamma,
                power=config.ema_power,
                model_cls=UNet1DModel,
                model_config=model.config,
            )

    elif model_name == "UNet2DModel":
        model = UNet2DModel(
            sample_size=tuple(config.sample_size),
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            time_embedding_type=config.time_embedding_type,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
        )

        if config.use_ema:
            ema_model = EMAModel(
                model.parameters(),
                decay=config.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=config.ema_inv_gamma,
                power=config.ema_power,
                model_cls=UNet2DModel,
                model_config=model.config,
            )

    elif model_name == "UNet2DConditionModel":
        model = UNet2DConditionModel(
            sample_size=tuple(config.sample_size),
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            time_embedding_type=config.time_embedding_type,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            mid_block_type=config.mid_block_type,
            block_out_channels=config.block_out_channels,
            cross_attention_dim=config.cross_attention_dim,
            encoder_hid_dim=config.text_encoder_hidden_size,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
        )

        if config.use_ema:
            ema_model = EMAModel(
                model.parameters(),
                decay=config.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=config.ema_inv_gamma,
                power=config.ema_power,
                model_cls=UNet2DConditionModel,
                model_config=model.config,
            )

    return model, ema_model


def build_text_encoder(config):
    model_name = config.text_encoder_name
    model = None
    tokenizer = None
    if model_name == "NumberEncoder":
        model = NumberEncoder(config.text_encoder_hidden_size)
        tokenizer = None
    elif model_name == "CLIPTextModel":
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    return model, tokenizer