
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class NumberEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, 
        hidden_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(1, hidden_size)

    def forward(self, x):
        # Input shape: (batch_size, 1)
        # Output shape: (batch_size, 1, text_encoder_hidden_size)
        x = self.encoder(x)
        x = x.unsqueeze(1)
        return x