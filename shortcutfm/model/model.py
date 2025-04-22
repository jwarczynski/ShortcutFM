from abc import ABC, abstractmethod
from typing import Optional, override

import torch
from torch import Tensor, nn
from torch.nn import Module

from shortcutfm.config import ModelConfig
from shortcutfm.nn import timestep_embedding


class BackboneTransformer(nn.Module, ABC):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """computes last hidden states of the transformer"""


class ModernBertBackbone(BackboneTransformer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(inputs_embeds=x).last_hidden_state


class BertEncoderBackbone(BackboneTransformer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(hidden_states=x).last_hidden_state


class FFNBackbone(BackboneTransformer):
    """Feed-forward network backbone for TransformerNetModel."""

    def __init__(
            self,
            input_dims: int,
            hidden_dims: int,
            num_layers: int,
    ):
        super().__init__(None)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dims if i == 0 else hidden_dims, hidden_dims),
                    nn.ReLU(),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class FFNModule(Module):
    def __init__(self, word_embedding: nn.Embedding, lm_head: nn.Linear, backbone: BackboneTransformer):
        super().__init__()
        self.word_embedding = word_embedding
        self.lm_head = lm_head
        self.backbone = backbone

    def get_embeddings(self, input_ids):
        return self.word_embedding(input_ids)

    def compute_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x: Tensor, time_steps: Tensor, shortcuts: Optional[Tensor] = None) -> Tensor:
        return self.backbone(x)


class FlowMatchingModel(Module):
    def __init__(self, module: Module, diffusion_steps, min_shortcut_size, scale_time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
        self.diffusion_steps = diffusion_steps
        self.min_shortcut_size = min_shortcut_size
        self.scale_time = scale_time

    def forward(self, x: Tensor, time_steps: Tensor, shortcuts: Optional[Tensor] = None) -> Tensor:
        """ Forward pass of the model. """
        if shortcuts is None:
            shortcuts = torch.zeros_like(time_steps, device=x.device)

        if self.scale_time:
            shortcuts = self._scale_shortcuts(shortcuts)
            time_steps = self._scale_time_steps(time_steps)

        return self.module(x, time_steps, shortcuts)

    def get_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.module.get_embeddings(input_ids)

    def compute_logits(self, hidden_repr: Tensor) -> Tensor:
        return self.module.compute_logits(hidden_repr)

    def _scale_time_steps(self, time_steps: Tensor) -> Tensor:
        return scale_diffusion_input(time_steps, self.diffusion_steps)

    def _scale_shortcuts(self, shortcuts: Tensor):
        return scale_diffusion_input(shortcuts, self.diffusion_steps)


def scale_diffusion_input(data: Tensor, diffusion_steps: int) -> Tensor:
    return data.float() * (1.0 / diffusion_steps)


class TransformerNetModel(Module):
    """Transformer network model for flow matching."""

    def __init__(
            self,
            *,
            word_embedding: nn.Embedding,
            lm_head: nn.Linear,
            time_embed: nn.Sequential,
            backbone_transformer: BackboneTransformer,
            shortcut_embedding: Optional[nn.Module] = None,
            input_up_proj: Optional[nn.Sequential] = None,
            position_embeddings: Optional[nn.Embedding] = None,
            layer_norm: Optional[nn.LayerNorm] = None,
            output_down_proj: Optional[nn.Sequential] = None,
            config: ModelConfig = None,
            position_ids: Optional[Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding
        self.lm_head = lm_head
        self.time_embed = time_embed
        self.input_up_proj = input_up_proj if input_up_proj is not None else nn.Identity()
        self.backbone_transformer = backbone_transformer
        self.position_embeddings = position_embeddings
        self.layer_norm = layer_norm if layer_norm is not None else nn.Identity()
        self.output_down_proj = output_down_proj if output_down_proj is not None else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.register_buffer("position_ids", position_ids)
        self.register_module("shortcut_embedding", shortcut_embedding)

    def get_embeddings(self, input_ids):
        word_embeddings = self.word_embedding(input_ids)
        if self.config.normalize_word_embedding:
            word_embeddings = word_embeddings / (word_embeddings.norm(dim=-1, keepdim=True) + 1e-10)
        return word_embeddings

    def compute_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x: Tensor, time_steps: Tensor, shortcuts: Tensor) -> Tensor:
        bsz, seq_len, *_ = x.size()

        timestep_emb = self.time_embed(timestep_embedding(time_steps, self.config.hidden_t_dim))

        x = self.input_up_proj(x)

        x = x + timestep_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Add shortcut embedding if available
        if self.shortcut_embedding is not None:
            shortcut_emb = self.shortcut_embedding(timestep_embedding(shortcuts, self.config.hidden_shortcut_dim))
            x = x + shortcut_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Add position embeddings if available
        if self.position_embeddings is not None:
            position_ids = self.position_ids[:, :seq_len]
            x = x + self.position_embeddings(position_ids)

        x = self.dropout(self.layer_norm(x))
        hidden_states = self.backbone_transformer(x)
        hidden_states = self.output_down_proj(hidden_states)

        return hidden_states


class StackedEmbeddingTransformerNetModel(TransformerNetModel):

    @override
    def forward(self, x: Tensor, time_steps: Tensor, shortcuts: Tensor) -> Tensor:
        bsz, seq_len, *_ = x.size()

        # Add position embeddings if available
        if self.position_embeddings is not None:
            position_ids = self.position_ids[:, :seq_len]
            x = x + self.position_embeddings(position_ids)

        # Add time embedding
        timestep_emb = self.time_embed(timestep_embedding(time_steps, self.config.hidden_t_dim))
        x = torch.cat((x, timestep_emb.unsqueeze(1).expand(-1, seq_len, -1)), dim=-1)

        # Add shortcut embedding if available
        if self.shortcut_embedding is not None:
            shortcut_emb = self.shortcut_embedding(timestep_embedding(shortcuts, self.config.hidden_shortcut_dim))
            x = torch.cat((x, shortcut_emb.unsqueeze(1).expand(-1, seq_len, -1)), dim=-1)

        x = self.input_up_proj(x)
        x = self.dropout(self.layer_norm(x))
        hidden_states = self.backbone_transformer(x)
        hidden_states = self.output_down_proj(hidden_states)

        return hidden_states
