from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module

from shortcutfm.config import ModelConfig
from shortcutfm.nn import timestep_embedding


class FlowMatchingModel(Module, ABC):
    def __init__(self, module: Module, diffusion_steps, min_shortcut_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
        self.diffusion_steps = diffusion_steps
        self.min_shortcut_size = min_shortcut_size

    def __call__(self, x: Tensor, time_steps: Tensor, shortcuts: Optional[Tensor] = None) -> Tensor:
        """ Forward pass of the model. """
        if shortcuts is None:
            shortcuts = torch.zeros_like(time_steps, device=x.device)

        shortcut_for_emebd_layer = self._scale_shorcuts(shortcuts)
        time_steps = self._scale_timesteps(time_steps)

        return self.module(x, time_steps, shortcut_for_emebd_layer)

    def get_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.module.get_embeddings(input_ids)

    def compute_logits(self, hidden_repr: Tensor) -> Tensor:
        return self.module.compute_logits(hidden_repr)

    def _scale_timesteps(self, time_steps: Tensor) -> Tensor:
        return scale_diffusion_input(time_steps, self.diffusion_steps)

    def _scale_shorcuts(self, shortcuts: Tensor):
        return torch.where(
            shortcuts == 0,
            shortcuts,
            (torch.log2(shortcuts) - np.log2(self.min_shortcut_size) + 1).to(torch.int)
        )


def scale_diffusion_input(data: Tensor, diffusion_steps: int) -> Tensor:
    return data.float() * (1.0 / diffusion_steps)


class TransformerNetModel(nn.Module):
    """
    The full Transformer module with attention and timestep embedding.

    :param modules: Dataclass containing the pre-built modules.
    :param config: The configuration object.
    """

    def __init__(
            self,
            word_embedding: nn.Embedding,
            lm_head: nn.Linear,
            time_embed: nn.Sequential,
            shortcut_embedding: nn.Embedding,
            input_up_proj: Optional[nn.Sequential],
            input_transformers: nn.Module,
            position_embeddings: nn.Embedding,
            LayerNorm: nn.LayerNorm,
            output_down_proj: Optional[nn.Sequential],
            config: ModelConfig,
            position_ids: Tensor,
    ):
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding
        self.lm_head = lm_head
        self.time_embed = time_embed
        self.shortcut_embedding = shortcut_embedding
        self.input_up_proj = input_up_proj
        self.input_transformers = input_transformers
        self.position_embeddings = position_embeddings
        self.LayerNorm = LayerNorm
        self.output_down_proj = output_down_proj
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.register_buffer("position_ids", position_ids)

    def get_embeddings(self, input_ids):
        return self.word_embedding(input_ids)

    def compute_logits(self, hidden_repr):
        if self.config.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.config.logits_mode == 2:  # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # shoruct_size, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(
                self.lm_head.weight,
                text_emb_t
            )  # (vocab, shoruct_size) x (shoruct_size, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0),
                hidden_repr.size(1)
            )  # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, time_steps: Tensor, shortcuts: Tensor) -> Tensor:
        bsz, seq_len, *_ = x.size()

        time_embed = self.time_embed(timestep_embedding(time_steps, self.config.hidden_t_dim))

        shorcut_embed = self.shortcut_embedding(shortcuts)

        if self.input_up_proj is not None:
            x = self.input_up_proj(x)

        position_ids = self.position_ids[:, : seq_len]
        x = (
                self.position_embeddings(position_ids) +
                x +
                time_embed.unsqueeze(1).expand(-1, seq_len, -1) +
                shorcut_embed.unsqueeze(1).expand(-1, seq_len, -1)
        )

        x = self.dropout(self.LayerNorm(x))
        hidden_states = self.input_transformers(x).last_hidden_state

        if self.output_down_proj is not None:
            hidden_states = self.output_down_proj(hidden_states)

        return hidden_states
