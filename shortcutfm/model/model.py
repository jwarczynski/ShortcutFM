from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module
from transformers import ModernBertModel
from transformers.models.bert.modeling_bert import BertEncoder

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

    :param word_embedding: Word embedding layer
    :type word_embedding: nn.Embedding
    :param lm_head: Language model head
    :type lm_head: nn.Linear
    :param time_embed: Time embedding layer
    :type time_embed: nn.Sequential
    :param shortcut_embedding: Shortcut embedding layer
    :type shortcut_embedding: nn.Embedding
    :param input_up_proj: Optional input projection layer
    :type input_up_proj: Optional[nn.Sequential]
    :param input_transformers: Main transformer module
    :type input_transformers: ModernBertModel | BertEncoder
    :param position_embeddings: Optional position embeddings layer
    :type position_embeddings: Optional[nn.Embedding]
    :param layer_norm: Optional layer normalization
    :type layer_norm: Optional[nn.LayerNorm]
    :param output_down_proj: Optional output projection layer
    :type output_down_proj: Optional[nn.Sequential]
    :param config: Model configuration
    :type config: ModelConfig
    :param position_ids: Optional position IDs tensor
    :type position_ids: Optional[Tensor]
    """

    config: ModelConfig
    word_embedding: nn.Embedding
    lm_head: nn.Linear
    time_embed: nn.Sequential
    shortcut_embedding: nn.Embedding
    input_up_proj: nn.Module
    input_transformer: BertEncoder | ModernBertModel
    position_embeddings: Optional[nn.Embedding]
    layer_norm: nn.Module
    output_down_proj: nn.Module
    dropout: nn.Dropout
    hidden_size: int

    def __init__(
            self,
            word_embedding: nn.Embedding,
            lm_head: nn.Linear,
            time_embed: nn.Sequential,
            shortcut_embedding: nn.Embedding,
            input_up_proj: Optional[nn.Sequential],
            input_transformers: BertEncoder | ModernBertModel,
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
        self.shortcut_embedding = shortcut_embedding
        self.input_up_proj = input_up_proj if input_up_proj is not None else nn.Identity()
        self.input_transformer = input_transformers
        self.position_embeddings = position_embeddings
        self.layer_norm = layer_norm if layer_norm is not None else nn.Identity()
        self.output_down_proj = output_down_proj if output_down_proj is not None else nn.Identity()
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

        x =  self.input_up_proj(x)

        # Add time and shortcut embeddings
        x = x + time_embed.unsqueeze(1).expand(-1, seq_len, -1) + shorcut_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # Add position embeddings if available
        if self.position_embeddings is not None:
            position_ids = self.position_ids[:, :seq_len]
            x = x + self.position_embeddings(position_ids)

        x = self.dropout(self.layer_norm(x))
        if isinstance(self.input_transformer, BertEncoder):
            hidden_states = self.input_transformer(hidden_states=x).last_hidden_state
        else:  # ModernBert
            hidden_states = self.input_transformer(inputs_embeds=x).last_hidden_state
        hidden_states = self.output_down_proj(hidden_states)

        return hidden_states
