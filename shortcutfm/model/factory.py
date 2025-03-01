from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import SiLU
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

from shortcutfm.config import ModelConfig
from shortcutfm.model.model import FlowMatchingModel, TransformerNetModel


@dataclass
class TransformerNetModelModules:
    """Dataclass to hold the modules for the TransformerNetModel."""

    word_embedding: nn.Embedding
    lm_head: nn.Linear
    time_embed: nn.Sequential
    shortcut_embedding: nn.Embedding
    input_up_proj: Optional[nn.Sequential]
    input_transformers: nn.Module  # Can be BertEncoder or BertModel.encoder
    position_embeddings: nn.Embedding
    LayerNorm: nn.LayerNorm
    output_down_proj: Optional[nn.Sequential]
    position_ids: Tensor


class TransformerNetModelFactory:
    """Factory class to create TransformerNetModel instances from configuration."""

    def __init__(self, config: ModelConfig):
        # TODO: validation of config e.g. diffusion steps, shorcut and so on
        self.config = config

    def build(self) -> FlowMatchingModel:
        """Builds and returns a TransformerNetModel instance."""
        modules = self._create_modules()
        module = TransformerNetModel(**modules.__dict__, config=self.config)

        return FlowMatchingModel(
            module=module,
            diffusion_steps=self.config.diffusion_steps,
            min_shortcut_size=self.config.min_shortcut_size
        )

    def _create_modules(self) -> TransformerNetModelModules:
        """Creates the necessary modules based on the configuration."""
        config = AutoConfig.from_pretrained(self.config.config_name)
        config.hidden_dropout_prob = self.config.dropout

        if self.config.max_position_embeddings is not None:
            config.max_position_embeddings = self.config.max_position_embeddings

        input_dims = self.config.input_dims
        word_embedding = nn.Embedding(self.config.vocab_size, input_dims)
        nn.init.normal_(word_embedding.weight, mean=0.0, std=1e-3)
        if self.config.sc_rate > 0:
            input_dims = self.config.input_dims * 2

        lm_head = nn.Linear(input_dims, self.config.vocab_size)
        with torch.no_grad():
            lm_head.weight = word_embedding.weight

        time_embed_dim = self.config.hidden_t_dim * 4
        time_embed = nn.Sequential(
            nn.Linear(self.config.hidden_t_dim, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        num_possible_shortcuts = np.log2(self.config.diffusion_steps) - np.log2(
            self.config.min_shortcut_size
        ) + 2  # include min_shortcut_size and 0
        shortcut_embedding = nn.Embedding(int(num_possible_shortcuts), config.hidden_size)

        num_possible_shortcuts = np.log2(self.config.diffusion_steps) + 1  # include 0
        shortcut_embedding = nn.Embedding(int(num_possible_shortcuts), config.hidden_size)

        input_up_proj = None
        if input_dims != config.hidden_size:
            input_up_proj = nn.Sequential(
                nn.Linear(input_dims, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )

        if self.config.init_pretrained == "bert":
            print("initializing from pretrained bert...")
            temp_bert = BertModel.from_pretrained(self.config.config_name, config=config)
            word_embedding = temp_bert.embeddings.word_embeddings
            with torch.no_grad():
                lm_head.weight = word_embedding.weight

            input_transformers = temp_bert.encoder
            position_embeddings = temp_bert.embeddings.position_embeddings
            layer_norm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif self.config.init_pretrained == "no":
            print("BertConfig: ", config)
            input_transformers = BertEncoder(config)
            position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            raise ValueError("invalid type of init_pretrained")

        output_down_proj = None
        if self.config.output_dims != config.hidden_size:
            output_down_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, self.config.output_dims),
            )

        position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))

        return TransformerNetModelModules(
            word_embedding,
            lm_head,
            time_embed,
            shortcut_embedding,
            input_up_proj,
            input_transformers,
            position_embeddings,
            layer_norm,
            output_down_proj,
            position_ids
        )
