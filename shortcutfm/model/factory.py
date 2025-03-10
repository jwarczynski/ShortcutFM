from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import SiLU
from transformers import AutoConfig, BertModel, ModernBertModel
from transformers.models.bert.modeling_bert import BertEncoder

from shortcutfm.config import ModelConfig
from shortcutfm.model.model import FlowMatchingModel, TransformerNetModel


@dataclass
class TransformerNetModelModules:
    """Dataclass to hold the modules for the TransformerNetModel."""

    word_embedding: nn.Embedding
    lm_head: nn.Linear
    time_embed: nn.Sequential
    input_transformer: nn.Module  # Can be BertEncoder or BertModel.encoder
    shortcut_embedding: Optional[nn.Module] = None
    input_up_proj: Optional[nn.Sequential] = None
    position_embeddings: Optional[nn.Embedding] = None
    layer_norm: Optional[nn.LayerNorm] = None
    output_down_proj: Optional[nn.Sequential] = None
    position_ids: Optional[Tensor] = None


class TransformerNetModelFactory:
    """Factory class to create TransformerNetModel instances from configuration."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.bert_config = self._create_bert_config()

    def _create_bert_config(self) -> AutoConfig:
        """Create and configure the BERT configuration.

        :return: Configured BERT config
        :rtype: AutoConfig
        """
        config = AutoConfig.from_pretrained(self.config.config_name)
        config.hidden_dropout_prob = self.config.dropout

        if self.config.max_position_embeddings is not None:
            config.max_position_embeddings = self.config.max_position_embeddings

        return config

    def build(self) -> FlowMatchingModel:
        """Builds and returns a TransformerNetModel instance.

        :return: Configured FlowMatchingModel instance
        :rtype: FlowMatchingModel
        """
        modules = self._create_modules()
        module = TransformerNetModel(**modules.__dict__, config=self.config)

        return FlowMatchingModel(
            module=module,
            diffusion_steps=self.config.diffusion_steps,
            min_shortcut_size=self.config.min_shortcut_size
        )

    def _create_modules(self) -> TransformerNetModelModules:
        """Creates all necessary modules based on the configuration.

        :return: Dataclass containing all model modules
        :rtype: TransformerNetModelModules
        """
        # Create base embeddings and projections
        word_embedding, lm_head = self._create_word_embeddings()
        time_embed = self._create_time_embedding(self.config.hidden_t_dim)

        # Create shortcut embedding if needed
        shortcut_embedding = None
        if self.config.hidden_shortcut_dim is not None:
            shortcut_embedding = self._create_shortcut_embedding()

        # Create input projection if needed
        input_up_proj = self._create_input_projection()

        # Create transformer backbone
        input_transformers, position_embeddings, layer_norm = self._create_transformer_backbone(word_embedding)

        # Create output projection if needed
        output_down_proj = self._create_output_projection()

        # Create position IDs
        position_ids = self._create_position_ids()

        return TransformerNetModelModules(
            word_embedding=word_embedding,
            lm_head=lm_head,
            time_embed=time_embed,
            shortcut_embedding=shortcut_embedding,
            input_up_proj=input_up_proj,
            input_transformer=input_transformers,
            position_embeddings=position_embeddings,
            layer_norm=layer_norm,
            output_down_proj=output_down_proj,
            position_ids=position_ids
        )

    def _create_word_embeddings(self) -> Tuple[nn.Embedding, nn.Linear]:
        """Create word embeddings and language model head.

        :return: Tuple of (word_embedding, lm_head)
        :rtype: Tuple[nn.Embedding, nn.Linear]
        """
        input_dims = self.config.input_dims
        word_embedding = nn.Embedding(self.config.vocab_size, input_dims)
        nn.init.normal_(word_embedding.weight, mean=0.0, std=self.config.word_embedding_std)

        lm_head = nn.Linear(input_dims, self.config.vocab_size, bias=True)
        with torch.no_grad():
            lm_head.weight = word_embedding.weight

        return word_embedding, lm_head

    def _create_time_embedding(self, input_dim, scale_factor: int = 4) -> nn.Sequential:
        """Create time embedding network.

        :return: Time embedding network
        :rtype: nn.Sequential
        """
        embedding_dim = input_dim * scale_factor
        return nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            SiLU(),
            nn.Linear(embedding_dim, self.bert_config.hidden_size),
        )

    def _create_shortcut_embedding(self) -> nn.Module:
        """Create shortcut embedding layer.

        :return: Shortcut embedding layer
        :rtype: nn.Module
        """
        return self._create_time_embedding(self.config.hidden_shortcut_dim)

    def _create_input_projection(self) -> Optional[nn.Sequential]:
        """Create input projection if dimensions don't match.

        :return: Input projection network or None if not needed
        :rtype: Optional[nn.Sequential]
        """
        input_dims = self.config.input_dims * (2 if self.config.sc_rate > 0 else 1)
        if input_dims != self.bert_config.hidden_size:
            return nn.Sequential(
                nn.Linear(input_dims, self.bert_config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
            )
        return None

    def _create_transformer_backbone(
            self,
            word_embedding: nn.Embedding
    ) -> Tuple[nn.Module, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
        """Create transformer backbone based on configuration.

        :param word_embedding: Word embedding layer
        :type word_embedding: nn.Embedding
        :return: Tuple of (input_transformers, position_embeddings, layer_norm)
        :rtype: Tuple[nn.Module, Optional[nn.Embedding], Optional[nn.LayerNorm]]
        """
        if self.config.init_pretrained == "bert":
            return self._create_bert_backbone(word_embedding)
        elif self.config.init_pretrained == "modern_bert":
            return self._create_modern_bert_backbone(word_embedding)
        else:
            raise ValueError(f"Invalid init_pretrained value: {self.config.init_pretrained}")

    def _create_bert_backbone(
            self,
            word_embedding: nn.Embedding
    ) -> Tuple[nn.Module, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
        """Create BERT-style transformer backbone.

        :param word_embedding: Word embedding layer
        :type word_embedding: nn.Embedding
        :return: Tuple of (input_transformers, position_embeddings, layer_norm)
        :rtype: Tuple[nn.Module, Optional[nn.Embedding], Optional[nn.LayerNorm]]
        """
        if self.config.use_pretrained_weights:
            temp_bert = BertModel.from_pretrained(self.config.config_name, config=self.bert_config)
            with torch.no_grad():
                word_embedding.weight = temp_bert.embeddings.word_embeddings.weight
            input_transformers = temp_bert.encoder
            position_embeddings = temp_bert.embeddings.position_embeddings
            layer_norm = temp_bert.embeddings.LayerNorm
        else:
            input_transformers = BertEncoder(self.bert_config)
            position_embeddings = nn.Embedding(
                self.bert_config.max_position_embeddings,
                self.bert_config.hidden_size
            )
            layer_norm = nn.LayerNorm(
                self.bert_config.hidden_size,
                eps=self.bert_config.layer_norm_eps
            )

        return input_transformers, position_embeddings, layer_norm

    def _create_modern_bert_backbone(
            self,
            word_embedding: nn.Embedding
    ) -> Tuple[nn.Module, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
        """Create ModernBERT-style transformer backbone.

        :param word_embedding: Word embedding layer
        :type word_embedding: nn.Embedding
        :return: Tuple of (input_transformers, position_embeddings, layer_norm)
        :rtype: Tuple[nn.Module, Optional[nn.Embedding], Optional[nn.LayerNorm]]
        """
        if self.config.use_pretrained_weights:
            temp_bert = ModernBertModel.from_pretrained(
                self.config.config_name,
                config=self.bert_config,
                trust_remote_code=True
            )
            with torch.no_grad():
                word_embedding.weight = temp_bert.embeddings.weight
            input_transformers = temp_bert
        else:
            input_transformers = ModernBertModel(self.bert_config)

        return input_transformers, None, None

    def _create_output_projection(self) -> Optional[nn.Sequential]:
        """Create output projection if dimensions don't match.

        :return: Output projection network or None if not needed
        :rtype: Optional[nn.Sequential]
        """
        if self.config.output_dims != self.bert_config.hidden_size:
            return nn.Sequential(
                nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.bert_config.hidden_size, self.config.output_dims),
            )
        return None

    def _create_position_ids(self) -> Tensor:
        """Create position IDs tensor.

        :return: Position IDs tensor
        :rtype: Tensor
        """
        return torch.arange(self.bert_config.max_position_embeddings).expand((1, -1))
