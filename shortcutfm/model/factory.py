from dataclasses import dataclass
from typing import Optional, Tuple, override

import torch
from torch import Tensor, nn
from torch.nn import SiLU
from transformers import AutoConfig, BertModel, ModernBertModel
from transformers.models.bert.modeling_bert import BertEncoder

from shortcutfm.config import ModelConfig
from shortcutfm.model.model import (
    BackboneTransformer,
    BertEncoderBackbone,
    FFNBackbone, FlowMatchingModel,
    ModernBertBackbone,
    StackedEmbeddingTransformerNetModel, TransformerNetModel,
)


@dataclass
class TransformerNetModelModules:
    """Dataclass to hold the modules for the TransformerNetModel."""

    word_embedding: nn.Embedding
    lm_head: nn.Linear
    time_embed: nn.Sequential
    backbone_transformer: nn.Module  # Can be BertEncoder or BertModel.encoder
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
        module = self.create_module(modules)

        return FlowMatchingModel(
            module=module,
            diffusion_steps=self.config.diffusion_steps,
            min_shortcut_size=self.config.min_shortcut_size
        )

    def create_module(self, modules):
        module = TransformerNetModel(**modules.__dict__, config=self.config)
        return module

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
        backbone_transformer, position_embeddings, layer_norm = self._create_transformer_backbone(word_embedding)
        if self.config.freeze_word_embedding:
            word_embedding.weight.requires_grad = False
            lm_head.weight.requires_grad = True
            print(f"word emebedding reuires grad: {word_embedding.weight.requires_grad}")
            print(f"lm head requires grad: {lm_head.weight.requires_grad}")

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
            backbone_transformer=backbone_transformer,
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
        vocab_size = self.config.vocab_size

        # Create word embedding layer
        word_embedding = nn.Embedding(vocab_size, input_dims)
        nn.init.normal_(word_embedding.weight, mean=0.0, std=self.config.word_embedding_std)

        # Create lm_head with conditional weight sharing
        lm_head = nn.Linear(input_dims, vocab_size, bias=True)
        with torch.no_grad():
            if self.config.freeze_word_embedding:
                # Independent weights: copy word_embedding weights to lm_head
                lm_head.weight.copy_(word_embedding.weight)
            else:
                # Shared weights: tie lm_head weights to word_embedding for efficiency
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
        activation = self.create_activation(self.config.projection_activation)
        input_dims = self.config.input_dims * (2 if self.config.sc_rate > 0 else 1)
        if input_dims != self.bert_config.hidden_size:
            return nn.Sequential(
                nn.Linear(input_dims, self.bert_config.hidden_size),
                activation,
                nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
            )
        return None

    def _create_transformer_backbone(
            self,
            word_embedding: nn.Embedding
    ) -> Tuple[BackboneTransformer, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
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
    ) -> Tuple[BertEncoderBackbone, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
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
            backbone_trasnformer = BertEncoderBackbone(input_transformers)
        else:
            input_transformers = BertEncoder(self.bert_config)
            backbone_trasnformer = BertEncoderBackbone(input_transformers)
            position_embeddings = nn.Embedding(
                self.bert_config.max_position_embeddings,
                self.bert_config.hidden_size
            )
            layer_norm = nn.LayerNorm(
                self.bert_config.hidden_size,
                eps=self.bert_config.layer_norm_eps
            )

        return backbone_trasnformer, position_embeddings, layer_norm

    def _create_modern_bert_backbone(
            self,
            word_embedding: nn.Embedding
    ) -> Tuple[ModernBertBackbone, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
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
            backbone_trasnformer = ModernBertBackbone(input_transformers)
        else:
            input_transformers = ModernBertModel(self.bert_config)
            backbone_trasnformer = ModernBertBackbone(input_transformers)

        return backbone_trasnformer, None, None

    def _create_output_projection(self) -> Optional[nn.Sequential]:
        """Create output projection if dimensions don't match.

        :return: Output projection network or None if not needed
        :rtype: Optional[nn.Sequential]
        """
        activation = self.create_activation(self.config.projection_activation)
        if self.config.output_dims != self.bert_config.hidden_size:
            return nn.Sequential(
                nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
                activation,
                nn.Linear(self.bert_config.hidden_size, self.config.output_dims),
            )
        return None

    def create_activation(self, activation: str) -> nn.Module:
        """Create activation function based on configuration.

        :param activation: Activation function name
        :type activation: str
        :return: Activation function module
        :rtype: nn.Module
        """
        if activation.lower() == "silu" or activation.lower() == "swish":  # Handle both names
            return nn.SiLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def _create_position_ids(self) -> Tensor:
        """Create position IDs tensor.

        :return: Position IDs tensor
        :rtype: Tensor
        """
        return torch.arange(self.bert_config.max_position_embeddings).expand((1, -1))


class StackedEmbeddingTransformerNetModelFactory(TransformerNetModelFactory):
    """Factory class to create TransformerNetModel instances with stacked embeddings from configuration."""

    @override
    def create_module(self, modules):
        module = StackedEmbeddingTransformerNetModel(**modules.__dict__, config=self.config)
        return module

    @override
    def _create_input_projection(self) -> Optional[nn.Sequential]:
        """Create input projection if dimensions don't match.

        :return: Input projection network or None if not needed
        :rtype: Optional[nn.Sequential]
        """
        input_dims = (
                (self.config.input_dims * 2 if self.config.sc_rate > 0 else self.config.input_dims)
                + self.config.hidden_t_dim
                + (self.config.hidden_shortcut_dim if self.config.hidden_shortcut_dim is not None else 0)
        )

        return nn.Sequential(
            nn.Linear(input_dims, self.bert_config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
        )

    @override
    def _create_time_embedding(self, input_dim, scale_factor: int = 4) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.config.hidden_t_dim, self.config.hidden_t_dim * scale_factor),
            nn.Tanh(),
            nn.Linear(self.config.hidden_t_dim * scale_factor, self.config.hidden_t_dim),
        )

    @override
    def _create_shortcut_embedding(self):
        return nn.Sequential(
            nn.Linear(self.config.hidden_shortcut_dim, self.config.hidden_shortcut_dim * 4),
            nn.Tanh(),
            nn.Linear(self.config.hidden_shortcut_dim * 4, self.config.hidden_shortcut_dim),
        )

    @override
    def _create_bert_backbone(
            self,
            word_embedding: nn.Embedding
    ) -> Tuple[BertEncoderBackbone, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
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
            backbone_trasnformer = BertEncoderBackbone(input_transformers)
        else:
            input_transformers = BertEncoder(self.bert_config)
            backbone_trasnformer = BertEncoderBackbone(input_transformers)
            input_dims = self.config.input_dims * 2 if self.config.sc_rate > 0 else self.config.input_dims
            position_embeddings = nn.Embedding(
                self.bert_config.max_position_embeddings,
                input_dims
            )
            layer_norm = nn.LayerNorm(
                self.bert_config.hidden_size,
                eps=self.bert_config.layer_norm_eps
            )

        return backbone_trasnformer, position_embeddings, layer_norm


class FFNFactory(TransformerNetModelFactory):
    """Factory class to create TransformerNetModel instances with FFN."""

    @override
    def _create_transformer_backbone(
            self,
            word_embedding: nn.Embedding
    ) -> Tuple[BackboneTransformer, Optional[nn.Embedding], Optional[nn.LayerNorm]]:
        """Create transformer backbone based on configuration.

        :param word_embedding: Word embedding layer
        :type word_embedding: nn.Embedding
        :return: Tuple of (input_transformers, position_embeddings, layer_norm)
        :rtype: Tuple[nn.Module, Optional[nn.Embedding], Optional[nn.LayerNorm]]
        """

        ffn = FFNBackbone(
            768,
            768,
            1,
        )

        return ffn, None, None


