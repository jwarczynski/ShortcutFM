from dataclasses import dataclass

import torch
from torch import nn

from shortcutfm.config import DiTModelConfig
from shortcutfm.model.DiT import DiT, DiTBlock, TimestepEmbedder
from shortcutfm.model.model import FlowMatchingModel


@dataclass
class DiTModules:
    """Dataclass to hold the modules for the DiT model."""

    word_embedding: nn.Embedding
    lm_head: nn.Linear
    t_embedder: TimestepEmbedder
    blocks: nn.ModuleList
    pos_embed: nn.Parameter
    shortcut_embedder: TimestepEmbedder | None = None


class DiTFactory:
    """Factory class to create DiT model instances from configuration."""

    def __init__(self, config: DiTModelConfig):
        self.config = config

    def build(self) -> FlowMatchingModel:
        """Builds and returns a DiT model wrapped in FlowMatchingModel.

        :return: Configured FlowMatchingModel instance with DiT as the module
        :rtype: FlowMatchingModel
        """
        modules = self._create_modules()
        module = self._create_dit_model(modules)

        return FlowMatchingModel(
            module=module,
            diffusion_steps=self.config.diffusion_steps,
            min_shortcut_size=self.config.min_shortcut_size,
            scale_time=self.config.scale_time,
        )

    def _create_modules(self) -> DiTModules:
        """Creates all necessary modules for the DiT model.

        :return: Dataclass containing all model modules
        :rtype: DiTModules
        """
        # Create word embeddings and lm_head
        word_embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        lm_head = nn.Linear(self.config.embedding_dim, self.config.vocab_size, bias=True)

        # Handle weight freezing and tying
        if self.config.freeze_word_embedding:
            word_embedding.weight.requires_grad = False
            lm_head.weight.requires_grad = True
            # Independent weights: copy word_embedding weights to lm_head
            with torch.no_grad():
                lm_head.weight.copy_(word_embedding.weight)
            print(f"DiT: word embedding requires grad: {word_embedding.weight.requires_grad}")
            print(f"DiT: lm head requires grad: {lm_head.weight.requires_grad}")
        else:
            # Shared weights: tie lm_head weights to word_embedding for efficiency
            with torch.no_grad():
                lm_head.weight = word_embedding.weight

        # Create time and shortcut embedders
        t_embedder = TimestepEmbedder(self.config.embedding_dim, self.config.hidden_t_dim)
        shortcut_embedder = None
        if self.config.hidden_shortcut_dim is not None:
            shortcut_embedder = TimestepEmbedder(self.config.embedding_dim, self.config.hidden_shortcut_dim)

        # Create position embeddings
        seq_len = self.config.max_position_embeddings or 128
        pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.config.embedding_dim), requires_grad=False)

        # Create transformer blocks
        num_heads = self.config.num_attention_heads
        depth = self.config.num_layers or 12
        mlp_ratio = self.config.mlp_ratio
        blocks = nn.ModuleList(
            [DiTBlock(self.config.embedding_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        return DiTModules(
            word_embedding=word_embedding,
            lm_head=lm_head,
            t_embedder=t_embedder,
            blocks=blocks,
            pos_embed=pos_embed,
            shortcut_embedder=shortcut_embedder,
        )

    def _create_dit_model(self, modules: DiTModules) -> DiT:
        """Creates a DiT model from the provided modules.

        :param modules: Dataclass containing all model modules
        :type modules: DiTModules
        :return: Configured DiT model
        :rtype: DiT
        """
        return DiT(
            word_embedding=modules.word_embedding,
            lm_head=modules.lm_head,
            t_embedder=modules.t_embedder,
            blocks=modules.blocks,
            pos_embed=modules.pos_embed,
            shortcut_embedder=modules.shortcut_embedder,
            config=self.config,
        )
