import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from shortcutfm.config import DiTModelConfig


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Shortcuts                    #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        *,
        word_embedding: nn.Embedding,
        lm_head: nn.Linear,
        t_embedder: TimestepEmbedder,
        blocks: nn.ModuleList,
        pos_embed: nn.Parameter,
        shortcut_embedder: TimestepEmbedder,
        config=DiTModelConfig,
    ):
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding
        self.lm_head = lm_head
        self.t_embedder = t_embedder
        self.register_module("shortcut_embedder", shortcut_embedder)
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.num_heads = blocks[0].attn.num_heads if blocks else 8

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Apply initialization to all modules except word_embedding if it should be frozen
        if hasattr(self.config, "freeze_word_embedding") and self.config.freeze_word_embedding:
            # Skip word_embedding initialization
            for name, module in self.named_children():
                if name != "word_embedding":
                    module.apply(_basic_init)
        else:
            # Apply to all modules
            self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding if needed
        if hasattr(self.pos_embed, "shape"):
            # Use a reasonable grid size based on sequence length
            seq_len = self.pos_embed.shape[1]
            pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
            pos_embed_data = get_1d_sincos_pos_embed(self.config.embedding_dim, pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))

        # Initialize timestep embedding MLP if it exists
        if hasattr(self.t_embedder, "mlp") and len(self.t_embedder.mlp) >= 3:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            if hasattr(block, "adaLN_modulation") and len(block.adaLN_modulation) > 0:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def get_embeddings(self, input_ids):
        word_embeddings = self.word_embedding(input_ids)
        if hasattr(self.config, "normalize_word_embedding") and self.config.normalize_word_embedding:
            word_embeddings = word_embeddings / (word_embeddings.norm(dim=-1, keepdim=True) + 1e-10)
        return word_embeddings

    def compute_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, t, shortcut):
        """
        Forward pass of DiT.
        x: (N, T) tensor of token indices
        t: (N,) tensor of diffusion timesteps
        shortcut: (N,) tensor of shortcut values
        """
        x = x + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)
        c = t
        if self.shortcut_embedder is not None:
            shortcut = self.shortcut_embedder(shortcut)  # (N, D)
            c = c + shortcut  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        return x


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
