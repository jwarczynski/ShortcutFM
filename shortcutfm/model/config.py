from typing import Optional

from pydantic import BaseModel


# TODO: validation of config e.g. diffusion steps, shorcut and so on
class TransformerNetModelConfig(BaseModel):
    """Pydantic class for the TransformerNetModel configuration."""

    input_dims: int
    hidden_size: int
    output_dims: int
    hidden_t_dim: int
    diffusion_steps: int
    min_shortcut_size: int
    dropout: float = 0.0
    config_name: str = "bert-base-uncased"
    vocab_size: int
    init_pretrained: str = "no"  # "bert" or "no"
    logits_mode: int = 1
    sc_rate: float = 0.5
    predict_t: bool = False
    max_position_embeddings: Optional[int] = None  # Add max position embeddings
