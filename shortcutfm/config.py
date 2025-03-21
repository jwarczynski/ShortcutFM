import random
from lightning import seed_everything
from pathlib import Path
from typing import Literal, Optional, Union

import exca
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, FilePath, computed_field, field_validator


class EMAConfig(BaseModel):
    """EMA-specific configuration settings"""
    smoothing: float = Field(default=0.99, description="EMA smoothing factor")
    half_life: Optional[float] = Field(default=None, description="Half-life for EMA decay")
    update_interval: int = Field(default=1, description="How often to update EMA weights")
    model_config = ConfigDict(extra="forbid")  # Add this line


class WandBConfig(BaseModel):
    """Weights & Biases logging configuration"""
    project_name: str = Field(default="test", description="Project name for logging")
    run_name: Optional[str] = Field(default=None, description="Run name for logging")
    resume: str = Field(default="allow", description="WandB resume behavior")
    enabled: bool = Field(default=True, description="Whether to enable WandB logging")
    run_id: Optional[str] = Field(default=None, description="WandB run ID")
    model_config = ConfigDict(extra="forbid")  # Add this line


class CheckpointConfig(BaseModel):
    """Checkpoint configuration settings"""
    save_folder: str = Field(default="checkpoints", description="Directory to save checkpoints")
    save_interval: int = Field(default=100, description="How often to save checkpoints")
    num_to_keep: int = Field(default=-1, description="Number of checkpoints to keep (-1 for all)")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing checkpoints")
    save_last: bool = Field(default=True, description="Whether to save the last checkpoint")
    save_top_k: int = Field(default=-1, description="Number of top checkpoints to save (-1 for all)")
    monitor: Optional[str] = Field(default=None, description="Metric to monitor for checkpoint selection")
    mode: str = Field(default="min", description="Mode for checkpoint selection")
    path: Optional[FilePath] = Field(
        default=None,
        description="Path to checkpoint file to resume from. None means start from scratch"
    )
    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    """Model architecture and behavior configuration"""
    input_dims: int = Field(default=128, description="Input dimension size")
    output_dims: int = Field(default=128, description="Output dimension size")
    hidden_size: int = Field(default=768, description="Hidden layer dimension size")
    hidden_t_dim: int = Field(default=128, description="Hidden time embedding dimension")
    hidden_shortcut_dim: Optional[int] = Field(default=128, description="Hidden shortcut embedding dimension")
    projection_activation: Literal["gelu", "relu", "silu", "tanh"] = Field(
        default="gelu", description="Activation function for projection layers"
    )
    diffusion_steps: int = Field(default=2048, description="Number of diffusion steps")
    min_shortcut_size: int = Field(default=32, description="Minimum shortcut size")
    dropout: float = Field(default=0.1, description="Dropout rate")
    config_name: Literal["bert-base-uncased", "answerdotai/ModernBERT-base"] = Field(
        default="bert-base-uncased",
        description="Name of the base model configuration to use"
    )
    vocab_size: int = Field(default=30522, description="Size of the vocabulary")
    init_pretrained: Literal["bert", "modern_bert"] = Field(
        default="bert",
        description="Which model architecture to use: 'bert' for BERT, 'modern_bert' for ModernBERT"
    )
    use_pretrained_weights: bool = Field(
        default=False,
        description="Whether to use pretrained weights (True) or random initialization (False)"
    )
    logits_mode: int = Field(default=1, description="Mode for logits computation")
    sc_rate: float = Field(default=0.5, description="Self-conditioning rate")
    predict_t: bool = Field(default=False, description="Whether to predict timestep")
    max_position_embeddings: Optional[int] = Field(default=None, description="Maximum position embeddings")
    word_embedding_std: float = Field(default=1.0, description="Standard deviation for word embedding initialization")
    parametrization: Literal["x0", "velocity"] = Field(default="x0", description="Parametrization for diffusion")
    stacked_embeddings: bool = Field(default=False, description="Whether to stack embeddings")
    freeze_word_embedding: bool = Field(default=False, description="Whether to freeze word embeddings")
    model_config = ConfigDict(extra="forbid")


class BaseSchedulerConfig(BaseModel):
    """Base class for scheduler configurations"""
    lr: float = Field(default=3e-4, description="Target learning rate")
    weight_decay: float = Field(default=0.1, description="Weight decay factor")
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class MyleSchedulerConfig(BaseSchedulerConfig):
    """Configuration for Myle scheduler"""
    type: Literal["myle"]
    warmup_steps: int = Field(..., description="Number of warmup steps")
    start_lr: float = Field(..., description="Initial learning rate")
    model_config = ConfigDict(extra="forbid")


class LinearSchedulerConfig(BaseSchedulerConfig):
    """Configuration for Linear scheduler"""
    type: Literal["linear"]
    start_factor: float = Field(..., description="Start factor for linear scheduler")
    end_factor: float = Field(..., description="End factor for linear scheduler")
    total_steps: Optional[int] = Field(default=None, description="Total steps for linear scheduler")
    model_config = ConfigDict(extra="forbid")


# Define the scheduler type union with discriminator
SchedulerConfig = Union[MyleSchedulerConfig, LinearSchedulerConfig]


class OptimizerConfig(BaseModel):
    """Optimizer and learning rate scheduler configuration"""
    scheduler: SchedulerConfig = Field(..., description="Scheduler configuration", discriminator='type')

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class TrainingConfig(BaseModel):
    """Training process configuration"""
    # Data configuration
    batch_size: int = Field(default=256, description="Batch size for training")
    training_data_path: Path = Field(description="Path to training dataset")
    validation_data_path: Path = Field(description="Path to validation dataset")

    # Training process settings
    log_interval: int = Field(default=1, description="How often to log metrics")
    val_interval: Optional[int] = Field(default=None, description="How often to run validation")
    check_val_every_n_epoch: Optional[int] = Field(default=5, description="How often to run validation")
    self_consistency_ratio: float = Field(default=0.25, description="Self-consistency ratio")
    max_steps: int = Field(default=60000, description="Maximum training steps")
    reduce_fn: str = Field(default="mean", description="Reduce function")
    gradient_clipping: Optional[float] = Field(default=None, description="Gradient clipping value")
    accumulate_grad_batches: int = Field(default=8, description="Number of batches to accumulate gradients")
    deterministic: bool = Field(default=True, description="Whether to use deterministic training")
    seed: int = Field(default=44, description="Random seed")
    limit_train_batches: Optional[int] = Field(
        default=None,
        description="Number of training batches per epoch (-1 for all)"
    )
    limit_val_batches: Optional[int] = Field(
        default=None,
        description="Number of validation batches per epoch (-1 for all)"
    )
    overfit_batches: Optional[Union[int, float]] = Field(
        default=0.0,
        description="Number of batches to overfit on. Can be int (number of batches) or float (fraction of batches)"
    )

    # Denoising and logging settings
    denoising_step_size: int = Field(
        default=32,
        description="Step size used during denoising process when shortcut_size is 0 or None"
    )
    num_val_batches_to_log: int = Field(
        default=1,
        description="Number of validation batches to log predictions for in WandB"
    )
    num_timestep_bins: int = Field(
        default=4,
        description="Number of linearly spaced bins for tracking losses at different timesteps"
    )
    prediction_shortcut_size: int = Field(default=None, description="Shortcut size for prediction")
    log_train_predictions_every_n_epochs: int = Field(
        default=100,
        description="Number of epochs between train prediction logging"
    )
    log_train_predictions_from_n_epochs: int = Field(
        default=1000,
        description="Number of training epochs to start logging train predictions from"
    )

    # Loss weights
    flow_matching_loss_weight: Optional[float] = Field(default=1.0, description="Weight for flow matching loss")
    consistency_loss_weight: Optional[float] = Field(default=1.0, description="Weight for consistency loss")
    nll_loss_weight: Optional[float] = Field(default=1.0, description="Weight for negative log likelihood loss")
    isotropy_loss_weight: Optional[float] = Field(default=1.0, description="Weight for isotropy loss")

    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration")
    wandb: WandBConfig = Field(default_factory=WandBConfig, description="Weights & Biases configuration")
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig, description="Checkpoint configuration")
    ema: Optional[EMAConfig] = Field(default_factory=EMAConfig, description="EMA configuration")

    # Runtime settings
    use_exca: bool = Field(default=False, description="Whether to use Exca for submitting tasks")
    dry_run: bool = Field(default=False, description="Whether this is a dry run")
    use_composer: bool = Field(default=False, description="Whether to use Composer for training")

    infra: exca.TaskInfra = exca.TaskInfra()

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @infra.apply
    def train(self) -> None:
        from shortcutfm.train.pl.trainer import get_lightning_trainer

        seed_everything(self.seed)
        random.seed(self.seed)
        trainer, model, train_dataloader, val_dataloader = get_lightning_trainer(self)
        if not self.dry_run:
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=self.checkpoint.path)


class GenerationConfig(BaseModel):
    """Configuration for model generation/inference."""
    # Path to training config YAML and the loaded config
    training_config_path: str = Field(description="Path to training config YAML file")

    # Model checkpoint and weights
    checkpoint_path: str = Field(description="Path to model checkpoint")
    use_ema_weights: bool = Field(default=True, description="Whether to use EMA weights for generation")

    # Data and batch settings
    test_data_path: str = Field(description="Path to test dataset")
    batch_size: int = Field(default=32, description="Batch size for generation")
    limit_test_batches: Optional[Union[float, int]] = Field(
        default=None,
        description="None for full dataset, float for fraction, int for number of batches"
    )

    # Generation settings
    generation_shortcut_size: int = Field(default=1, description="Size of generation shortcut")
    seed: int = Field(default=44, description="Random seed for reproducibility")
    output_folder: str = Field(default="outputs", description="Folder to save generation outputs")

    model_config = ConfigDict(validate_assignment=True)

    @computed_field
    @property
    def training_config(self) -> TrainingConfig:
        """Load and return the training configuration."""
        if not Path(self.training_config_path).exists():
            raise ValueError(f"Training config file not found: {self.training_config_path}")

        with open(self.training_config_path, "r") as f:
            yaml_cfg = OmegaConf.load(f)

        return TrainingConfig(**OmegaConf.to_container(yaml_cfg, resolve=True))

    @field_validator('output_folder')
    @classmethod
    def modify_output_folder(cls, v: str, info) -> str:
        """Append seed value to the output folder path and ensure uniqueness"""
        if v is None:
            raise ValueError("output_folder must be specified")

        base_path = Path(v)
        seed_str = f"seed_{info.data['seed']}"
        path = base_path / seed_str

        # Find a unique path by appending a number if needed
        counter = 1
        while path.exists():
            path = base_path / f"{seed_str}_v{counter}"
            counter += 1

        # Create the unique output directory
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
