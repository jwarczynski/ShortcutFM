import random
from pathlib import Path
from typing import Literal

import exca
import torch
from lightning import seed_everything
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    computed_field,
    field_validator,
)


class EMAConfig(BaseModel):
    """EMA-specific configuration settings"""

    smoothing: float = Field(default=0.99, description="EMA smoothing factor")
    half_life: float | None = Field(default=None, description="Half-life for EMA decay")
    update_interval: int = Field(default=1, description="How often to update EMA weights")
    model_config = ConfigDict(extra="forbid")  # Add this line


class WandBConfig(BaseModel):
    """Weights & Biases logging configuration"""

    project_name: str = Field(default="test", description="Project name for logging")
    run_name: str | None = Field(default=None, description="Run name for logging")
    resume: str = Field(default="allow", description="WandB resume behavior")
    enabled: bool = Field(default=True, description="Whether to enable WandB logging")
    run_id: str | None = Field(default=None, description="WandB run ID")
    model_config = ConfigDict(extra="forbid")  # Add this line


class CheckpointConfig(BaseModel):
    """Checkpoint configuration settings"""

    save_folder: str = Field(default="checkpoints", description="Directory to save checkpoints")
    save_interval: int = Field(default=1000, description="How often to save checkpoints")
    num_to_keep: int = Field(default=15, description="Number of checkpoints to keep (-1 for all)")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing checkpoints")
    save_last: bool = Field(default=True, description="Whether to save the last checkpoint")
    save_top_k: int = Field(default=-1, description="Number of top checkpoints to save (-1 for all)")
    monitor: str | None = Field(default=None, description="Metric to monitor for checkpoint selection")
    mode: str = Field(default="min", description="Mode for checkpoint selection")
    path: FilePath | None = Field(
        default=None,
        description="Path to checkpoint file to resume from. None means start from scratch",
    )
    enabled: bool = Field(default=True, description="Whether to enable checkpointing")
    model_config = ConfigDict(extra="forbid")


class BaseModelConfig(BaseModel):
    """Base model configuration with common parameters for all architectures"""

    # Common parameters for all architectures
    # type: str  # Discriminator field
    input_dims: int = Field(default=128, description="Input dimension size")
    output_dims: int = Field(default=128, description="Output dimension size")
    hidden_size: int = Field(default=768, description="Hidden layer dimension size")
    hidden_t_dim: int | None = Field(default=128, description="Hidden time embedding dimension")
    hidden_shortcut_dim: int | None = Field(default=128, description="Hidden shortcut embedding dimension")
    diffusion_steps: int = Field(default=2048, description="Number of diffusion steps")
    min_shortcut_size: int = Field(default=1, description="Minimum shortcut size")
    dropout: float = Field(default=0.1, description="Dropout rate")
    tokenizer_config_name: Literal["bert-base-uncased", "answerdotai/ModernBERT-base", "Helsinki-NLP/opus-mt-en-de"] = Field(
        default="bert-base-uncased",
        description="Name of the tokenizer configuration to use",
    )
    config_name: Literal["bert-base-uncased", "answerdotai/ModernBERT-base"] = Field(
        default="bert-base-uncased",
        description="Name of the base model configuration to use",
    )
    vocab_size: int = Field(default=30522, description="Size of the vocabulary")
    null_token_id: int = Field(default=15, description="ID of the null token used for classifier-free guidance")
    sc_rate: float = Field(default=0.5, description="Self-conditioning rate")
    max_position_embeddings: int | None = Field(default=None, description="Maximum position embeddings")
    word_embedding_std: float = Field(default=1.0, description="Standard deviation for word embedding initialization")
    parametrization: Literal["x0", "velocity"] = Field(default="x0", description="Parametrization for diffusion")
    freeze_word_embedding: bool = Field(default=False, description="Whether to freeze word embeddings")
    freeze_lm_head: bool = Field(
        default=False,
        description="Whether to freeze language model head (if shared with embeddings)."
        "If both freeze_word_embedding and freeze_lm_head are True, parameters are shared and both are frozen.",
    )
    tie_word_embedding: bool = Field(
        default=True,
        description="Whether to tie word embeddings with language model head weights"
        " (if true freeze_word_embedding and ).",
    )
    use_pretrained_embeddings: bool = Field(
        default=False, description="Whether to use only pretrained word embeddings (not full model weights)"
    )
    normalize_word_embedding: bool = Field(default=False, description="Whether to normalize word embeddings")
    scale_time: bool = Field(
        default=False,
        description="Whether to scale time and shortcut embeddings by the diffusion steps",
    )
    default_shortcut: Literal["0", "t"] = Field(default="t", description="Default shortcut for flow matching loss")
    use_default_t_for_shortcut: bool = Field(
        default=False,
        description="If True, use default shortcut instead of shortcut_size"
        "for step1/step2 prediction in consistency criterion",
    )
    model_config = ConfigDict(extra="forbid")


class TransformerModelConfig(BaseModelConfig):
    """Configuration for Transformer architecture"""

    type: Literal["transformer"] = "transformer"
    use_pretrained_weights: bool = Field(
        default=False,
        description="Whether to use pretrained weights (True) or random initialization (False)",
    )
    logits_mode: int = Field(default=1, description="Mode for logits computation")
    predict_t: bool = Field(default=False, description="Whether to predict timestep")
    projection_activation: Literal["gelu", "relu", "silu", "tanh"] = Field(
        default="gelu", description="Activation function for projection layers"
    )
    model_config = ConfigDict(extra="forbid")


class StackedModelConfig(BaseModelConfig):
    """Configuration for Stacked architecture"""

    type: Literal["stacked"] = "stacked"
    use_pretrained_weights: bool = Field(
        default=False,
        description="Whether to use pretrained weights (True) or random initialization (False)",
    )
    logits_mode: int = Field(default=1, description="Mode for logits computation")
    predict_t: bool = Field(default=False, description="Whether to predict timestep")
    projection_activation: Literal["gelu", "relu", "silu", "tanh"] = Field(
        default="gelu", description="Activation function for projection layers"
    )
    model_config = ConfigDict(extra="forbid")


class FFNModelConfig(BaseModelConfig):
    """Configuration for FFN architecture"""

    type: Literal["ffn"] = "ffn"
    use_pretrained_weights: bool = Field(
        default=False,
        description="Whether to use pretrained weights (True) or random initialization (False)",
    )
    num_layers: int = Field(
        default=3,
        description="Number of ffn transformer layers",
    )
    model_config = ConfigDict(extra="forbid")


class DiTModelConfig(BaseModelConfig):
    """Configuration for DiT architecture"""

    type: Literal["dit"] = "dit"
    embedding_dim: int = Field(default=128, description="Embedding dimension for DiT model")
    num_attention_heads: int = Field(default=8, description="Number of attention heads for DiT model")
    mlp_ratio: float = Field(default=4.0, description="MLP ratio for DiT model")
    num_layers: int = Field(default=12, description="Number of DiT transformer layers")
    model_config = ConfigDict(extra="forbid")


class ShortcutTokenModelConfig(BaseModelConfig):
    """Configuration for architecture that handles shortcuts as separate tokens"""

    type: Literal["shortcut_token"] = "shortcut_token"
    use_pretrained_weights: bool = Field(
        default=False,
        description="Whether to use pretrained weights (True) or random initialization (False)",
    )
    logits_mode: int = Field(default=1, description="Mode for logits computation")
    predict_t: bool = Field(default=False, description="Whether to predict timestep")
    projection_activation: Literal["gelu", "relu", "silu", "tanh"] = Field(
        default="gelu", description="Activation function for projection layers"
    )
    model_config = ConfigDict(extra="forbid")


class MaskedDiffusionModelConfig(BaseModelConfig):
    """Configuration for masked (absorbing-state) discrete diffusion architecture"""

    type: Literal["masked_diffusion"] = "masked_diffusion"
    use_pretrained_weights: bool = Field(
        default=False,
        description="Whether to use pretrained weights (True) or random initialization (False)",
    )
    logits_mode: int = Field(default=1, description="Mode for logits computation")
    predict_t: bool = Field(default=False, description="Whether to predict timestep")
    projection_activation: Literal["gelu", "relu", "silu", "tanh"] = Field(
        default="gelu", description="Activation function for projection layers"
    )
    mask_token_id: int = Field(default=103, description="ID of the [MASK] token used for corruption")
    ce_importance_weighting: Literal["one_over_t", "none"] = Field(
        default="one_over_t",
        description="Importance weighting of the masked cross-entropy loss (MDLM/LLaDA uses 1/t)",
    )
    consistency_loss_space: Literal["l2_embedding", "kl"] = Field(
        default="l2_embedding",
        description="Space in which the shortcut consistency loss is computed:"
        " L2 between expected embeddings or KL between token distributions",
    )
    unmask_strategy: Literal["random", "confidence"] = Field(
        default="random",
        description="How to choose which masked positions to unmask when composing"
        " two shortcut steps and during inference",
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("sc_rate")
    @classmethod
    def _validate_no_self_conditioning(cls, v: float) -> float:
        if v != 0.0:
            raise ValueError("Self-conditioning (sc_rate > 0) is not supported for masked_diffusion")
        return v


# Define the model config union with discriminator
ModelConfig = (
    TransformerModelConfig
    | StackedModelConfig
    | FFNModelConfig
    | DiTModelConfig
    | ShortcutTokenModelConfig
    | MaskedDiffusionModelConfig
)


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
    total_steps: int | None = Field(default=None, description="Total steps for linear scheduler")
    model_config = ConfigDict(extra="forbid")


# Define the scheduler type union with discriminator
SchedulerConfig = MyleSchedulerConfig | LinearSchedulerConfig


class OptimizerConfig(BaseModel):
    """Optimizer and learning rate scheduler configuration"""

    scheduler: SchedulerConfig = Field(..., description="Scheduler configuration", discriminator="type")

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class PaddingStrategyConfig(BaseModel):
    mark_first_padding: bool = Field(default=True, description="Whether to mark the first padding token")
    mark_second_padding: bool = Field(default=True, description="Whether to mark the second padding token")
    model_config = ConfigDict(extra="forbid")


class MVFLossConfig(BaseModel):
    """Configuration for von Mises-Fisher loss"""

    regularization_type: Literal["norm_penalized", "dot_product_scaled", "cosine_penalized"] = Field(
        default="norm_penalized", description="Type of von Mises-Fisher loss to use"
    )
    lambda_1: float = Field(default=0.02, description="Regularization parameter for NormPenalizedVMFLoss")
    lambda_2: float = Field(default=0.1, description="Regularization parameter for DotProductScaledVMFLoss")
    cosine_threshold: float = Field(default=0.2, description="Cosine threshold for CosinePenalizedVMFLoss")
    cosine_penalty_scale: float = Field(default=1.0, description="Cosine threshold for CosinePenalizedVMFLoss")
    model_config = ConfigDict(extra="forbid")


class LossConfig(BaseModel):
    """Loss function configuration"""

    type: Literal["vmf", "mse"] = Field(default="mse", description="Type of loss function")
    mvf_loss_config: MVFLossConfig | None = Field(
        default_factory=MVFLossConfig,
        description="Configuration for von Mises-Fisher loss",
    )
    model_config = ConfigDict(extra="forbid")


class TimeShortcutConfig(BaseModel):
    """Configuration for time shortcut"""

    type: Literal["timestep_first", "shortcut_first"] = Field(
        default="timestep_first", description="Type of time shortcut sampling"
    )
    shortcut_sampler: Literal["uniform", "loss_aware", "all_max"] = Field(
        default="uniform", description="Type of shortcut sampling strategy"
    )
    time_sampler: Literal["uniform", "loss_aware", "all_max"] = Field(
        default="uniform", description="Type of time sampling strategy"
    )
    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    """Training process configuration"""

    num_gpus: int | str = Field(default="auto", description="Number of GPUs to use")

    # Data configuration
    batch_size: int = Field(default=256, description="Batch size for training")
    training_data_path: Path = Field(description="Path to training dataset")
    validation_data_path: Path = Field(description="Path to validation dataset")
    padding_strategy: PaddingStrategyConfig = Field(
        default_factory=PaddingStrategyConfig,
        description="Configuration for padding strategy",
    )

    # Training process settings
    log_interval: int = Field(default=1, description="How often to log metrics")
    val_interval: int | None = Field(default=None, description="How often to run validation")
    check_val_every_n_epoch: int | None = Field(default=5, description="How often to run validation")
    self_consistency_ratio: float = Field(default=0.25, description="Self-consistency ratio")
    consistency_start_step: int = Field(default=0, description="Global step to start consistency training")

    # Classifier-free guidance settings
    cfg_guidance_scale: float = Field(
        default=1.0, description="Scale factor for classifier-free guidance (1.0 means no guidance)"
    )
    cfg_start_step: int | None = Field(
        default=None, description="Global step to start applying classifier-free guidance"
    )
    cfg_probability: float = Field(
        default=0.5, description="Probability of training unconditionally (discarding conditioning)"
    )

    shortcut_target_x_start_probability: float = Field(
        default=0.0,
        description="Probability of using x_start or ground truth velocity as shortcut target"
        " in consistency criterion (0.5 for half probability)",
    )

    max_steps: int = Field(default=60000, description="Maximum training steps")
    reduce_fn: str = Field(default="mean", description="Reduce function")
    gradient_clipping: float | None = Field(default=None, description="Gradient clipping value")
    accumulate_grad_batches: int = Field(default=8, description="Number of batches to accumulate gradients")
    deterministic: bool = Field(default=True, description="Whether to use deterministic training")
    seed: int = Field(default=44, description="Random seed")
    limit_train_batches: int | None = Field(
        default=None, description="Number of training batches per epoch (-1 for all)"
    )
    limit_val_batches: int | None = Field(
        default=None, description="Number of validation batches per epoch (-1 for all)"
    )
    overfit_batches: int | float | None = Field(
        default=0.0,
        description="Number of batches to overfit on. Can be int (number of batches) or float (fraction of batches)",
    )

    # Denoising and logging settings
    denoising_step_size: int = Field(
        default=32,
        description="Step size used during denoising process when shortcut_size is 0 or None",
    )
    prediction_shortcut_size: int = Field(default=None, description="Shortcut size for prediction")

    num_val_batches_to_log: int = Field(
        default=1,
        description="Number of validation batches to log predictions for in WandB",
    )
    num_timestep_bins: int = Field(
        default=4,
        description="Number of linearly spaced bins for tracking losses at different timesteps",
    )
    log_train_predictions_every_n_epochs: int = Field(
        default=100, description="Number of epochs between train prediction logging"
    )
    log_train_predictions_from_n_epochs: int = Field(
        default=1000,
        description="Number of training epochs to start logging train predictions from",
    )
    normalize_embeddings: bool = Field(
        default=False,
        description="Whether to normalize word embeddings and language model head weights to lie on a hypersphere"
        "after each optimizer step",
    )

    # time and shortcut sampling
    time_shortcut_sampling: TimeShortcutConfig = Field(
        default_factory=TimeShortcutConfig,
        description="Configuration for time and shortcut sampling",
    )

    # Loss configuration
    loss: LossConfig = Field(default_factory=LossConfig, description="Loss function configuration")

    # Loss weights
    flow_matching_loss_weight: float | None = Field(default=1.0, description="Weight for flow matching loss")
    consistency_loss_weight: float | None = Field(default=1.0, description="Weight for consistency loss")
    nll_loss_weight: float | None = Field(default=1.0, description="Weight for negative log likelihood loss")
    isotropy_loss_weight: float | None = Field(default=1.0, description="Weight for isotropy loss")
    normalize_flow_matching_loss: bool = Field(default=False, description="Whether to normalize flow matching loss")

    # Component configurations
    model: ModelConfig = Field(..., description="Model configuration", discriminator="type")
    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration")
    wandb: WandBConfig = Field(default_factory=WandBConfig, description="Weights & Biases configuration")
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig, description="Checkpoint configuration")
    ema: EMAConfig | None = Field(default_factory=EMAConfig, description="EMA configuration")

    # Runtime settings
    use_exca: bool = Field(default=False, description="Whether to use Exca for submitting tasks")
    dry_run: bool = Field(default=False, description="Whether this is a dry run")
    use_composer: bool = Field(default=False, description="Whether to use Composer for training")

    # Environment variables for distributed training
    environment_variables: dict[str, str] = Field(
        default_factory=lambda: {
            "NCCL_TIMEOUT": "300",
            "NCCL_BLOCKING_WAIT": "1",
            "NCCL_DEBUG": "INFO",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_MULTIPROCESSING_SHARING_STRATEGY": "file_system",
        },
        description="Environment variables to set for distributed training"
    )

    infra: exca.TaskInfra = exca.TaskInfra()

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @infra.apply
    def train(self) -> None:
        import os

        from shortcutfm.train.pl.trainer import get_lightning_trainer
        from shortcutfm.utils.logging_utils import configure_logging_for_slurm

        # Configure logging for SLURM/EXCA jobs first
        configure_logging_for_slurm()

        # Set environment variables for distributed training from configuration
        for key, value in self.environment_variables.items():
            os.environ[key] = value

        # Set environment variables for distributed training from configuration
        for key, value in self.environment_variables.items():
            os.environ[key] = value

        seed_everything(self.seed)
        random.seed(self.seed)
        trainer, model, train_dataloader, val_dataloader = get_lightning_trainer(self)
        if not self.dry_run:
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=self.checkpoint.path)


class PlotAnalysisConfig(BaseModel):
    """Configuration for cosine similarity and velocity analysis."""

    # Primary analysis focus - cosine similarity with/without ground truth interpolation
    run_cosine_analysis: bool = Field(default=True, description="Whether to run cosine similarity analysis")

    # Legacy analysis options (can be disabled for focus on cosine analysis)
    run_token_analysis: bool = Field(default=True, description="Whether to run token analysis")
    run_embedding_analysis: bool = Field(default=True, description="Whether to run embedding analysis")
    run_interpolation_analysis: bool = Field(default=True, description="Whether to run interpolation analysis")
    run_quality_analysis: bool = Field(default=True, description="Whether to run quality analysis")

    # Analysis parameters
    num_examples: int = Field(default=3, description="Number of examples to analyze in detail")
    top_k_tokens: int = Field(default=5, description="Number of top tokens to track in analysis")

    # Ground truth interpolation comparison (main focus)
    use_ground_truth_embeddings: list[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether to test with/without ground truth embeddings in cosine similarity analysis"
    )

    # Parameter testing
    analysis_shortcut_sizes: list[int] = Field(
        default_factory=lambda: [256, 512, 1024, 2048],
        description="Shortcut sizes to test in quality analysis"
    )
    analysis_step_sizes: list[int] = Field(
        default_factory=lambda: [256, 512, 1024, 2048],
        description="Step sizes to test in quality analysis"
    )

    # Embedding analysis parameters
    embedding_k: int = Field(default=100, description="Number of top frequent tokens for embedding analysis")

    # Plot formatting
    figure_size: tuple[int, int] = Field(default=(10, 8), description="Figure size for plots")
    save_format: str = Field(default="png", description="Format to save plots in")
    dpi: int = Field(default=300, description="DPI for saved plots")

    model_config = ConfigDict(extra="forbid")


class GenerationConfig(BaseModel):
    """Configuration for model generation/inference."""

    # Path to training config YAML and the loaded config
    training_config_path: str | None = Field(
        default=None,
        description="Path to training config YAML file. Required unless checkpoint_list_file is provided"
    )

    # Model checkpoint and weights
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to model checkpoint. Required unless checkpoint_list_file is provided"
    )
    checkpoint_list_file: str | None = Field(
        default=None,
        description="Path to file containing list of checkpoints to process. Alternative to individual checkpoint_path"
    )
    use_ema_weights: bool = Field(default=True, description="Whether to use EMA weights for generation")

    # Data and batch settings
    test_data_path: str | None = Field(
        default=None,
        description="Path to test dataset. If None, will be auto-determined from training_config_path and split"
    )
    split: Literal["test", "valid"] = Field(
        default="test",
        description="Which dataset split to use for generation (test or valid)"
    )
    batch_size: int = Field(default=32, description="Batch size for generation")
    limit_test_batches: float | int | None = Field(
        default=None,
        description="None for full dataset, float for fraction, int for number of batches",
    )

    # Generation settings
    generation_shortcut_size: int = Field(
        default=1,
        description="Shorcut value passed to the model in each denoising step."
        "Do no influence number of denoising steps.",
    )
    denoising_step_size: int = Field(
        default=1, description="Determines the number of denoising steps. Do not influence the shortcut size."
    )
    seed: int = Field(default=44, description="Random seed for reproducibility")
    output_folder: str = Field(default="outputs", description="Folder to save generation outputs")
    generation_suffix: str = Field(default="", description="Suffix to append to generation output files")

    # Evaluation settings
    run_evaluation: bool = Field(default=True, description="Whether to run evaluation after generation")
    evaluation_device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use for evaluation (e.g., BERTScore computation)",
    )
    use_fallback_processing: bool = Field(
        default=False,
        description="Whether to use fallback processing for empty predictions during evaluation"
    )

    # Plot analysis settings
    run_plot_analysis: bool = Field(
        default=True,
        description="Whether to run comprehensive plot analysis after generation"
    )
    plot_analysis_config: PlotAnalysisConfig = Field(
        default_factory=PlotAnalysisConfig,
        description="Configuration for plot analysis"
    )

    # Runtime settings
    use_exca: bool = Field(default=False, description="Whether to use Exca for submitting generation tasks")
    force_regeneration: bool = Field(
        default=False,
        description="Whether to force regeneration even if metrics files already exist"
    )

    # Infrastructure for exca job submission
    infra: exca.TaskInfra = exca.TaskInfra()

    model_config = ConfigDict(validate_assignment=True)

    def model_dump(self, **kwargs):
        """Override model_dump to exclude computed fields for exca compatibility."""
        # Set exclude to exclude computed fields
        if 'exclude' not in kwargs:
            kwargs['exclude'] = set()
        elif not isinstance(kwargs['exclude'], set):
            kwargs['exclude'] = set(kwargs['exclude']) if kwargs['exclude'] else set()

        kwargs['exclude'].add('training_config')
        kwargs['exclude'].add('effective_test_data_path')
        return super().model_dump(**kwargs)

    @computed_field
    @property
    def training_config(self) -> TrainingConfig | None:
        """Load and return the training configuration."""
        if self.training_config_path is None:
            return None

        if not Path(self.training_config_path).exists():
            raise ValueError(f"Training config file not found: {self.training_config_path}")

        with open(self.training_config_path) as f:
            yaml_cfg = OmegaConf.load(f)

        return TrainingConfig(**OmegaConf.to_container(yaml_cfg, resolve=True)) # type: ignore

    @computed_field
    @property
    def effective_test_data_path(self) -> str | None:
        """Get the effective test data path, either explicit or auto-determined."""
        if self.test_data_path is not None:
            return self.test_data_path

        if self.training_config_path is None:
            return None

        # Auto-determine from training config and split
        from shortcutfm.decoding.generation_runner import determine_test_data_path
        return determine_test_data_path(self.training_config_path, self.split)

    @field_validator("output_folder")
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
        while path.exists() and any(path.iterdir()):  # Check if directory exists AND has content
            path = base_path / f"{seed_str}_v{counter}"
            counter += 1

        # Don't create the directory here - let it be created when actually needed
        # This prevents empty directories from being created
        return str(path)

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        """Validate that split is either 'test' or 'valid'."""
        if v not in ["test", "valid"]:
            raise ValueError("split must be either 'test' or 'valid'")
        return v

    def model_post_init(self, __context) -> None:
        """Validate that either individual checkpoint fields or checkpoint_list_file is provided."""
        has_individual = self.training_config_path is not None and self.checkpoint_path is not None
        has_list = self.checkpoint_list_file is not None

        if not has_individual and not has_list:
            raise ValueError(
                "Either provide training_config_path + checkpoint_path OR checkpoint_list_file"
            )

        if has_individual and has_list:
            # Both provided - checkpoint_list_file takes precedence, ignore individual fields
            pass

    @infra.apply
    def generate(self) -> None:
        """Run model generation/inference with this configuration."""
        from shortcutfm.decoding.generation_runner import run_generation_with_evaluation
        from shortcutfm.utils.logging_utils import configure_logging_for_slurm

        # Configure logging for SLURM/EXCA jobs first
        configure_logging_for_slurm()

        run_generation_with_evaluation(self)
