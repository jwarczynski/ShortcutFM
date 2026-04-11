"""
Evaluation configuration for standalone evaluation script.
This module contains only the evaluation configuration to avoid heavy imports.
"""

from pathlib import Path
from typing import Literal

import exca
from pydantic import BaseModel, ConfigDict, Field


class EvaluationConfig(BaseModel):
    """Configuration for evaluation jobs on Slurm cluster"""
    
    # Required parameters matching argparse arguments
    output_dir: str = Field(..., description="Directory containing generation outputs")
    tokenizer: str = Field(default="bert-base-uncased", description="Tokenizer to use for decoding")
    device: str = Field(default="cuda", description="Device to use for BERTScore computation")
    use_fallback_processing: bool = Field(default=False, description="Use fallback processing for empty predictions")
    suffix: str = Field(default="", description="Suffix to append to output files")
    batch_qqp: bool = Field(default=False, description="Batch evaluate all seed directories in QQP folder")
    skip_existing: bool = Field(default=True, description="Skip directories that already have metrics files")
    force_overwrite: bool = Field(default=False, description="Force overwrite existing metrics files")
    
    # Exca-specific settings
    use_exca: bool = Field(default=False, description="Whether to use Exca for submitting evaluation tasks")
    
    # Infrastructure for exca job submission
    infra: exca.TaskInfra = exca.TaskInfra()
    
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    @infra.apply
    def run_evaluation(self):
        """Run evaluation with exca infrastructure"""
        from shortcutfm.evaluation import evaluate_generations
        from shortcutfm.utils.logging_utils import configure_logging_for_slurm
        import logging
        from pathlib import Path
        
        # Configure logging for SLURM/EXCA jobs first
        configure_logging_for_slurm()
        logger = logging.getLogger(__name__)
        
        logger.info(f"Running evaluation for: {self.output_dir}")
        
        output_path = Path(self.output_dir)
        if not output_path.exists():
            logger.error(f"Output directory does not exist: {output_path}")
            raise ValueError(f"Output directory does not exist: {output_path}")
        
        try:
            metrics = evaluate_generations(
                output_path,
                self.tokenizer,
                self.device,
                self.use_fallback_processing,
                suffix=self.suffix
            )
            
            logger.info("Evaluation completed successfully!")
            logger.info(f"Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
