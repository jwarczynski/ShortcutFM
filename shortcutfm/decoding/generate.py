from shortcutfm.batch import collate
from shortcutfm.text_datasets import TextDataset
from datasets import Dataset
from torch.utils.data import DataLoader
from shortcutfm.decoding.prediction_strategies import X0PredictionStrategy, SelfConditioningPredictionDecorator
from transformers import AutoTokenizer
import numpy as np
import torch

from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.model.config import TransformerNetModelConfig
from omegaconf import DictConfig, OmegaConf as om
from shortcutfm.shortcut_samplers import TimeAndShorcutStampler, ShortcutSampler
from shortcutfm.criteria import X0FlowMatchingCriterion, CompositeCriterion, NllCriterion, X0ConsistencyCrterion, \
    SelfConditioningFlowMatchingCriterionDecorator, SelfConditioningConsistencyCriterionDecorator
from shortcutfm.train.pl.train_unit import TrainModule
from shortcutfm.utils import parse_args

if __name__ == '__main__':
    cfg = parse_args()

    transformer_model_config = TransformerNetModelConfig(**cfg.model_config)
    model = TransformerNetModelFactory(transformer_model_config).build().to("cuda")


    checkpoint = torch.load("checkpoints/last.ckpt")
    for key in checkpoint.keys():
        print(f"Key: {key}")
        if isinstance(checkpoint[key], dict):
            for subkey in checkpoint[key].keys():
                print(f"  Subkey: {subkey}")


    ema_weights = checkpoint["callbacks"]["EMACallback"]["shadow_params"]
    ema_weights_clean = {key.replace("criterion.model.", ""): value for key, value in ema_weights.items()}

    for name, param in model.named_parameters():
        if name in ema_weights_clean:
            print(f"Loading EMA weight for {name}")
            param.data.copy_(ema_weights_clean[name])
        else:
            print(f"Skipping {name}, not found in EMA checkpoint")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    x0_strategy = X0PredictionStrategy(model, 2048, tokenizer)
    sc_decorator = SelfConditioningPredictionDecorator(x0_strategy, model, 2048, tokenizer)

    test_ds = Dataset.load_from_disk("datasets/tokenized/QQP-Official/test")
    val_text_ds = TextDataset(test_ds)

    test_dataloader = DataLoader(
        val_text_ds,
        batch_size=cfg.data_config.batch_size,
        collate_fn=collate,
        shuffle=not torch.distributed.is_initialized(),
    )

    for b in test_dataloader:
        b.seqs = b.seqs.to(torch.device('cuda'))
        b.input_ids_mask = b.input_ids_mask.to(torch.device('cuda'))
        b.padding_mask = b.padding_mask.to(torch.device('cuda'))
        result = sc_decorator.denoise(b, shortcut_size=64)
        # save results to a file
        with open("results.txt", "w") as f:
            for i, example in enumerate(result, start=1):  # Iterate over batch size
                f.write(f"Example {i}:\n")  # Label each example
                for t, prediction in zip(range(2048, 0, -64), example):
                    f.write(f"  Timestep {t}: {prediction}\n")  # Indent timesteps
                f.write("\n")  # Add spacing between examples

        print(result.shape)
        break