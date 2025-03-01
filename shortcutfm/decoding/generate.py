import lightning as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from shortcutfm.batch import collate
from shortcutfm.criteria import X0FlowMatchingCriterion, CompositeCriterion, \
    SelfConditioningFlowMatchingCriterionDecorator
from shortcutfm.model.config import TransformerNetModelConfig
from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.shortcut_samplers import TimeAndShorcutStampler, ShortcutSampler
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import EMACallback, SaveTestOutputsCallback
from shortcutfm.train.pl.train_unit import TrainModule
from shortcutfm.utils import parse_args

if __name__ == '__main__':
    cfg = parse_args()

    transformer_model_config = TransformerNetModelConfig(**cfg.model_config)
    model = TransformerNetModelFactory(transformer_model_config).build()

    # Define Criterions
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    flow_matching_criterion = X0FlowMatchingCriterion(model, diffusion_steps=cfg.model_config.diffusion_steps,
                                                      tokenizer=tokenizer)
    self_conditioning_flow_matching_criterion = SelfConditioningFlowMatchingCriterionDecorator(
        flow_matching_criterion, self_conditioning_ratio=cfg.model_config.sc_rate
    )

    shortcut_sampler = ShortcutSampler(
        diffusion_steps=cfg.model_config.diffusion_steps, min_shortcut_size=cfg.model_config.min_shortcut_size
    )
    time_and_shortcut_sampler = TimeAndShorcutStampler(
        shortcut_sampler,
        cfg.model_config.diffusion_steps,
        cfg.model_config.min_shortcut_size
    )

    criterion = CompositeCriterion(
        criteria=(self_conditioning_flow_matching_criterion,),
        criteria_weights=(1,),
        model=model,
        diffusion_steps=cfg.model_config.diffusion_steps,
        self_consistency_ratio=cfg.training_config.self_consistency_ratio,
        sampler=time_and_shortcut_sampler,
    )

    unit = TrainModule.load_from_checkpoint("checkpoints/last.ckpt", criterion=criterion)
    unit.set_prediction_shorcut_size(32)

    test_ds = Dataset.load_from_disk("datasets/tokenized/QQP-Official/test")
    val_text_ds = TextDataset(test_ds)

    test_dataloader = DataLoader(
        val_text_ds,
        batch_size=cfg.data_config.batch_size,
        collate_fn=collate,
        shuffle=not torch.distributed.is_initialized(),
    )

    ema_callback = EMACallback(decay=0.999, update_interval=1, ema_eval=True)
    checkpoint = torch.load("checkpoints/last.ckpt", map_location="cpu")

    if "EMACallback" in checkpoint["callbacks"]:
        ema_callback_state = checkpoint["callbacks"]["EMACallback"]
        ema_callback.load_state_dict(ema_callback_state)

    save_outputs_callback = SaveTestOutputsCallback(
        save_path="test_outputs.txt", diff_steps=2048, shortcut_size=64, start_example_idx=1
    )
    trainer = pl.Trainer(
        callbacks=[ema_callback, save_outputs_callback],
        limit_test_batches=2,
    )
    trainer.test(unit, dataloaders=test_dataloader)
