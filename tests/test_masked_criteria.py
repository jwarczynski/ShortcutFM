import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from shortcutfm.batch import EncoderBatch, MaskedDiffusionBatch, MaskedShortcutBatch
from shortcutfm.masked_criteria import (
    MaskedCompositeCriterion,
    MaskedConsistencyCriterion,
    MaskedDiffusionCriterion,
    select_positions_to_unmask,
)

VOCAB_SIZE = 20
MASK_ID = 3
DIFFUSION_STEPS = 8
HIDDEN = 6


def make_training_cfg(**model_overrides):
    model = SimpleNamespace(
        mask_token_id=MASK_ID,
        unmask_strategy="random",
        ce_importance_weighting="one_over_t",
        consistency_loss_space="l2_embedding",
        use_default_t_for_shortcut=False,
        diffusion_steps=DIFFUSION_STEPS,
    )
    for k, v in model_overrides.items():
        setattr(model, k, v)
    return SimpleNamespace(
        model=model,
        consistency_start_step=0,
        self_consistency_ratio=0.5,
    )


class IdentityLogitsModel(nn.Module):
    """Model stub: embeddings from a real embedding table, logits favor the input token
    shifted by +1 (deterministic, so composition behavior is predictable)."""

    def __init__(self):
        super().__init__()
        self.module = SimpleNamespace(
            word_embedding=nn.Embedding(VOCAB_SIZE, HIDDEN),
        )
        self._last_input_embeds = None

    def forward(self, x, t, shortcuts):
        self._last_input_embeds = x
        return x

    def get_embeddings(self, input_ids):
        self._last_ids = input_ids
        return self.module.word_embedding(input_ids)

    def compute_logits(self, hidden):
        # Fixed logits independent of hidden values except batch/seq shape:
        # every position strongly predicts token (last_ids + 1) % VOCAB_SIZE
        ids = self._last_ids
        logits = torch.full((*ids.shape, VOCAB_SIZE), -10.0)
        target = (ids + 1) % VOCAB_SIZE
        logits.scatter_(-1, target.unsqueeze(-1), 10.0)
        return logits

    def eval(self):
        return self


def make_batch(bsz=2, seq_len=8):
    seqs = torch.arange(4, 4 + seq_len).repeat(bsz, 1) % VOCAB_SIZE
    padding_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    padding_mask[:, -1] = 0  # last position is padding
    input_ids_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    input_ids_mask[:, :2] = 0  # first two positions are source
    return EncoderBatch(
        seqs=seqs, padding_mask=padding_mask, input_ids_mask=input_ids_mask, global_step=0
    )


class TestSelectPositionsToUnmask(unittest.TestCase):
    def test_random_selects_exact_count_within_masked(self):
        masked = torch.tensor([[True, True, True, False], [False, True, False, False]])
        num = torch.tensor([2, 1])
        logits = torch.zeros(2, 4, VOCAB_SIZE)
        selected = select_positions_to_unmask(masked, num, logits, "random")
        self.assertTrue((selected & ~masked).sum() == 0)
        self.assertEqual(selected.sum(-1).tolist(), [2, 1])

    def test_confidence_prefers_high_prob_positions(self):
        masked = torch.tensor([[True, True, True, True]])
        logits = torch.zeros(1, 4, VOCAB_SIZE)
        logits[0, 2, 5] = 10.0  # position 2 is most confident
        selected = select_positions_to_unmask(masked, torch.tensor([1]), logits, "confidence")
        self.assertTrue(selected[0, 2].item())
        self.assertEqual(selected.sum().item(), 1)


class TestMaskedDiffusionCriterion(unittest.TestCase):
    def setUp(self):
        self.model = IdentityLogitsModel()
        self.criterion = MaskedDiffusionCriterion(
            self.model,
            diffusion_steps=DIFFUSION_STEPS,
            tokenizer=MagicMock(),
            training_cfg=make_training_cfg(),
        )
        self.batch = make_batch()

    def test_corrupt_full_masking_at_t_max(self):
        denoise_mask = self.batch.input_ids_mask * self.batch.padding_mask
        t = torch.full((2,), DIFFUSION_STEPS)
        x_t, mask_indicator = self.criterion.corrupt(self.batch.seqs, t, denoise_mask)
        self.assertTrue(torch.equal(mask_indicator.long(), denoise_mask))
        self.assertTrue((x_t[mask_indicator] == MASK_ID).all())

    def test_corrupt_never_masks_source_or_padding(self):
        torch.manual_seed(0)
        denoise_mask = self.batch.input_ids_mask * self.batch.padding_mask
        for t_val in [1, DIFFUSION_STEPS // 2, DIFFUSION_STEPS]:
            t = torch.full((2,), t_val)
            x_t, mask_indicator = self.criterion.corrupt(self.batch.seqs, t, denoise_mask)
            self.assertTrue((mask_indicator & (denoise_mask == 0)).sum() == 0)
            # source and padding tokens unchanged
            self.assertTrue(torch.equal(x_t[denoise_mask == 0], self.batch.seqs[denoise_mask == 0]))

    def test_corrupt_forces_at_least_one_mask(self):
        torch.manual_seed(0)
        denoise_mask = self.batch.input_ids_mask * self.batch.padding_mask
        for _ in range(20):
            _, mask_indicator = self.criterion.corrupt(self.batch.seqs, torch.tensor([1, 1]), denoise_mask)
            self.assertTrue((mask_indicator.sum(-1) >= 1).all())

    def test_corrupt_masking_rate_statistics(self):
        torch.manual_seed(0)
        seq_len = 2000
        seqs = torch.randint(4, VOCAB_SIZE, (1, seq_len))
        denoise_mask = torch.ones(1, seq_len, dtype=torch.long)
        t = torch.tensor([DIFFUSION_STEPS // 2])
        _, mask_indicator = self.criterion.corrupt(seqs, t, denoise_mask)
        rate = mask_indicator.float().mean().item()
        self.assertAlmostEqual(rate, 0.5, delta=0.05)

    def test_ce_weighting_uniform_logits(self):
        model = MagicMock()
        model.get_embeddings.return_value = torch.zeros(2, 8, HIDDEN)
        model.return_value = torch.zeros(2, 8, HIDDEN)
        model.compute_logits.return_value = torch.zeros(2, 8, VOCAB_SIZE)
        criterion = MaskedDiffusionCriterion(
            model,
            diffusion_steps=DIFFUSION_STEPS,
            tokenizer=MagicMock(),
            training_cfg=make_training_cfg(),
        )
        batch = make_batch()
        denoise_mask = batch.input_ids_mask * batch.padding_mask
        t = torch.tensor([2, 4])
        x_t, mask_indicator = criterion.corrupt(batch.seqs, t, denoise_mask)
        md_batch = MaskedDiffusionBatch(
            seqs=batch.seqs,
            padding_mask=batch.padding_mask,
            input_ids_mask=batch.input_ids_mask,
            x_t=x_t,
            mask_indicator=mask_indicator,
            t=t,
            global_step=0,
        )
        losses = criterion(md_batch, world_size=1)
        # uniform logits => per-token CE == log(V); per-sample loss ==
        # (T/t) * log(V) * (num_masked / num_target)
        num_target = denoise_mask.sum(-1).float()
        num_masked = mask_indicator.sum(-1).float()
        expected = (DIFFUSION_STEPS / t.float()) * math.log(VOCAB_SIZE) * num_masked / num_target
        torch.testing.assert_close(losses["masked_ce_loss"], expected, atol=1e-5, rtol=1e-5)


class TestMaskedConsistencyCriterion(unittest.TestCase):
    def setUp(self):
        self.model = IdentityLogitsModel()
        self.criterion = MaskedConsistencyCriterion(
            self.model,
            diffusion_steps=DIFFUSION_STEPS,
            training_cfg=make_training_cfg(),
        )
        self.batch = make_batch()

    def _make_shortcut_batch(self, t_val=4, d_val=2):
        denoise_mask = self.batch.input_ids_mask * self.batch.padding_mask
        t = torch.full((2,), t_val)
        criterion_helper = MaskedDiffusionCriterion(
            self.model, DIFFUSION_STEPS, MagicMock(), make_training_cfg()
        )
        x_t, mask_indicator = criterion_helper.corrupt(self.batch.seqs, t, denoise_mask)
        return MaskedShortcutBatch(
            seqs=self.batch.seqs,
            padding_mask=self.batch.padding_mask,
            input_ids_mask=self.batch.input_ids_mask,
            x_t=x_t,
            mask_indicator=mask_indicator,
            t=t,
            shortcut_size=torch.full((2,), d_val),
            global_step=0,
        )

    def test_build_x_t_minus_d_counts(self):
        torch.manual_seed(0)
        batch = self._make_shortcut_batch(t_val=4, d_val=2)
        logits_1 = self.model.compute_logits_from_ids(batch.x_t) if hasattr(
            self.model, "compute_logits_from_ids"
        ) else None
        self.model._last_ids = batch.x_t
        logits_1 = self.model.compute_logits(None)
        x_tmd, still_masked, unmasked_now = self.criterion._build_x_t_minus_d(
            batch.x_t, batch.mask_indicator, logits_1, batch.t, batch.shortcut_size
        )
        num_masked = batch.mask_indicator.sum(-1).float()
        expected_keep = torch.round(num_masked * (batch.t - batch.shortcut_size).float() / batch.t.float())
        torch.testing.assert_close(still_masked.sum(-1).float(), expected_keep)
        # revealed positions carry step-1 argmax
        self.assertTrue(torch.equal(x_tmd[unmasked_now], logits_1.argmax(-1)[unmasked_now]))
        # still-masked positions keep the mask token
        self.assertTrue((x_tmd[still_masked] == MASK_ID).all())

    def test_loss_zero_when_prediction_matches_target(self):
        torch.manual_seed(0)
        batch = self._make_shortcut_batch()
        # IdentityLogitsModel's logits depend only on input ids, and prediction and
        # step-1 share the same input x_t. With d such that step 2 sees the same
        # still-masked positions, target==prediction on still-masked positions when the
        # model is deterministic per-position. Instead of relying on that, monkeypatch
        # the target to equal the prediction and assert exact zero.
        pred_logits = self.criterion._predict(batch)
        self.criterion._compute_shortcut_target = lambda b: pred_logits.detach()
        for space in ["l2_embedding", "kl"]:
            self.criterion.loss_space = space
            losses = self.criterion(batch, world_size=1)
            torch.testing.assert_close(
                losses["consistency_loss"],
                torch.zeros_like(losses["consistency_loss"]),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_loss_only_on_masked_positions(self):
        torch.manual_seed(0)
        batch = self._make_shortcut_batch()
        per_token = self.criterion.compute_losses(batch, world_size=1)["consistency_loss"]
        self.assertTrue((per_token[~batch.mask_indicator] == 0).all())

    def test_gradient_flows_through_prediction_only(self):
        torch.manual_seed(0)
        batch = self._make_shortcut_batch()

        class GradModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = SimpleNamespace(word_embedding=nn.Embedding(VOCAB_SIZE, HIDDEN))
                self.head = nn.Linear(HIDDEN, VOCAB_SIZE)

            def forward(self, x, t, shortcuts):
                return x

            def get_embeddings(self, ids):
                return self.module.word_embedding(ids)

            def compute_logits(self, hidden):
                return self.head(hidden)

        model = GradModel()
        criterion = MaskedConsistencyCriterion(
            model, DIFFUSION_STEPS, training_cfg=make_training_cfg()
        )
        losses = criterion(batch, world_size=1)
        loss = losses["consistency_loss"].mean()
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())


class TestMaskedCompositeCriterion(unittest.TestCase):
    def _make_composite(self, self_consistency_ratio=0.5, consistency_start_step=0):
        cfg = make_training_cfg()
        cfg.self_consistency_ratio = self_consistency_ratio
        cfg.consistency_start_step = consistency_start_step

        class GradModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = SimpleNamespace(word_embedding=nn.Embedding(VOCAB_SIZE, HIDDEN))
                self.head = nn.Linear(HIDDEN, VOCAB_SIZE)

            def forward(self, x, t, shortcuts):
                return x

            def get_embeddings(self, ids):
                return self.module.word_embedding(ids)

            def compute_logits(self, hidden):
                return self.head(hidden)

            def eval(self):
                return self

        model = GradModel()
        masked = MaskedDiffusionCriterion(model, DIFFUSION_STEPS, MagicMock(), cfg)
        consistency = MaskedConsistencyCriterion(model, DIFFUSION_STEPS, training_cfg=cfg)

        sampler = MagicMock()
        sampler.side_effect = lambda batch_size, device: (
            torch.full((batch_size,), DIFFUSION_STEPS, device=device),
            torch.ones(batch_size, device=device),
        )
        ts_sampler = MagicMock()
        ts_sampler.side_effect = lambda batch_size, device: (
            torch.full((batch_size,), 4, device=device),
            torch.full((batch_size,), 2, device=device),
            torch.ones(batch_size, device=device),
        )

        composite = MaskedCompositeCriterion(
            masked_criterion=masked,
            consistency_criterion=consistency,
            masked_ce_weight=1.0,
            consistency_weight=1.0,
            model=model,
            diffusion_steps=DIFFUSION_STEPS,
            self_consistency_ratio=self_consistency_ratio,
            sampler=sampler,
            time_shortcut_sampler=ts_sampler,
            training_cfg=cfg,
        )
        return composite

    def test_batch_split_and_loss_keys(self):
        torch.manual_seed(0)
        composite = self._make_composite(self_consistency_ratio=0.5)
        batch = make_batch(bsz=4)
        result = composite(batch, world_size=1)
        self.assertIn("masked_ce_loss", result)
        self.assertIn("consistency_loss", result)
        self.assertIn("loss", result)
        self.assertIn("timestep", result)
        self.assertIn("shortcut", result)
        self.assertEqual(result["masked_ce_loss"].shape[0], 2)
        self.assertEqual(result["consistency_loss"].shape[0], 2)
        self.assertEqual(result["timestep"].shape[0], 4)
        self.assertEqual(result["loss"].dim(), 0)
        self.assertTrue(result["loss"].requires_grad)

    def test_consistency_disabled_before_start_step(self):
        torch.manual_seed(0)
        composite = self._make_composite(consistency_start_step=100)
        batch = make_batch(bsz=4)
        batch.global_step = 0
        result = composite(batch, world_size=1)
        self.assertNotIn("consistency_loss", result)
        self.assertEqual(result["masked_ce_loss"].shape[0], 4)


class TestMaskedDenoise(unittest.TestCase):
    def setUp(self):
        self.model = IdentityLogitsModel()
        self.criterion = MaskedDiffusionCriterion(
            self.model,
            diffusion_steps=DIFFUSION_STEPS,
            tokenizer=MagicMock(),
            training_cfg=make_training_cfg(),
        )
        self.batch = make_batch()

    def test_no_mask_left_and_source_preserved(self):
        torch.manual_seed(0)
        out = self.criterion.denoise(self.batch, shortcut_size=0, step_size=2, probe_every_step=False)
        denoise_mask = self.batch.input_ids_mask * self.batch.padding_mask
        self.assertTrue((out[denoise_mask == 1] != MASK_ID).all())
        self.assertTrue(torch.equal(out[denoise_mask == 0], self.batch.seqs[denoise_mask == 0]))

    def test_output_shapes_all_modes(self):
        torch.manual_seed(0)
        bsz, seq_len = self.batch.seqs.shape
        step = 2
        num_steps = len(range(DIFFUSION_STEPS, 0, -step))

        out = self.criterion.denoise(self.batch, shortcut_size=0, step_size=step, probe_every_step=True)
        self.assertEqual(tuple(out.shape), (bsz, num_steps, seq_len))
        self.assertEqual(out.dtype, torch.long)

        out = self.criterion.denoise(self.batch, shortcut_size=0, step_size=step, probe_every_step=False)
        self.assertEqual(tuple(out.shape), (bsz, seq_len))

        out = self.criterion.denoise(
            self.batch, shortcut_size=0, step_size=step, probe_every_step=False, return_logits=True
        )
        self.assertEqual(tuple(out.shape), (bsz, seq_len, VOCAB_SIZE))

        out = self.criterion.denoise(
            self.batch, shortcut_size=0, step_size=step, probe_every_step=True, return_logits=True
        )
        self.assertEqual(tuple(out.shape), (bsz, num_steps, seq_len, VOCAB_SIZE))

    def test_shortcut_conditioning_passed(self):
        torch.manual_seed(0)
        calls = []
        original_forward = self.model.forward

        def spy_forward(x, t, shortcuts):
            calls.append(shortcuts.clone())
            return original_forward(x, t, shortcuts)

        self.model.forward = spy_forward
        self.criterion.denoise(self.batch, shortcut_size=4, step_size=4, probe_every_step=False)
        for shortcuts in calls:
            self.assertTrue((shortcuts == 4).all())

    def test_requires_step_or_shortcut(self):
        with self.assertRaises(ValueError):
            self.criterion.denoise(self.batch)
        with self.assertRaises(ValueError):
            self.criterion.denoise(self.batch, shortcut_size=0)


if __name__ == "__main__":
    unittest.main()
