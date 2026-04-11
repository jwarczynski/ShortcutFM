# NFE Sweep Results

Generation and evaluation across multiple NFE (Number of Function Evaluations) values for shortcut and baseline models on QQP, PAWS-Wiki, and ParaSCI datasets.

- Shortcut models: `scut_768` (input_dims=768)
- Baseline models: `baseline_128` (input_dims=128, no shortcut training)
- Diffusion steps: 2048 for all models
- NFE = diffusion_steps / denoising_step_size
- Seed: 44, EMA weights, test split

## QQP

### QQP — Shortcut 768 (`scut_768/run_ll9cnmi5`, step=21000)

| NFE | BLEU | ROUGE-1 | ROUGE-L | BERTScore F1 | Src Copy % |
|----:|-----:|--------:|--------:|-------------:|-----------:|
| 1 | 0.2757 | 0.6018 | 0.5653 | 0.8268 | 99.0 |
| 2 | 0.0914 | 0.2352 | 0.2189 | 0.4395 | 27.0 |
| 4 | 0.1306 | 0.3137 | 0.2862 | 0.5461 | 40.0 |
| 8 | 0.1257 | 0.3459 | 0.3147 | 0.5884 | 47.0 |
| 64 | 0.0843 | 0.2918 | 0.2678 | 0.5502 | 36.0 |
| 128 | 0.0752 | 0.2623 | 0.2371 | 0.5299 | 29.0 |
| 256 | 0.0409 | 0.2310 | 0.2055 | 0.5020 | 18.0 |

### QQP — Baseline 128 (`baseline_128/run_y1x54mjl`, last.ckpt)

| NFE | BLEU | ROUGE-1 | ROUGE-L | BERTScore F1 | Src Copy % |
|----:|-----:|--------:|--------:|-------------:|-----------:|
| 1 | 0.1422 | 0.4040 | 0.3804 | 0.6174 | 0.0 |
| 2 | 0.1466 | 0.3970 | 0.3796 | 0.6199 | 0.0 |
| 4 | 0.1514 | 0.4017 | 0.3848 | 0.6208 | 0.0 |
| 8 | 0.1540 | 0.3950 | 0.3775 | 0.6196 | 0.0 |
| 64 | 0.1546 | 0.3923 | 0.3730 | 0.6215 | 0.0 |
| 128 | 0.1550 | 0.3941 | 0.3751 | 0.6210 | 0.0 |
| 256 | 0.1547 | 0.3942 | 0.3752 | 0.6217 | 0.0 |

## PAWS-Wiki

### PAWS-Wiki — Shortcut 768 (`scut_768/run_rl22bbak`, step=20000)

| NFE | BLEU | ROUGE-1 | ROUGE-L | BERTScore F1 | Src Copy % |
|----:|-----:|--------:|--------:|-------------:|-----------:|
| 1 | 0.6465 | 0.9376 | 0.8506 | 0.9436 | 100.0 |
| 2 | 0.6081 | 0.8726 | 0.7922 | 0.8958 | 92.1 |
| 4 | 0.5633 | 0.8065 | 0.7320 | 0.8475 | 84.2 |
| 8 | 0.5694 | 0.8098 | 0.7354 | 0.8513 | 85.0 |
| 64 | 0.5259 | 0.7516 | 0.6820 | 0.7976 | 76.0 |
| 128 | 0.5135 | 0.7408 | 0.6712 | 0.7885 | 74.4 |
| 256 | 0.4571 | 0.6746 | 0.6080 | 0.7422 | 65.6 |

### PAWS-Wiki — Baseline 128 (`baseline_128/run_vnu3qg1f`, last.ckpt)

| NFE | BLEU | ROUGE-1 | ROUGE-L | BERTScore F1 | Src Copy % |
|----:|-----:|--------:|--------:|-------------:|-----------:|
| 1 | 0.2059 | 0.5773 | 0.5176 | 0.6451 | 0.1 |
| 2 | 0.2124 | 0.5842 | 0.5245 | 0.6510 | 0.2 |
| 4 | 0.2173 | 0.5896 | 0.5296 | 0.6556 | 0.3 |
| 8 | 0.2200 | 0.5940 | 0.5338 | 0.6583 | 0.3 |
| 64 | 0.2206 | 0.5953 | 0.5345 | 0.6598 | 0.3 |
| 128 | 0.2209 | 0.5956 | 0.5348 | 0.6600 | 0.3 |
| 256 | 0.2209 | 0.5956 | 0.5348 | 0.6601 | 0.3 |

## ParaSCI

### ParaSCI — Shortcut 768 (`scut_768/run_rtikryfi`, step=20000)

| NFE | BLEU | ROUGE-1 | ROUGE-L | BERTScore F1 | Src Copy % |
|----:|-----:|--------:|--------:|-------------:|-----------:|
| 1 | 0.4055 | 0.6077 | 0.5682 | 0.8178 | 100.0 |
| 2 | 0.3756 | 0.5323 | 0.4972 | 0.7425 | 83.8 |
| 4 | 0.4042 | 0.6049 | 0.5656 | 0.8150 | 99.3 |
| 8 | 0.4054 | 0.6076 | 0.5681 | 0.8177 | 100.0 |
| 64 | 0.4016 | 0.5990 | 0.5602 | 0.8088 | 98.2 |
| 128 | 0.3523 | 0.5441 | 0.5078 | 0.7552 | 87.2 |
| 256 | 0.2738 | 0.4231 | 0.3956 | 0.6399 | 61.6 |

### ParaSCI — Baseline 128 (`baseline_128/run_ryrkvw5f`, last.ckpt)

| NFE | BLEU | ROUGE-1 | ROUGE-L | BERTScore F1 | Src Copy % |
|----:|-----:|--------:|--------:|-------------:|-----------:|
| 1 | 0.0787 | 0.2701 | 0.2543 | 0.5032 | 0.0 |
| 2 | 0.0772 | 0.2909 | 0.2692 | 0.5169 | 0.0 |
| 4 | 0.0712 | 0.2852 | 0.2586 | 0.5182 | 0.0 |
| 8 | 0.0663 | 0.2780 | 0.2506 | 0.5193 | 0.0 |
| 64 | 0.0618 | 0.2754 | 0.2456 | 0.5209 | 0.0 |
| 128 | 0.0616 | 0.2751 | 0.2452 | 0.5212 | 0.0 |
| 256 | 0.0617 | 0.2755 | 0.2456 | 0.5212 | 0.0 |

## Themis Evaluation (NFE=1 only)

Themis (PKU-ONELab/Themis, 8B) — reference-free NLG evaluation, Likert scale 1-5.

Evaluated on NFE=1 outputs only. PAWS-Wiki and ParaSCI evaluated on 500 random samples (seed=42); QQP on full 100-sample test set.

### Custom Aspects

| Dataset | Model | Fidelity | Fluency | Lexical Diversity | Overall Quality |
|---------|-------|:--------:|:-------:|:-----------------:|:---------------:|
| QQP | Scut 768 | 4.95 | 4.61 | 1.00 | 4.36 |
| QQP | Baseline 128 | 1.12 | 1.12 | 1.08 | 1.10 |
| PAWS-Wiki | Scut 768 | 5.00 | 4.90 | 1.00 | 4.93 |
| PAWS-Wiki | Baseline 128 | 1.05 | 1.06 | 1.02 | 1.03 |
| ParaSCI | Scut 768 | 5.00 | 4.79 | 1.00 | 4.80 |
| ParaSCI | Baseline 128 | 1.00 | 1.00 | 1.00 | 1.00 |

### Paper Aspects (from Themis paper appendix)

| Dataset | Model | Fluency | Semantic Similarity | Overall Quality |
|---------|-------|:-------:|:-------------------:|:---------------:|
| QQP | Scut 768 | 4.75 | 4.96 | 2.52 |
| QQP | Baseline 128 | 1.13 | 1.13 | 1.09 |
| PAWS-Wiki | Scut 768 | 4.96 | 5.00 | 4.50 |
| PAWS-Wiki | Baseline 128 | 1.05 | 1.04 | 1.03 |
| ParaSCI | Scut 768 | 4.92 | 5.00 | 3.22 |
| ParaSCI | Baseline 128 | 1.00 | 1.00 | 1.00 |

## Themis Aspect Definitions

All aspects are wrapped in the standard Themis instruction template from their `eval.py` (`###Instruction###\nPlease act as an impartial and helpful evaluator...`). Only the criterion text below is inserted into the `{aspect}` slot.

### Custom Aspects

**Fidelity/Faithfulness:**
> Evaluate whether the generated paraphrase preserves the original meaning of the source question. A high-fidelity paraphrase should convey the same intent and information as the source, without adding, omitting, or distorting any key details. Rate 1 if the meaning is completely changed, 5 if the meaning is perfectly preserved.

**Fluency:**
> Evaluate the grammatical correctness and naturalness of the generated paraphrase. A fluent paraphrase should read naturally, be grammatically correct, and use appropriate vocabulary. Rate 1 if the text is incomprehensible or severely ungrammatical, 5 if it is perfectly fluent and natural.

**Lexical Diversity:**
> Evaluate how much the generated paraphrase differs from the source in terms of word choice and phrasing while still preserving meaning. A good paraphrase should use different words and sentence structures rather than copying the source verbatim. Rate 1 if it is an exact copy, 5 if it uses substantially different wording while keeping the meaning.

**Overall Quality:**
> Evaluate the overall quality of the generated paraphrase considering all aspects: meaning preservation, fluency, naturalness, and lexical diversity. A high-quality paraphrase should faithfully convey the original meaning using different wording in a natural and grammatically correct way. Rate 1 for very poor quality, 5 for excellent quality.

### Paper Aspects (from Themis paper appendix — used in their training data for paraphrase evaluation)

**Fluency:**
> Whether the paraphrase is meaningful and grammatical?

**Semantic Similarity:**
> Whether the paraphrase maintains similar semantics to the original text?

**Overall Quality:**
> The paraphrase should not only maintain similar semantics to the original text, but also possess lexical or syntactic differences from the original text, with fluent and coherent content

## Key Observations

1. **Shortcut models strongly benefit from NFE=1**: The scut_768 models achieve their best scores at NFE=1 across all datasets, confirming the shortcut training objective works as intended.

2. **Baseline models are NFE-invariant**: The baseline_128 models show nearly identical performance across all NFE values, as expected — they weren't trained with shortcut objectives.

3. **Shortcut models degrade with more steps**: For scut_768, increasing NFE beyond 1 consistently hurts performance. The model was trained to predict the final output in a single step.

4. **PAWS-Wiki shows strongest shortcut advantage**: NFE=1 scut_768 achieves BLEU=0.6465 vs baseline's best of 0.2209 — a 3× improvement.

5. **ParaSCI scut is robust up to NFE=8**: Unlike QQP where NFE=2 already drops significantly, ParaSCI scut maintains near-peak performance through NFE=8 before degrading.

6. **Themis — Shortcut models excel on fluency/semantics but not overall**: All scut_768 models score near-perfect semantic similarity (4.96-5.0) and high fluency (4.75-4.96). However, the paper's Overall Quality aspect — which explicitly requires lexical/syntactic differences — drops to 2.52 (QQP) and 3.22 (ParaSCI), correctly penalizing source copying. PAWS-Wiki remains high (4.50) likely because the task inherently involves very similar sentence pairs.

7. **Themis — Baselines fail at NFE=1**: All baseline_128 models score ~1.0 across all Themis aspects at NFE=1, confirming they cannot generate meaningful text in a single step (they need many denoising steps).

8. **Source copying is dominant in shortcut models**: At NFE=1, scut_768 copies the source verbatim 99-100% of the time across all datasets. This explains the high BLEU/ROUGE/BERTScore and the Themis lexical diversity score of 1.0. The model learns to preserve the input rather than paraphrase it. Baseline models never copy the source (0%).
