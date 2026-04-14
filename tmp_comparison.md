## QQP: Shortcut (fixed) vs Baseline

### Standard Metrics

| Model | NFE | BLEU | ROUGE-1 | ROUGE-L | BERTScore F1 | Src Copy % |
|-------|----:|-----:|--------:|--------:|-------------:|-----------:|
| Shortcut (fixed) | 1 | 0.194 | 0.511 | 0.485 | 0.711 | 5.0 |
| Shortcut (fixed) | 4 | 0.201 | 0.533 | 0.500 | 0.728 | — |
| Baseline | 1 | 0.142 | 0.404 | 0.380 | 0.617 | 0.0 |
| Baseline | 4 | 0.151 | 0.402 | 0.385 | 0.621 | 0.0 |
| Baseline | 256 | 0.155 | 0.394 | 0.375 | 0.622 | 0.0 |

### Themis — Paper Aspects (NFE=1)

| Model | Fluency | Semantic Similarity | Overall Quality |
|-------|:-------:|:-------------------:|:---------------:|
| Shortcut (fixed) | 1.63 | 1.67 | 1.42 |
| Baseline | 1.13 | 1.13 | 1.09 |

The fixed shortcut model at NFE=1 outperforms the baseline at any NFE on all standard metrics. Themis scores are low for both but the shortcut model is notably higher. The shortcut model at NFE=4 further improves, reaching 0.201 BLEU and 0.728 BERTScore — surpassing the baseline's best (NFE=256) by a clear margin in a fraction of the steps.
