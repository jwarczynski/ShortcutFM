# Decode study — masked diffusion, inference-time decode modes

Validation split, BLEU (NLTK method4) + copy%. Consistency-trained checkpoints conditioned on d=step_size; ablation on d=0. schedule-* rows reproduce R0.

## main-50k (epoch=690-step=49000-val_bleu=0.0000.ckpt)

| NFE | decode mode | d | BLEU | copy% |
|---|---|---|---|---|
| 4 | schedule-conf | 512 | 0.2343 | 4.6 |
| 4 | schedule-rand | 512 | 0.2581 | 4.2 |
| 4 | threshold-0.9 | 512 | 0.2358 | 4.8 |
| 4 | threshold-0.95 | 512 | 0.2054 | 4.9 |
| 4 | block-16-conf | 512 | 0.2386 | 4.8 |
| 4 | block-32-conf | 512 | 0.2330 | 4.6 |
| 16 | schedule-conf | 128 | 0.1808 | 6.5 |
| 16 | schedule-rand | 128 | 0.2691 | 4.6 |
| 16 | threshold-0.9 | 128 | 0.2899 | 6.1 |
| 16 | threshold-0.95 | 128 | 0.2276 | 6.4 |
| 16 | block-16-conf | 128 | 0.2339 | 6.2 |
| 16 | block-32-conf | 128 | 0.1818 | 6.5 |

## ablation-50k (epoch=704-step=50000-val_bleu=0.0000.ckpt)

| NFE | decode mode | d | BLEU | copy% |
|---|---|---|---|---|
| 4 | schedule-conf | 0 | 0.2681 | 3.9 |
| 4 | schedule-rand | 0 | 0.2648 | 2.7 |
| 4 | threshold-0.9 | 0 | 0.2678 | 3.3 |
| 4 | threshold-0.95 | 0 | 0.2580 | 3.5 |
| 4 | block-16-conf | 0 | 0.2660 | 4.1 |
| 4 | block-32-conf | 0 | 0.2679 | 3.9 |
| 16 | schedule-conf | 0 | 0.2186 | 6.7 |
| 16 | schedule-rand | 0 | 0.2722 | 2.9 |
| 16 | threshold-0.9 | 0 | 0.3176 | 3.6 |
| 16 | threshold-0.95 | 0 | 0.3181 | 4.1 |
| 16 | block-16-conf | 0 | 0.2564 | 6.3 |
| 16 | block-32-conf | 0 | 0.2203 | 6.7 |

## tclip-cons-10k (epoch=140-step=10000-val_bleu=0.0000.ckpt)

| NFE | decode mode | d | BLEU | copy% |
|---|---|---|---|---|
| 4 | schedule-conf | 512 | 0.2551 | 6.2 |
| 4 | schedule-rand | 512 | 0.2275 | 3.8 |
| 4 | threshold-0.9 | 512 | 0.2277 | 5.7 |
| 4 | threshold-0.95 | 512 | 0.2005 | 4.7 |
| 4 | block-16-conf | 512 | 0.2528 | 6.1 |
| 4 | block-32-conf | 512 | 0.2553 | 6.2 |
| 16 | schedule-conf | 128 | 0.2569 | 12.5 |
| 16 | schedule-rand | 128 | 0.2551 | 5.9 |
| 16 | threshold-0.9 | 128 | 0.2514 | 9.2 |
| 16 | threshold-0.95 | 128 | 0.2420 | 10.3 |
| 16 | block-16-conf | 128 | 0.2728 | 10.7 |
| 16 | block-32-conf | 128 | 0.2588 | 12.5 |

## tclip-nocons-10k (epoch=140-step=10000-val_bleu=0.0000.ckpt)

| NFE | decode mode | d | BLEU | copy% |
|---|---|---|---|---|
| 4 | schedule-conf | 0 | 0.2684 | 4.2 |
| 4 | schedule-rand | 0 | 0.2157 | 2.3 |
| 4 | threshold-0.9 | 0 | 0.2292 | 4.1 |
| 4 | threshold-0.95 | 0 | 0.2171 | 4.0 |
| 4 | block-16-conf | 0 | 0.2671 | 4.2 |
| 4 | block-32-conf | 0 | 0.2671 | 4.2 |
| 16 | schedule-conf | 0 | 0.2248 | 6.3 |
| 16 | schedule-rand | 0 | 0.2586 | 3.0 |
| 16 | threshold-0.9 | 0 | 0.2950 | 5.4 |
| 16 | threshold-0.95 | 0 | 0.2883 | 4.7 |
| 16 | block-16-conf | 0 | 0.2573 | 5.1 |
| 16 | block-32-conf | 0 | 0.2253 | 6.3 |

