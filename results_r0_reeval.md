# R0 re-evaluation — masked diffusion, corrected decoding

Validation split, BLEU (NLTK method4) + copy%. Ablation conditioned on d=0.

## main-50k (epoch=690-step=49000-val_bleu=0.0000.ckpt)

| NFE | strategy | d | BLEU | copy% |
|---|---|---|---|---|
| 1 | confidence | 2048 | 0.2212 | 3.3 |
| 1 | random | 2048 | 0.2212 | 3.3 |
| 4 | confidence | 512 | 0.2343 | 4.6 |
| 4 | random | 512 | 0.2548 | 4.3 |
| 16 | confidence | 128 | 0.1808 | 6.5 |
| 16 | random | 128 | 0.2657 | 4.7 |
| 64 | confidence | 32 | 0.2034 | 8.3 |
| 64 | random | 32 | 0.2645 | 4.5 |

## main-10k (epoch=140-step=10000-val_bleu=0.0000.ckpt)

| NFE | strategy | d | BLEU | copy% |
|---|---|---|---|---|
| 1 | confidence | 2048 | 0.0888 | 1.0 |
| 1 | random | 2048 | 0.0888 | 1.0 |
| 4 | confidence | 512 | 0.0966 | 0.4 |
| 4 | random | 512 | 0.1092 | 1.2 |
| 16 | confidence | 128 | 0.0219 | 0.1 |
| 16 | random | 128 | 0.1243 | 1.5 |
| 64 | confidence | 32 | 0.0462 | 0.6 |
| 64 | random | 32 | 0.1333 | 1.5 |

## ablation-50k (epoch=704-step=50000-val_bleu=0.0000.ckpt)

| NFE | strategy | d | BLEU | copy% |
|---|---|---|---|---|
| 1 | confidence | 0 | 0.2128 | 2.0 |
| 1 | random | 0 | 0.2128 | 2.0 |
| 4 | confidence | 0 | 0.2681 | 3.9 |
| 4 | random | 0 | 0.2594 | 3.1 |
| 16 | confidence | 0 | 0.2186 | 6.7 |
| 16 | random | 0 | 0.2660 | 3.5 |
| 64 | confidence | 0 | 0.2358 | 7.7 |
| 64 | random | 0 | 0.2681 | 3.8 |

## ablation-10k (epoch=140-step=10000-val_bleu=0.0000.ckpt)

| NFE | strategy | d | BLEU | copy% |
|---|---|---|---|---|
| 1 | confidence | 0 | 0.1242 | 1.7 |
| 1 | random | 0 | 0.1242 | 1.7 |
| 4 | confidence | 0 | 0.1698 | 3.8 |
| 4 | random | 0 | 0.1587 | 2.6 |
| 16 | confidence | 0 | 0.1322 | 3.3 |
| 16 | random | 0 | 0.1683 | 3.2 |
| 64 | confidence | 0 | 0.1368 | 4.0 |
| 64 | random | 0 | 0.1688 | 2.9 |

