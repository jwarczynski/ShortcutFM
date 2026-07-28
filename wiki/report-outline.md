---
tags: [report, outline]
date: 2026-07-28
related: [[bug-consistency-onset-collapse]] [[exp-r0-reeval-results]] [[exp-round2-bertinit]] [[bug-bert-init-copy-collapse]] [[bug-eval-nfe1-random]]
---

# Final report outline (HTML page for the user — build when search closes)

Key findings to feature, in narrative order:

1. **Eval was broken** ([[bug-eval-nfe1-random]]): val BLEU measured at NFE=1 + random
   unmasking; continuous paper's 0.276 was 99% src-copy. Honest continuous bar: 0.155.
2. **R0 corrected picture** ([[exp-r0-reeval-results]]): masked diffusion 0.266-0.268
   at ~50k steps, no copy pathology, BLEU rises with NFE. Anti-LLaDA finding:
   confidence unmasking underperforms random at NFE>=16 on short targets.
3. **Copy collapse trap** ([[bug-bert-init-copy-collapse]]): bert-init at lr 1e-4 =
   100% copying (BLEU 0.271 fake). copy% metric caught in minutes what took the
   continuous project a paper cycle. bert-init at lr 3e-5 is fine and strong.
4. **Recipe found** ([[exp-round2-bertinit]]): bert-init + lr 3e-5 + t-clip[0.15,0.95]
   ~ 0.25-0.27 bleu16 at 10k steps (5x fewer than from-scratch ceiling).
5. **STAR FINDING — consistency-onset collapse** ([[bug-consistency-onset-collapse]]):
   combo-30k at 0.27 bleu16 AND 0.27 bleu1 by 5k steps, then collapse exactly at
   consistency_start_step=5000; bleu1 never recovered. Rule: shortcut/consistency
   training from step 0 or ramped — never abrupt onset. Also: pre-collapse combo =
   best 1-step model of the search (0.26-0.27 @ NFE=1 at 5k steps).
6. **Shortcut question** (pending r3 candidate pair): at 50k from-scratch, consistency
   bought NFE=1 parity but not a higher ceiling; short targets barely need multi-step.
   Real test = longer-target tasks (summarization/MT — roadmap).
7. Infra story: 3 clusters (hgx unstable, athena reliable, pcss huge/H100 + gotchas),
   wiki system, all bugs/gotchas documented as they happened.
