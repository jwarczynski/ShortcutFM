# 5-minute talk script — One-step NAR Text Generation with Shortcut Flow Matching

Aim: ~4:50, ~580 words at ~120 wpm. **Bold** = vocal emphasis.
Slim version — earlier 8-min recording cut down by trimming subtleties and dataset descriptions.

## Pronunciation cheat-sheet

| On the slide | Say it as | Notes |
|---|---|---|
| **QQP** | "Q-Q-P" | Spell the three letters. |
| **PAWS-Wiki** | "paws wiki" | One syllable, like an animal's *paws*. |
| **ParaSCI** | "PA-ruh-sye" | *sci* like in *psy*chology — **not** "ski". |
| **BERT** | "burt" | |
| **BLEU** | "blue" | |
| **BERTScore** | "burt score" | |

---

## Slide 1 — Title  ·  ~10 s

Hi, I'm Jędrzej. Our paper is on **one-step non-autoregressive text generation** with **shortcut flow matching**.

---

## Slide 2 — The bottleneck  ·  ~22 s

The problem in one breath:

Autoregressive models — great quality, but high latency, because they decode one token at a time.

Non-autoregressive — fast, but quality drops.

Diffusion-based non-autoregressive — better quality, but a hundred to a thousand denoising steps, so latency comes back.

Our question: can we generate quality text in **one denoising step**?

---

## Slide 3 — Flow matching  ·  ~28 s

Quick background on flow matching.

In embedding space, draw a straight line from random noise to the target text. Train a network to predict the **velocity** along that path — the local direction of motion. That's the flow-matching loss.

At inference, we sample noise and integrate a few small Euler steps along the trajectory.

---

## Slide 4 — The 1-step problem  ·  ~25 s

In principle, you could decode in one step — start from noise, take one big jump along the predicted velocity, and you're done.

The problem is that the velocity model only knows the **local** direction. The real trajectory curves, so a single big jump overshoots and lands far from any real text embedding.

We need a model that knows where the trajectory is **going overall** — not just where it's pointing right now.

---

## Slide 5 — Our idea: shortcuts  ·  ~32 s

Our idea: extend the network with an extra input — the **step size**, which we call $d$.

When $d$ is near zero, the model behaves like standard flow matching.

When $d$ is large, the model predicts the *average direction* across a longer step — accounting for the curve. We call that a **shortcut**.

In the animation: small step size means many tiny chords; large step size means one big diagonal jump. Either way, both end at the target embedding.

---

## Slide 6 — Self-consistency  ·  ~32 s

How do we *train* the model to produce useful shortcuts? With **self-consistency**.

The idea: one step of size $2d$ should land in the same place as two consecutive steps of size $d$. The animation shows it: two violet arrows — one orange shortcut — same endpoint.

We just turn that into a squared loss.

The clever part: this is a recursion. **One trained model handles every step count** — train once, decode at one step, two steps, four — whatever your latency budget allows.

---

## Slide 7 — Experimental setup  ·  ~14 s

A BERT-base style transformer encoder, around a hundred million parameters, with step size and time injected per layer.

We evaluate on three paraphrase datasets: **Q-Q-P**, **PAWS-Wiki**, and **ParaSCI** [*"para-sye"*].

---

## Slide 8 — Main results  ·  ~32 s

Main BLEU results.

The two non-autoregressive rows decode in **a single step**. The Transformer row is an autoregressive baseline that decodes token by token, shown for context.

Headline: on **PAWS-Wiki**, BLEU **more than doubles** — from 21 to almost 43 — and **matches the Transformer baseline at 42**, with one forward pass instead of one per token.

QQP and ParaSCI also show clear, statistically significant improvements over the flow-matching baseline.

---

## Slide 9 — Takeaways  ·  ~25 s

To wrap up: shortcut flow matching transfers cleanly from vision to text. In a single denoising step, BLEU more than doubles versus the flow-matching baseline. On PAWS-Wiki we even match a Transformer in one forward pass. And one trained model serves any step count.

The paper has more — multi-step decoding sweeps, self-conditioning, classifier-free guidance, BERT-init, and an LLM-based evaluation.

Code at the link. Thanks!

---

## Per-slide timing — target ~4:00, with safety margin

| Slide | Time |
|---|---|
| 1 — Title | 0:10 |
| 2 — The bottleneck | 0:22 |
| 3 — Flow matching | 0:28 |
| 4 — The 1-step problem | 0:25 |
| 5 — Our idea | 0:32 |
| 6 — Self-consistency | 0:32 |
| 7 — Setup | 0:14 |
| 8 — Main results | 0:32 |
| 9 — Takeaways | 0:25 |
| **Total** | **3:40** |

Aim for ~4:00 spoken with natural pauses → fits 5:00 limit comfortably.
If you still come in around 5:30 after recording, ×1.1 speed-up is fine and inaudible.

## What was cut from the longer version

- Coauthor names on slide 1 (slide already lists them).
- "Inference trajectory averages over many conditional paths" on slide 3 — interesting but a footnote.
- "2000 steps" callout on slide 4 — visual already conveys it.
- Frans 2025 citation on slide 5 — credited on slide.
- "Squared loss between the 2d prediction and average of two d predictions" mathematical detail on slide 6 — equation is on screen.
- Dataset descriptions on slide 7 — names only, audience reads.
- Per-dataset numbers on slide 8 (QQP 14→19, ParaSCI 6→14) — the chart shows them; spoken word focuses on the headline.

## Delivery tips

- Speak a little slower than feels natural. Going faster to fit 5 minutes makes you sound rushed; cuts already give you the budget.
- Pause briefly after **"in one forward pass"** (slide 8) — biggest punchline of the talk.
- Don't read the formulas out loud — the audience reads them while you describe what they mean.
- If you finish at 4:30, that's fine. Better than running over.
