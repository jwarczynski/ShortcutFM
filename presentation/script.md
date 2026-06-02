# 5-minute talk script — One-step NAR Text Generation with Shortcut Flow Matching

Aim: ~5:00. Word count ~720, delivered at ~150 wpm.
Notes for the speaker are in *italics* / `[brackets]`. **Bold** = vocal emphasis.

---

## Slide 1 — Title  ·  ~15 s

Hi everyone, I'm Jędrzej Warczyński. This is joint work with Ondřej Dušek and Mateusz Lango, on **one-step non-autoregressive text generation with shortcut flow matching**.

`[advance]`

---

## Slide 2 — The bottleneck  ·  ~30 s

Let me set up the problem.

Autoregressive models are great in quality, but slow — they decode one token at a time, so latency scales with the output length.

Non-autoregressive models predict all tokens in parallel, but pay a quality penalty.

Recent diffusion-based NAR methods recover most of that quality — but at the cost of hundreds or thousands of denoising steps. So latency is back.

Our question: can we generate high-quality text in **one denoising step**?

`[advance]`

---

## Slide 3 — Flow matching  ·  ~40 s

We start from flow matching. We define a straight-line path in embedding space, from random noise z₀ to the target z₁, parameterised by t. We then train a network f-theta to match the **velocity** along that path — and that's the flow-matching loss, L-FM.

At inference, we sample noise and integrate a few small Euler steps along the learned trajectory.

One detail worth noting: although the *training* path is straight, the *inference* trajectory the model actually follows is curved — because at each point, f-theta averages over many possible target paths.

`[advance]`

---

## Slide 4 — The 1-step problem  ·  ~32 s

In principle, flow matching can decode in one step — just take z₀ and add f-theta at zero. But f-theta only knows the **local** velocity. And because the real trajectory curves, this one-step prediction overshoots — landing far from any real text embedding.

That's why prior NAR continuous-diffusion approaches use, for example, two thousand steps to be competitive.

What we need is a model that knows where the trajectory is **going** — not just where it's pointing right now.

`[advance]`

---

## Slide 5 — Our idea: shortcuts  ·  ~38 s

Our solution: extend the network with a **step-size argument**, d. So we go from f-theta of z_t and t, to s-theta of z_t, t, **and d**.

When d goes to zero, the model recovers standard flow matching — predicting local velocity.

When d is larger, the model predicts an *average direction* over a step of that size — accounting for the curve ahead.

`[let the animation play through one or two cycles — points to the chord chain]`

You can see it in the animation: with very small d, many tiny steps; with large d, one big jump — both landing at z₁.

We adapt this idea from Frans and colleagues at ICLR 2025, originally proposed for image generation.

`[advance]`

---

## Slide 6 — Self-consistency  ·  ~42 s

How do we train s-theta to actually take useful shortcuts? With **self-consistency**.

The idea: one step of size 2d should equal two consecutive steps of size d.

`[let the animation cycle once — point to the violet steps, then the orange shortcut]`

Two violet steps of size d land at z_{t+2d} — and one orange step of size 2d lands at exactly the same point.

We turn this into a squared loss between the 2d prediction and the average of two d predictions — that's L-SC.

This recursion bootstraps shortcuts of *any* size. So **one model handles every step count** — train once, decode at one, two, four, or however many steps you want.

`[advance]`

---

## Slide 7 — Experimental setup  ·  ~28 s

For experiments, the architecture is a BERT-base style transformer encoder — about a hundred million parameters. The step size d and time t are embedded and injected into every block.

We evaluate on three paraphrase datasets: QQP — short Quora questions, PAWS-Wiki — Wikipedia structural rewrites, and ParaSCI — scientific paper sentences.

We report BLEU and BERTScore, ROUGE, and an LLM-based metric, Themis.

`[advance]`

---

## Slide 8 — Main results  ·  ~42 s

Here are the BLEU results. The two NAR rows — flow-matching baseline and our shortcut model — both decode in **a single step**. The Transformer AR row is an autoregressive upper baseline that decodes token by token, shown for context.

On QQP we go from 14 to 19 — a clear improvement.

On PAWS-Wiki, BLEU **more than doubles**, from 21 to almost 43 — and that actually matches the Transformer AR baseline at 42, with one forward pass instead of one per token.

On ParaSCI, BLEU more than doubles again — six to 13.

All gains over the flow-matching baseline are statistically significant at p less than zero point zero zero one.

`[advance]`

---

## Slide 9 — Takeaways  ·  ~28 s

To wrap up: shortcut flow matching adapts cleanly from vision to text.

In one denoising step, BLEU more than doubles versus the flow-matching baseline at the same step count. On PAWS-Wiki we even match a Transformer AR baseline in a single forward pass.

And one model handles all step counts — no retraining for different decoding budgets.

The paper has more — multi-step decoding sweeps, combinations with self-conditioning and classifier-free guidance, BERT-init strategies, and LLM-based evaluation.

Code and paper at the link. Thanks!

---

## Estimated total: 4 min 55 s

| Slide | Time |
|---|---|
| 1 — Title | 0:15 |
| 2 — Bottleneck | 0:30 |
| 3 — Flow matching | 0:40 |
| 4 — 1-step problem | 0:32 |
| 5 — Our idea | 0:38 |
| 6 — Self-consistency | 0:42 |
| 7 — Setup | 0:28 |
| 8 — Main results | 0:42 |
| 9 — Takeaways | 0:28 |
| **Total** | **4:55** |

## Delivery tips

- **Pace yourself on slides 5 and 6** — those have animations cycling in the background. Let one cycle play before you advance, otherwise the room only sees a frozen first frame.
- The phrase "**one model handles every step count**" on slide 6 is the strongest punchline of the methods section — slow down for it.
- On slide 8, when you say "matches the Transformer AR baseline in **one forward pass**" — that's the headline of the whole talk. Pause for half a second before "one forward pass."
- If you're running long, the cuttable line is "All gains over the flow-matching baseline are statistically significant at p less than zero point zero zero one" on slide 8 — the audience reads it, you don't have to say it.
- If you finish in 4:30, that's fine — gives you a buffer for the chair's transition to questions.
