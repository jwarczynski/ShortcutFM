# 5-minute talk script — One-step NAR Text Generation with Shortcut Flow Matching

Aim: ~5:00, ~720 words at ~150 wpm. **Bold** = vocal emphasis.
Read it out loud — equations on the slide are visual, the spoken words are paraphrases.

---

## Slide 1 — Title  ·  ~15 s

Hi everyone, I'm Jędrzej Warczyński. This is joint work with Ondřej Dušek and Mateusz Lango. Our paper is about **one-step non-autoregressive text generation** with **shortcut flow matching**.

---

## Slide 2 — The bottleneck  ·  ~30 s

Let me set up the problem.

Autoregressive models give great quality, but they're slow — they decode one token at a time, so latency scales with the output length.

Non-autoregressive models predict all tokens in parallel. That's fast, but quality drops.

Recent diffusion-based non-autoregressive methods recover most of that quality — but only by running hundreds or thousands of denoising steps. So the latency is back.

Our question is simple: can we generate high-quality text in **a single denoising step**?

---

## Slide 3 — Flow matching  ·  ~40 s

Our starting point is flow matching.

The idea is: in embedding space, draw a straight-line path from random noise on one end to the target text on the other. We then train a network to predict the **velocity** along that path — basically, "in which direction should I move at this point in time."

That's the **flow-matching loss**, shown on the slide.

At inference time, we sample noise and integrate a few small Euler steps along the learned trajectory.

There's an important subtlety, though: the *training* path is a straight line for a fixed pair of noise and target. But at inference, the model has to handle *any* noise input — and so the trajectory it actually follows curves, because it averages over many possible target paths.

---

## Slide 4 — The 1-step problem  ·  ~32 s

In principle, you could decode in one step — start from noise, take one big jump along the predicted velocity, and you're done. The problem is that the velocity model only knows the **local** direction. And because the real trajectory curves, that one-step jump overshoots and lands far from any real text embedding.

That's why prior continuous-diffusion methods for text use, say, two thousand denoising steps to stay competitive.

What we actually need is a model that knows where the trajectory is **going overall** — not just where it's pointing at the current moment.

---

## Slide 5 — Our idea: shortcuts  ·  ~38 s

So here's our idea: extend the network with an extra input — the **step size**, which we call $d$.

When $d$ goes to zero, the model behaves like standard flow matching — predicting the local velocity.

When $d$ is larger, the model predicts the *average direction* across a longer step — accounting for the curve ahead. We call that prediction a **shortcut**.

You can see it in the animation: with a very small step size, the model takes many tiny chords along the curve; with a large step size, it takes one big diagonal jump. Either way, both end at the same target embedding.

We adapt this idea from Frans and colleagues at ICLR 2025, who proposed it for image generation.

---

## Slide 6 — Self-consistency  ·  ~42 s

The natural question is: how do we *train* the model to produce useful shortcuts?

The trick is **self-consistency**.

The idea: one step of size $2d$ should land in the same place as two consecutive steps of size $d$.

You can see it in the animation. The two violet arrows are two small steps. The orange dashed arrow is one shortcut step that's twice as big — and it ends at exactly the same point.

We just turn that into a squared loss — that's the **self-consistency loss** on the slide.

The clever part is that this is a recursion — small shortcuts bootstrap into larger ones. Which means **one trained model handles every step count**: train once, decode at one step, two steps, four steps — whatever your latency budget allows.

---

## Slide 7 — Experimental setup  ·  ~28 s

The architecture is a BERT-base style transformer encoder — about a hundred million parameters — with the step size and time embedded and injected into every block.

We evaluate on three paraphrase datasets: QQP, which is short Quora questions; PAWS-Wiki, which is Wikipedia sentences with structural rewrites; and ParaSCI, scientific paper sentences.

We report BLEU and BERTScore, plus ROUGE and an LLM-based metric called Themis.

---

## Slide 8 — Main results  ·  ~42 s

Here are the main BLEU results.

The two non-autoregressive rows — the flow-matching baseline and our shortcut model — both decode in **a single step**. The Transformer row is an autoregressive upper baseline that decodes token by token, shown for context.

On QQP we go from 14 to 19 — a clear improvement.

On PAWS-Wiki, BLEU **more than doubles**, from 21 to almost 43 — which actually **matches the Transformer baseline at 42**. We're getting that quality in **one forward pass**, instead of one per token.

On ParaSCI, BLEU more than doubles again — from 6 to almost 14.

All these gains over the flow-matching baseline are statistically significant.

---

## Slide 9 — Takeaways  ·  ~28 s

To wrap up.

Shortcut flow matching transfers cleanly from vision to text generation.

In a single denoising step, BLEU more than doubles compared to the flow-matching baseline at the same step count — and on PAWS-Wiki, we even match a Transformer baseline in one forward pass.

And the same model serves any step count, so we don't have to retrain for different latency budgets.

There's more in the paper — multi-step decoding sweeps, combinations with self-conditioning and classifier-free guidance, BERT-initialisation strategies, and an LLM-based evaluation.

Code and paper at the link. Thanks!

---

## Per-slide timing

| Slide | Time |
|---|---|
| 1 — Title | 0:15 |
| 2 — The bottleneck | 0:30 |
| 3 — Flow matching | 0:40 |
| 4 — The 1-step problem | 0:32 |
| 5 — Our idea: shortcuts | 0:38 |
| 6 — Self-consistency | 0:42 |
| 7 — Experimental setup | 0:28 |
| 8 — Main results | 0:42 |
| 9 — Takeaways | 0:28 |
| **Total** | **4:55** |

## Delivery tips

- Slides 5 and 6 have looping animations. Speak at a normal pace — by the time you finish each slide's text the animation will have shown one or two full cycles.
- The strongest punchline of the methods half is on slide 6: **"one trained model handles every step count."** Slow down there.
- The strongest punchline of the results half is on slide 8: **"matches the Transformer baseline in one forward pass."** Half-second pause before "one forward pass."
- If you're running long, you can drop the "statistically significant" line on slide 8 — it's on the screen.
- Aim to finish around 4:45–4:55 so you don't have to rush the takeaway.
