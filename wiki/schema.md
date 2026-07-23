---
tags: [meta]
last_updated: 2026-07-23
---

# Schema — How this wiki works

This wiki is a persistent knowledge base for the ShortcutFM project. It's maintained by LLM agents based on conversations, code inspection, experiment results, and external sources. Humans curate and ask questions; agents do the bookkeeping.

## Directory layout

```
wiki/
├── schema.md       This file — conventions and workflows
├── index.md        Catalog: all pages with one-line summaries
├── log.md          Append-only chronological log
├── roadmap.md      What's next: planned runs and branch logic
│
├── experiments/    Per-run pages (config, hyperparams, outcome, wandb link)
├── bugs/           Bugs found and how they were fixed
├── decisions/      Design choices with rationale
├── concepts/       Reusable research ideas (masked diffusion, shortcuts, etc.)
└── infrastructure/ Clusters, wandb, datasets, disk
```

## Page conventions

**Every page starts with YAML frontmatter:**

```yaml
---
tags: [masked-diffusion, training, bug]
status: fixed           # open | in-progress | fixing | fixed | wontfix | planned | complete (bugs/decisions/experiments)
date: 2026-07-21        # creation or primary event date
related: [[exp-masked-v1-qqp]]   # wiki-links to related pages
---
```

Frontmatter is Obsidian Dataview-friendly. Keep tags lowercase-hyphenated.

**Wiki links** use Obsidian style: `[[page-name]]` (file name without `.md`). Refer to files by stem, not path.

**Code references** use repo-relative paths so agents can read them directly:
`shortcutfm/masked_criteria.py:MaskedDiffusionCriterion`.

**Commit references** use short SHA: ``c647a52``.

**Wandb runs:** include run ID plus project (`jedrasowicz/Thesis/r43zphwi`).

**SLURM jobs:** include job ID plus cluster (`995218 on hgx2`).

## Page types

**Experiment page** — one entry per training/eval run (or tightly-coupled pair of runs). Config file, hyperparameters, wandb link, SLURM job ID, checkpoint path, outcome, what we learned.

**Bug page** — symptom, root cause, fix (with commit SHA), how to detect, status. Keep these even after fixed — future agents need to know why the code looks the way it does.

**Decision page** — a non-obvious choice we made. What alternatives were considered, why we picked this one, what would change our minds.

**Concept page** — reusable research idea. Definition, why it matters here, implementation pointers, references.

**Infrastructure page** — external systems (clusters, wandb, datasets). Credentials referenced by name, not value.

## Workflows

### Ingesting a new source or conversation

1. Identify which pages are affected (read `index.md`, scan related areas).
2. Update those pages in place — don't create duplicates.
3. Append to `log.md` with `## [YYYY-MM-DD] ingest | <source>`.
4. Update `index.md` if new pages were created or existing summaries changed.

### Answering a question

1. Read `index.md` to find candidate pages.
2. Read candidates in full.
3. Synthesize answer. Cite pages via `[[page-name]]`.
4. If the answer is substantive, consider creating a new page. **Good answers should be filed back into the wiki.**

### Lint pass (periodic)

Check for:
- Contradictions between pages (e.g. two pages disagreeing on a hyperparameter).
- Orphan pages (no inbound links).
- Stale claims (newer experiments overrode an older conclusion).
- Missing concept pages (something referenced everywhere but no dedicated page).
- Dead links (`[[broken]]`).

### Creating a new page

Use the existing naming pattern. Prefer descriptive kebab-case: `exp-masked-v1-qqp.md`, not `mv1.md`. Keep one concept per page.

## Style

- Present tense for current state ("The criterion masks tokens with probability t/T").
- Past tense for experiments and decisions ("We trained 50k steps and observed…").
- No marketing language. Be specific. Include numbers (loss values, step counts, BLEU, dates).
- Prefer bullet points for enumerations, prose for reasoning.
- Include relevant commit SHAs, SLURM job IDs, wandb run IDs.
- If something is uncertain or speculative, mark it as such.

## What this wiki is not

- **Not a tutorial.** It's for agents and humans who already know the domain.
- **Not API docs.** Code docstrings serve that purpose.
- **Not a notebook.** Prefer stable pages over ad-hoc analysis dumps. Ad-hoc work goes in `log.md` with links to any artifacts.
