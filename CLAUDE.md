# ShortcutFM

## Wiki — read first, update often

`wiki/` is the project's persistent knowledge base, maintained by agents across conversations. It is the single source of truth for project context, decisions, experiments, bugs, and learned tips that don't live in code.

**Before starting any non-trivial task:**
1. Read `wiki/index.md` for the catalog of pages.
2. Read `wiki/schema.md` once per session to understand conventions (frontmatter, `[[wiki-links]]`, page types).
3. Read every `wiki/` page that looks relevant — prior experiments, known bugs, decisions, cluster notes. Past agents have already done much of the work; don't redo it or contradict it.
4. Skim `wiki/log.md` for recent activity.

**During and after work, keep the wiki current:**
- New experiment run? Add or update a page under `wiki/experiments/` (config, hyperparams, wandb link, SLURM job ID, outcome).
- Found and fixed a bug? Create or update a page under `wiki/bugs/` with symptom, root cause, fix commit SHA.
- Made a non-obvious design choice? Add to `wiki/decisions/` with alternatives considered and rationale.
- Learned a useful tip, gotcha, command incantation, or cluster quirk? File it under the relevant `wiki/infrastructure/` or `wiki/concepts/` page.
- Append a one-line entry to `wiki/log.md` (`## [YYYY-MM-DD] <type> | <title>`, newest first) for any substantive change.
- Update `wiki/index.md` whenever you create a new page or materially change an existing summary.
- Update `wiki/roadmap.md` when priorities or next steps shift.

Treat wiki updates as part of the task, not optional cleanup. If a conversation produced new knowledge — about the codebase, the clusters, an experiment, a bug, a design choice — that knowledge belongs in the wiki before the conversation ends. Future agents (and your future self) depend on it.
