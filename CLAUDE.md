# CLAUDE.md

General project guide (build, run, key files, coding constraints): @AGENT.md

## Standing goals for EVERY task

Every task in this repository has two goals besides the task itself. They apply to every
change, however small. A task is **not finished** until both are met, or until you have told
the user why one does not apply ("no code or algorithm change, so nothing to update" is a
valid reason, but say it).

### Goal 1: Keep the onboarding guide up to date

The repository has an explanatory website for a **new junior employee on their first day**:
someone who can program but knows nothing about genetic programming, satellite edge
computing, or this codebase.

- **Location:** `docs/onboarding/`. Entry point: `docs/onboarding/index.html`. It is a static
  site: plain HTML/CSS/JS, no build step, opens by double-clicking `index.html`. If the
  folder does not exist yet, create it in the first task that touches code.
- **Update it in the same task** whenever you change behaviour, architecture, algorithms,
  parameters, file layout, build/run commands, or the data pipeline. Pages that describe code
  you changed must match the code when you finish.
- **Audience and tone:** explain as to a junior colleague. Define every term the first time it
  appears (chromosome, fitness, crossover, snapshot, SEN, ...). Go from the big picture to the
  detail: what problem we solve, then how the pieces fit, then each module. Prefer diagrams,
  worked examples and small code excerpts with file links over long prose.
- **What it must cover** (grow it over time): the problem (task scheduling in LEO satellite
  edge computing); the GP approach in plain words and then with the maths; an architecture
  map (C++ core in `include/scheduler`, Python bindings in `bindings.cpp`, Python runners
  `main.py`, `data_processing/`, metrics, experiments); how to build, run and test;
  configuration (`config.json5`, `runConfigurations/`); a glossary; and "where do I start if I
  want to change X".
- Keep a short **changelog page** (`docs/onboarding/changelog.html`) with one dated entry
  per task that changed the guide.
- At the end of the task, tell the user which pages you added or changed.

### Goal 2: Every genetic-algorithm design choice must rest on the literature and on sound maths

The GP/GA solution must be defensible in a thesis defence. For any change to the algorithm
(representation, primitive/terminal set, initialisation, selection, crossover, mutation,
elitism, bloat control, fitness function and its weights, multi-objective handling,
parameters such as population size, depth, rates, generations, evaluation protocol):

- **Cite peer-reviewed sources** (journal/conference papers, established textbooks such as
  Koza 1992/1994, Poli–Langdon–McPhee's *A Field Guide to GP*, Eiben–Smith). Each choice
  must point to at least one paper that proposes, analyses or empirically supports it, in
  this or a closely related setting (GP hyper-heuristics for scheduling, cloud/fog/edge and
  satellite task scheduling).
- **Never invent or guess a citation.** Check every new reference (authors, title, venue,
  year, DOI) against Crossref, the publisher, or the paper itself before adding it. If you
  cannot verify it, say so and do not add it.
- **Record references** in `thesis/references.bib`, and add a reading note to
  `thesis/literature_notes.md`: what the paper says, how much of it was actually read, and why
  it supports the choice. Cite the bib key in code comments next to the relevant
  implementation (e.g. `// tournament selection, see [koza1994gp]`).
- **Give the mathematical formulation:** define the operator or objective formally
  (notation, equations, assumptions), and when relevant give the known theoretical properties
  (selection pressure, schema/building-block arguments, bloat theory, complexity, convergence
  or no-free-lunch caveats, Pareto dominance for multi-objective fitness). Put it in the thesis
  (`thesis/chapters/method.tex` or the right chapter) and a readable version in the onboarding
  guide.
- **Back empirical claims with experiments:** fixed seeds, several independent runs, and an
  appropriate statistical test (e.g. Wilcoxon rank-sum / Mann-Whitney U with effect size),
  as in `thesis/experiments/`. Do not call one setting "better" from a single run.
- **If a requested change has no support in the literature**, or contradicts it, tell the user
  before implementing it. Name what the literature recommends instead, and let them decide.
  If they still want it, label it clearly as a novel or heuristic choice in the code, thesis
  and guide.
- At the end of the task, list for the user the references that back the change.
