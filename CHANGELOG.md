# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.2.0] 2026-11-09

### Features

- Llama.cpp backend with content-hash cache, async batching and cost accounting (#8)
- GSM8K, BIG-Bench Hard, HotpotQA, MATH-500 and IFEval tasks with seeded splits (#9)
- Seven prompt optimizers, experiment runner, bootstrap statistics and report (#10)
- DSPy 3.4.1 head-to-head harness, design records and architecture diagram (#11)
- Length clause in task contracts, report analyses and run robustness (#16)
- DSPy head-to-head table and selection optimism net of the seed's validation gap (#19)

### Bug fixes

- Robust long runs and results location; data, model and threat cards (#12)
- Lean llama-server settings for the shared 8 GB machine (#18)
- Stop a stale llama-server reliably before starting the next model (#20)
- A GSM8K answer too long for a float scores 0 instead of stopping the run (#21)

### Documentation

- Complete citation metadata and bump it on release (#6)
- Diagrams on white in the shared theme, make diagram for every diagram, and a self-contained Definition of Done (#13)
- First real-model results for the seed prompt, bootstrapped few-shot and DSPy zero-shot (#17)
- Citation metadata for the real-model release (#24)

### Build and CI

- Build and attach the sdist and wheel to every release (#14)
- DSPy's optional optuna dependency so the MIPROv2 head-to-head runs (#22)

### Maintenance

- Latest toolchain and CI versions (#5)
- Release script keeps hard-coded version strings in step (#7)
- Release script keeps nested crates and packages on the release version (#15)

## [0.1.0] 2026-11-03

### Features

- Add core prompt model, simulated language model, and metrics
- Add bootstrap, instruction-search, and compound optimizers
- Add bundled task, ablation evaluation, and command line interface
- Add a deterministic optimizer micro-benchmark
- Add HTTP service, stress benchmark, and operations documentation
- Add production observability (metrics and readiness) and reword footer

### Documentation

- Write README, architecture diagram, and responsible-AI cards
- Use plain, factual wording (#14)
- Standard README structure and language framing (#16)
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)
- Describe the supply chain accurately (#3)

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Scaffold prompt-optimizer project
- Bump continuous integration action versions and dev dependencies
- Remove Dependabot configuration
- Release tooling, changelog and agent guides (#4)
