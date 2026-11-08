# ADR-0006: A content-hash response cache makes every run free to replay and resumable

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

A single optimization run takes tens of minutes to hours on the shared M1, servers are
restarted when switching models, and the machine is shared with other jobs that can starve
or interrupt a run. Results must be reproducible from committed files without a model.

## Decision

- Key every model call by the SHA-256 of its canonical request (model id, messages,
  `max_tokens`, temperature, seed, stop strings) and store responses in one SQLite file in
  WAL mode with a long busy timeout, shared by all processes and by the DSPy comparison.
- Write each response the moment it arrives (`LlamaCppLM` calls back per request), so an
  interrupted batch resumes from the last finished call, not from the batch start.
- Retry transport errors with capped exponential backoff for about two minutes, so a server
  restart does not abort a run.
- Make optimizers deterministic given their seed and the model's answers. Rerunning a run
  replays its finished prefix from the cache and continues with the first unanswered call;
  run and test result files are written atomically and skipped when complete.
- Count replayed calls at their original usage, so cost columns are identical between the
  first run and any rerun.

## Alternatives considered

- **Checkpoint optimizer state explicitly.** Seven different search loops would each need
  serialization code; replay through the cache gives the same effect for all of them.
- **DSPy's own disk cache for the comparison.** Would split accounting across two caches;
  routing DSPy through the same client keeps one ledger (ADR-0009).

## Consequences

- Tables regenerate from committed result files; any run can be replayed from the cache.
- The cache grows with every distinct request (about 1 KB per call); it lives outside the
  repository in `.bench-data/prompt-optimizer/cache/`.
