# ADR-0010: Report bootstrap intervals and paired differences on shared examples

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

Test sets are a few hundred examples, so accuracy differences of a few points are within
sampling noise. Comparing two methods by whether their separate intervals overlap is
conservative and ignores that both were scored on the same examples.

## Decision

- Each accuracy carries a 95% percentile bootstrap interval over examples (10,000
  resamples, fixed seed).
- Each method is compared with the seed prompt (and DSPy rows with each other) by a paired
  bootstrap of the per-example difference on the examples both scored, with a two-sided
  p-value; McNemar's exact test is available for binary scores.
- Tables are computed only on examples every row has scored, and say how many.

## Alternatives considered

- **Normal-approximation intervals.** Fine for accuracy near 50%, poor near 0 or 100% and
  for F1-style scores; the bootstrap handles both.
- **Multiple optimization seeds only.** Captures search variance but not test-sample
  variance; both are reported where compute allowed extra seeds.

## Consequences

- Small differences are reported as ties when the paired interval contains 0.
- Intervals are reproducible to the digit from the committed per-example records.
