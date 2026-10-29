# End-to-End Case Study - Compiling and Serving a Prompt

How a naive prompt becomes a production classifier, and what each step buys.

## Scenario

A team must label customer messages as `refund`, `shipping`, or `cancel`. They start with
a naive prompt ("Answer the customer message with a single word.") that scores **0.333**
on held-out data: with no label vocabulary and no examples, the model guesses.

## Compiling

```bash
python -m prompt_optimizer compile
```

The compound optimizer searches instructions and demonstrations against a held-out
validation split and emits a prompt that pairs a vocabulary-bearing instruction with one
well-chosen demonstration. On the held-out test split it scores **1.000** (see
`python -m prompt_optimizer evaluate`), the gradient being baseline 0.333, instruction
search 0.833, few-shot 0.833, compound 1.000.

## Serving

```bash
python -m prompt_optimizer serve --port 8000
curl -s -X POST localhost:8000/predict -d '{"text": "I need a refund for my money"}'
```

```json
{"text": "I need a refund for my money", "label": "refund"}
```

The compile happens once at startup; each prediction is a nearest-neighbour lookup over a
handful of demonstrations, serving on the order of 100000 requests per second per core.

## What an operator sees

- **Cost shape.** Expensive once (compile), cheap forever (predict). The opposite of a
  per-request model call.
- **Reproducibility.** Compilation is deterministic, so the served prompt is auditable and
  versionable as configuration.
- **Quality control.** The `/compile` endpoint returns the exact instruction and
  demonstrations in production, so the behaviour is fully inspectable.

## Takeaway

Treating the prompt as something you compile against a metric turns prompt engineering
into a reproducible build step whose output is cheap to serve and easy to audit.
