# BOOTH Evaluation Harness (Tier-4)

A manual-run harness for measuring how well BOOTH performs end-to-end. Not run
in CI because it's **slow**, **non-deterministic**, and requires a real LLM +
Neo4j. Think of it as the "check it actually works in the wild" stage.

The harness answers three questions:

1. **Cache hit rate**: of the N questions in a test set, how many hit an
   approved FewShot after `K` curation rounds?
2. **Cache precision**: when we hit the cache, is the returned answer correct?
3. **Latency**: what's the p50 / p95 end-to-end latency for a cache hit?

## Layout

```
eval/
├── README.md            <- you are here
├── run_eval.py          <- entry point; prints a summary table
├── testsets/
│   └── starter.yaml     <- 20 example questions with expected answers
└── results/             <- JSON dumps of past runs (git-ignored)
```

## Running

```bash
# From the package root
pip install -e ".[dev]"
python eval/run_eval.py \
  --testset eval/testsets/starter.yaml \
  --out eval/results/$(date +%Y%m%d_%H%M%S).json
```

The harness loads each question, calls `BOOTHRetriever.query`, and records:

- whether the response was a cache hit
- the returned answer
- the expected answer
- a simple string-contains match (swap in an LLM judge for real evaluation)
- end-to-end latency

## Writing a test set

Test sets are YAML files with the following shape:

```yaml
name: "Starter questions"
description: "A small bootstrap set; replace with your domain's questions."
questions:
  - text: "How many users are there?"
    expected_keywords: ["users", "count"]
    expected_to_hit: true

  - text: "What countries does our data cover?"
    expected_keywords: ["United Kingdom", "United States"]
    expected_to_hit: true
```

`expected_to_hit: true` means "by the time you run eval, this question should
match an approved Query template." Use `false` for questions you expect to
still be in the curation queue.

## What this harness is NOT

- It's not a replacement for unit / integration tests. Those still run in CI
  every push.
- It's not a benchmark vs. other retrievers. For that, compare against a
  vanilla `VectorRetriever` from neo4j-graphrag on the same Neo4j.
- It doesn't grade answer quality beyond keyword presence. For semantic
  matching, plug in an LLM judge (see `run_eval.py`'s `--judge` flag, which is
  currently a placeholder).
