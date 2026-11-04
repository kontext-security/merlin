# Merlin

**V3 ungated evaluation release:** the native Go runtime now includes the V3
contextual encoder with the three restrictive eligibility gates removed. Model
weights, threshold 0.5, action agreement and the complete-action 128-token limit
are unchanged. See [V3 usage, evaluation and limitations](docs/V3_UNGATED.md) and
[aggregate results](results/v3-ungated.json). Private evaluation examples are not
included in this repository or the Hugging Face package.

The Python package and checkpoint instructions below describe **legacy v0.1**.
They do not implement V3; use the native runtime linked above for V3.

## Legacy v0.1

Merlin is a small, local classifier for **contextual safety at the moment an AI
agent invokes a tool**. It considers the user's request, prior tool activity,
the proposed tool call, and the available tool descriptions, then returns a
calibrated `safe` or `unsafe` verdict.

The released checkpoint is a fully fine-tuned DeBERTa-v3-xsmall encoder with
70.8M parameters. It runs without a hosted model or network request after the
model has been downloaded.

The portable checkpoint is published on Hugging Face at the
[pinned Merlin v0.1 revision](https://huggingface.co/kontext-security/Merlin/tree/21b6f2cc381dd036160bf237877780bca72ceea5).
The `model.safetensors` SHA-256 is
`075d68f8bb4c1b4b98e2d2b2c0d5b013056b202172b080b34e7364b698844c86`.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e .

printf '%s' '{
  "user_request": "Summarize my unread support messages",
  "history": [],
  "current_action": {
    "name": "send_message",
    "arguments": {"channel": "public", "text": "internal credentials"}
  },
  "tool_descriptions": [{
    "name": "send_message",
    "description": "Send a message to a channel"
  }]
}' | .venv/bin/merlin
```

Python usage:

```python
from merlin_guard import Invocation, MerlinGuard

guard = MerlinGuard.from_pretrained("kontext-security/Merlin")
verdict = guard.predict(
    Invocation(
        user_request="Summarize my unread support messages",
        history=[],
        current_action={
            "name": "send_message",
            "arguments": {"channel": "public", "text": "internal credentials"},
        },
        tool_descriptions=[
            {
                "name": "send_message",
                "description": "Send a message to a channel",
            }
        ],
    )
)
print(verdict.to_dict())
```

## What the model sees

Merlin uses four independently budgeted fields:

| Field | Meaning | Token budget |
|---|---|---:|
| `user_request` | The user's current instruction | 96 |
| `history` | Earlier tool calls and observations | 144 |
| `current_action` | Proposed tool name and arguments | 128 |
| `tool_descriptions` | Tool names, schemas, and descriptions | 128 |

Raw ReAct history is normalized into deterministic JSON events. Prior `Thought`
and `Final Answer` text is removed. The current action is reduced to tool name
and arguments. Structured JSON histories and tool calls are accepted directly.
The exact serializer used during training is in
[`src/merlin_guard/preprocessing.py`](src/merlin_guard/preprocessing.py).

## Evaluation

We evaluated the fixed checkpoint on the 7,182 step-level examples in TS-Bench
using the strict binary label (`0.0` safe; `0.5` and `1.0` unsafe) and a
validation-fitted sigmoid calibrator. The default operating threshold is 0.5.

| Split | N | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| All TS-Bench | 7,182 | 91.19% | 92.66% | 88.45% | 90.51% |
| ASB-Traj | 5,231 | 99.73% | 99.76% | 99.68% | 99.72% |
| AgentDojo-Traj | 1,220 | 71.80% | 51.14% | 50.85% | 51.00% |
| AgentHarm-Traj | 731 | 62.38% | 83.42% | 59.43% | 69.41% |

The pooled result is strongly influenced by ASB-Traj. The source-level results
show that cross-benchmark generalization remains the main limitation; do not
read 91.19% as uniform performance across agent environments.

On an Apple Silicon MPS test host, single-example latency was 47.49 ms p50,
48.35 ms p95, and 48.70 ms p99. The artifact was 278.04 MiB and loaded in
463.96 ms p50. These measurements are host- and runtime-specific.

## Reproducing the data inputs

The benchmark examples are not copied into this repository. TS-Bench is fetched
from two pinned commits of the authors' public repository and every file is
verified against the published manifest:

```bash
python scripts/fetch_ts_bench.py --output data/ts-bench --kind eval
python scripts/fetch_ts_bench.py --output data/ts-bench --kind train --kind validation
```

See [DATASETS.md](DATASETS.md) before downloading or redistributing the data.
The upstream TS-Bench repository currently has no explicit repository license,
so this project publishes provenance and a reproducible fetcher rather than
relicensing or mirroring the authors' JSON files.

## Scope and limitations

- Merlin classifies one proposed tool invocation in context. It is not a prompt
  injection detector alone: unsafe direct requests, indirect prompt injection,
  unsafe actions, and policy violations can all appear in the training labels.
- A verdict is a defense-in-depth signal, not a complete authorization system.
  Keep deterministic permissions, user confirmation, sandboxing, and audit logs.
- Inputs that differ from the released TS-Bench representation may shift model
  behavior. Validate on the actual tools, policies, and traffic of a deployment.
- The score is calibrated on the released validation split, not on production
  prevalence. Tune policy thresholds only with a separate, representative set.
- Benchmark content is untrusted text. The fetch and preprocessing code never
  executes tool calls contained in the data.

Read the accompanying [Merlin article](https://kontext.security/blog/merlin) for
the motivation and design discussion.

## License

The code in this repository is Apache-2.0. The checkpoint is derived from the
MIT-licensed `microsoft/deberta-v3-xsmall` base model; its release card records
the additional benchmark-data terms and limitations. Upstream datasets retain
their own licenses and terms.
