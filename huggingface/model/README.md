---
license: other
license_name: merlin-research-release
base_model: microsoft/deberta-v3-xsmall
library_name: transformers
pipeline_tag: text-classification
tags:
  - agents
  - agent-safety
  - tool-use
  - prompt-injection
  - deberta-v3
---

# Merlin

Merlin is a 70.8M-parameter local encoder that classifies a proposed AI-agent
tool invocation as `safe` or `unsafe` using its surrounding context.

This repository contains the portable checkpoint. Use the reference package at
<https://github.com/kontext-security/merlin>; a generic Transformers pipeline
does **not** reproduce the benchmark because Merlin uses four independently
budgeted fields, deterministic history normalization, and a validation-fitted
calibrator.

## Inputs

1. user request
2. prior interaction history
3. current tool name and arguments
4. tool descriptions/schemas

Prior ReAct history is converted to canonical JSON tool/argument/observation
events. `Thought` and `Final Answer` text is excluded. The current action is
reduced to tool name and arguments. Each field has its own token budget within a
512-token packed sequence.

## Results

Strict binary evaluation on TS-Bench (`0.0` safe; `0.5` and `1.0` unsafe), fixed
0.5 threshold:

| Split | N | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| All TS-Bench | 7,182 | 91.19% | 92.66% | 88.45% | 90.51% |
| ASB-Traj | 5,231 | 99.73% | 99.76% | 99.68% | 99.72% |
| AgentDojo-Traj | 1,220 | 71.80% | 51.14% | 50.85% | 51.00% |
| AgentHarm-Traj | 731 | 62.38% | 83.42% | 59.43% | 69.41% |

The pooled score is dominated by ASB-Traj and is not evidence of uniform
cross-environment performance. See the GitHub repository for the full protocol,
preprocessing code, data provenance, system measurements, and limitations.

## Training

- Base: `microsoft/deberta-v3-xsmall` at revision
  `4b419818330868dff6a60ad3e6b1c730f8b8c0c6`
- Train: 2,192 examples (841 safe, 1,351 unsafe)
- Validation: 789 examples
- Epochs: 5; learning rate: 2e-5; batch size: 8; gradient accumulation: 2
- Sigmoid calibration coefficient: 1.427213430140093
- Sigmoid calibration intercept: 2.953687013257505
- Default threshold: 0.5

## Intended use

Use Merlin as a low-latency, local safety signal before executing an agent tool
call. It is not a complete authorization layer and should be combined with
least privilege, deterministic policy, sandboxing, and human confirmation for
consequential actions.

Do not use it as the sole control for high-impact actions, as a general content
moderator, or outside the documented input representation without evaluation.

## Data and license

The base model is MIT licensed. Merlin was fine-tuned on TS-Bench data from
<https://github.com/MurrayTom/ToolSafe>. That repository had no explicit
repository license at release time, so the checkpoint is marked `other` and no
raw TS-Bench examples are mirrored here. Review upstream terms before commercial
use or redistribution. Reference code is Apache-2.0.

