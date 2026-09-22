# Dataset provenance

Merlin was trained and evaluated with the data released for TS-Bench in the
ToolSafe project:

- Repository: <https://github.com/MurrayTom/ToolSafe>
- Paper: <https://arxiv.org/abs/2601.10156>
- Evaluation commit: `46358fa424a927a895c6c8322f99032c4eb5155e`
- Training/validation commit: `e54c46326353fc8ed286f2d602ac28f616defff3`

The training split contains step-level examples derived from AgentAlign and ASB.
The evaluation split contains AgentDojo-Traj, AgentHarm-Traj, and ASB-Traj. The
strict binary target maps TS-Bench score `0.0` to safe and scores `0.5` or `1.0`
to unsafe.

## Why the JSON is not mirrored here

The upstream ToolSafe repository did not contain an explicit `LICENSE` file when
this release was prepared on 2026-09-22. Public availability is not by itself a
grant to redistribute or relicense a dataset. We therefore publish the pinned
source URLs, byte sizes, and SHA-256 hashes in
[`manifests/ts_bench_assets.json`](manifests/ts_bench_assets.json), plus a fetcher
that downloads the original files from the authors' repository.

Users are responsible for reviewing ToolSafe and the underlying AgentAlign,
ASB, AgentDojo, and AgentHarm terms before use or redistribution. If the ToolSafe
authors add or clarify a redistribution license, a hosted dataset mirror can be
added without changing the experimental identity because the source objects are
already checksum-pinned.

## Safety

These files contain adversarial instructions and potentially harmful tool calls.
Treat every field as inert, untrusted text. Never execute actions from the data.

