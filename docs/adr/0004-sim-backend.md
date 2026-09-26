# ADR-0004: Write our own PyTorch LIF engine (COO/edge-list, CPU-float64 oracle); Brian2 is the parity oracle; eonsystems fly-brain is not a backend

Status: Accepted (2026-09-26, owner approved)

## Context

The brief asked us to evaluate `eonsystemspbc/fly-brain` as the backend before
writing our own. Verified facts (`docs/research/landscape_sim_and_longtail.md` §A2):
it is a GPL-2.0-or-later benchmark harness with one author, 15 commits, no tests,
no CI, no importable API; its PyTorch runner integrates with forward Euler
(`v += dt/tau*(g-(v-v_rest))`), so it cannot be spike-for-spike identical to
Brian2's exact integrator by construction; device is CUDA-or-CPU (no MPS); the
PyTorch path is unseeded; its Brian2 comparison results are not committed.

The reference (Shiu et al. 2024, MIT) uses Brian2 `method='linear'` (= `exact`,
closed-form update per 0.1 ms step), delay 18 steps, refractory 22 steps (0 for
Poisson-driven neurons), schedule `groups → thresholds → synapses → resets`, and
Poisson drive as one 68.75 mV kick per event. `Kisame76/drosophila-brain-mlx`
(MIT) proved that a float64 oracle transcribing Brian2's closed-form coefficients
reproduces Brian2 spike-for-spike on an 800-neuron subnet driven by a fixed input
spike train, and that a float32 GPU lane lands within 5 % of spikes.

Substrate facts verified on this machine (torch 2.14.0): MPS supports sparse COO
matmul and `index_add_` but not CSR, not float64, and `index_add_` is not
deterministic on MPS.

## Decision

1. `flyconn.sim` is our own engine, in PyTorch, with:
   - a **float64 CPU reference path** implementing Brian2's exact per-step update
     (coefficients `exp(-dt/t_mbr)`, `exp(-dt/tau)` and the exact g→v cross term),
     Brian2's slot order, a delay ring buffer and refractory masks; this path is
     bit-deterministic under a seed and is the only path for which we claim
     exactness;
   - a **float32 batched path** (trials on a leading dimension) using the
     connectome as COO / edge list with `torch.sparse.mm` (COO) or gather +
     `index_add_`; CSR only when the device is CPU/CUDA; MPS and CUDA results are
     validated as "within rounding" (spike-count gate), not bit-identical.
2. Validation ladder, each rung a test (per brief):
   1. fixed-spike-train Brian2 gate on small subnets: float64 path spike-for-spike
      equal to Brian2 2.10.x; float32 paths within 5 % of spikes (design copied,
      with attribution, from drosophila-brain-mlx);
   2. Shiu golden results on v630 (GOLDEN_RESULTS.md §3.3), 30 trials;
   3. throughput benchmarks on CPU, MPS, CUDA recorded in `benchmarks/`.
3. Brian2 (`brian2>=2.10`) is a test-only extra; we also run rung 1 against
   `brian2==2.5.1` in the scheduled job when a Python 3.10 environment is
   available, to honour the paper's pin.
4. Poisson input is seeded per trial via the torch generator; parity with Brian2's
   RNG stream is not attempted (impossible), so rung 2 compares distributions.
5. Silencing defaults to zeroing **outgoing** weights (matches the reference code,
   not its README); "silence inputs too" is a flag.

## Alternatives considered

- Adopt eonsystems fly-brain: rejected (Euler, no MPS, unseeded, GPL-2.0, no API).
- Adopt drosophila-brain-mlx: rejected as backend (Apple-only), adopted as design.
- Brian2 as the production engine: rejected for batching/GPU/MPS reasons; kept as oracle.
- Norse/snnTorch/Lava/JAX: no existing Shiu port; a bespoke exact-LIF loop is
  ~200 lines and needs no framework beyond torch.

## Consequences

- Determinism claim is scoped: bit-exact on CPU float64; statistically
  reproducible (seeded) elsewhere. Documented on the caveats page.
- MaleCNS runs are labelled **uncalibrated** until the re-calibration protocol
  (M4) is validated; constants were fitted to FlyWire (Shiu w_syn chosen so 100 Hz
  sugar drive gives ~80 % of maximal MN9 firing).

## Evidence

`docs/research/landscape_sim_and_longtail.md` §A1–A6; `docs/GOLDEN_RESULTS.md` §3.
