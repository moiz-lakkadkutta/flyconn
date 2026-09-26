# Scientific caveats

Grows with each milestone. Seeds:

- A connectome is wiring; it carries no measured dynamics.
- Neurotransmitter identities are classifier predictions (Eckstein et al. 2024 lineage; MaleCNS body-level file gives argmax + confidence only).
- Weights are synapse counts under a detection and confidence threshold that differs per dataset (see `DATA_SOURCES.md`).
- Simulation outputs are model predictions; the Shiu et al. constants were fitted to FlyWire, so MaleCNS runs are uncalibrated until stated otherwise.
- Bit-exact determinism holds on CPU float64 only; MPS/CUDA results are seeded but summation-order dependent.
