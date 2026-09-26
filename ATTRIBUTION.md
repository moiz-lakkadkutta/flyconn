# Attribution

Ideas and reference implementations flyconn draws on (all permissively licensed;
no code copied from GPL projects):

- Shiu PK, Sterne GR, Spiller N, et al. *A Drosophila computational brain model reveals sensorimotor processing.* Nature 634, 210–219 (2024). Model definition and constants; repository github.com/philshiu/Drosophila_brain_model (MIT).
- Kisame76/drosophila-brain-mlx (MIT): fixed-spike-train Brian2 parity gate, float64 oracle pattern, pack manifest, degree-preserving shuffle design.
- YijieYin/connectome_interpreter (MIT): multi-hop effective-connectivity kernels, used as an optional dependency.
- sjcabs/fly_connectome_data_tutorial (MIT code): harmonized metadata vocabulary.
- eonsystemspbc/fly-brain (GPL-2.0+): *ideas only* (trial batching, comparison metrics); no code reused.

Datasets are cited per result via `citations()`; see `docs/DATA_SOURCES.md`.
