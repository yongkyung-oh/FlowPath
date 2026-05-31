# FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-1f6feb.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/39643)
[![DOI](https://img.shields.io/badge/DOI-10.1609%2Faaai.v40i29.39643-blue.svg)](https://doi.org/10.1609/aaai.v40i29.39643)
[![arXiv](https://img.shields.io/badge/arXiv-2511.10841-b31b1b.svg)](https://arxiv.org/abs/2511.10841)

**Published version (preferred citation):** Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 40, no. 29, pp. 24594-24603, 2026.
[Official page](https://ojs.aaai.org/index.php/AAAI/article/view/39643) ·
DOI: [10.1609/aaai.v40i29.39643](https://doi.org/10.1609/aaai.v40i29.39643)

**arXiv preprint:** [arXiv:2511.10841](https://arxiv.org/abs/2511.10841),
DOI: [10.48550/arXiv.2511.10841](https://doi.org/10.48550/arXiv.2511.10841)

**Authors:** YongKyung Oh, Dong-Young Lim, Sungil Kim

**TL;DR:** FlowPath is a learnable control path for neural controlled
differential equations (Neural CDEs). Instead of a fixed interpolation, it
uses invertible neural flows to learn a continuous, data-driven manifold for
robust irregular time series classification, even under heavy missingness.

**Keywords:** neural controlled differential equations · invertible neural
flows · irregular time series classification · learnable control path

## Overview

Modeling continuous-time dynamics from sparse, irregularly-sampled time series
remains a fundamental challenge. Neural controlled differential equations offer
a principled framework, but their performance is highly sensitive to how
discrete observations are lifted into continuous control paths. Most existing
models rely on fixed interpolation schemes that impose simplistic geometric
assumptions and often distort the data manifold, especially under high
missingness.

**FlowPath** is a learnable path construction method built on invertible neural
flows. Instead of linking observations through a predefined interpolant, it
learns a continuous, data-adaptive manifold subject to invertibility
constraints that promote information-preserving and stable transformations.
This inductive bias separates FlowPath from prior unconstrained learnable path
models. On benchmark datasets and a real-world case study, FlowPath improves
classification accuracy over fixed interpolants and non-invertible
architectures, underscoring the value of modeling both the dynamics along the
path and the geometry of the path itself.

## Method

FlowPath extends Neural Differential Equation frameworks through:

* **Invertible path construction:** data-adaptive path via invertible flows.
* **Geometry-aware control paths:** continuous paths that better reflect the
  latent manifold than fixed interpolants.
* **NCDE compatibility:** the learned path plugs into NCDE backbones.

## Code architecture

* `torch-ists/`: utilities and differential-equation models for irregular TS.
* `PAMAP2/`: human activity recognition and the sensor-drop experiment.

## Citation

If you use this software or method, please cite the published AAAI paper:

```bibtex
@article{oh2026flowpath,
  title   = {FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification},
  author  = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {29},
  pages   = {24594--24603},
  year    = {2026},
  doi     = {10.1609/aaai.v40i29.39643},
  url     = {https://ojs.aaai.org/index.php/AAAI/article/view/39643}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
