# FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification

[![arXiv](https://img.shields.io/badge/arXiv-2511.10841-b31b1b.svg)](https://arxiv.org/abs/2511.10841)

**Paper:**
[FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification](https://arxiv.org/abs/2511.10841)
(arXiv:2511.10841, Nov 2025) ·
DOI: [10.48550/arXiv.2511.10841](https://doi.org/10.48550/arXiv.2511.10841)

**Authors:** YongKyung Oh, Dong-Young Lim, Sungil Kim

**TL;DR:** FlowPath is a learnable control path for neural controlled
differential equations (Neural CDEs). Instead of a fixed interpolation, it
uses invertible neural flows to learn a continuous, data-driven manifold for
robust irregular time series classification, even under heavy missingness.

**Keywords:** neural controlled differential equations · invertible neural
flows · irregular time series classification · learnable control path

---

## Overview

Modeling continuous-time dynamics from sparse and irregularly-sampled time
series remains a fundamental challenge. Neural controlled differential
equations offer a principled framework, but their performance is highly
sensitive to how discrete observations are lifted into continuous control
paths. Most existing models rely on fixed interpolation schemes that impose
simplistic geometric assumptions and often distort the data manifold,
especially under high missingness.

**FlowPath** is a learnable path construction method built on invertible
neural flows. Instead of linking observations through a predefined
interpolant, FlowPath learns a continuous, data-adaptive manifold subject to
invertibility constraints that promote information-preserving and stable
transformations. This inductive bias separates FlowPath from prior
unconstrained learnable path models.

Empirical evaluations on benchmark datasets and a real-world case study show
that FlowPath improves classification accuracy over fixed interpolants and
non-invertible architectures, highlighting the importance of modeling both the
dynamics along the path and the geometry of the path itself.

## Method

FlowPath extends Neural Differential Equation frameworks through:

* **Invertible path construction:** data-adaptive path via invertible flows.
* **Geometry-aware control paths:** continuous paths that better reflect the
  latent manifold than fixed interpolants.
* **NCDE compatibility:** the learned path plugs into NCDE backbones.

## Code architecture

The repository contains two primary components:

* `torch-ists/`: utilities and differential-equation models for irregular TS.
* `PAMAP2/`: human activity recognition and the sensor-drop experiment.

## Citation

If you use this software or method, please cite the arXiv preprint:

```bibtex
@misc{oh2025flowpath,
  title         = {FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification},
  author        = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
  year          = {2025},
  eprint        = {2511.10841},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2511.10841},
  url           = {https://arxiv.org/abs/2511.10841}
}
```

## References

> [1] Oh, Y., Lim, D.-Y., & Kim, S. (2025). DualDynamics: Synergizing
> Implicit and Explicit Methods for Robust Irregular Time Series Analysis.
> AAAI-25 (pp. 19730-19739). AAAI Press.
> https://doi.org/10.1609/AAAI.V39I18.34173
>
> [2] Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022).
> Graph-Guided Network for Irregularly Sampled Multivariate Time Series. ICLR.

## License

Released under the [MIT License](LICENSE).
