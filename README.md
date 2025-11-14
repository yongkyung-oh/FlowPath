# **FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification**

Modeling continuous-time dynamics from sparse and irregularly-sampled time series remains a fundamental challenge. Neural controlled differential equations offer a principled framework, but their performance is highly sensitive to how discrete observations are lifted into continuous control paths. Most existing models rely on fixed interpolation schemes that impose simplistic geometric assumptions and often distort the data manifold, especially under high missingness.

**FlowPath** is a learnable path construction method built on invertible neural flows. Instead of linking observations through a predefined interpolant, FlowPath learns a continuous, data-adaptive manifold subject to invertibility constraints that promote information-preserving and stable transformations. This inductive bias separates FlowPath from prior unconstrained learnable path models.

Empirical evaluations on benchmark datasets and a real-world case study show that FlowPath improves classification accuracy over fixed interpolants and non-invertible architectures. These results highlight the importance of modeling both the dynamics along the path and the geometry of the path itself.

---

## **Code architecture**

The repository contains two primary components:

* `torch-ists`: Core utilities for irregular time series and differential equation models.
* `PAMAP2`: Implementation of human activity recognition and sensor drop experiment.

---

> [1] Oh, Y., Lim, D.-Y., & Kim, S. (2025). DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis. In T. Walsh, J. Shah, & Z. Kolter (Eds.), AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25—March 4, 2025, Philadelphia, PA, USA (pp. 19730–19739). AAAI Press. https://doi.org/10.1609/AAAI.V39I18.34173

> [2] Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022) Graph-Guided Network for Irregularly Sampled Multivariate Time Series. In International Conference on Learning Representations.

---

## **Core implementation details**

FlowPath extends Neural Differential Equation frameworks through:

* **Invertible Path Construction**: Learns a data-adaptive path using invertible neural flows.
* **Geometry-aware Control Paths**: Produces continuous paths that better reflect the latent manifold than fixed interpolants.
* **Compatibility with NCDE Models**: The learned path can be integrated into NCDE backbones for downstream classification.

---

## **Citation**

```bibtex

```
