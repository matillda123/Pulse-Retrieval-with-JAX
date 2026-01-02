# Pulse-Retrieval with JAX

Uses Jax to implement various algorithms for pulse-retrieval in ultrafast optics.  

## Installation  
To install (create a new environment and) run:
```
pip install git+https://github.com/matillda123/pulse-retrieval-with-jax.git
```

## Description  
Available methods are FROG, Chirp-Scans (material dispersion or pulse-shaping), Time-Domain-Ptychography, Two-Dimensional Spectral-Shearing Interferometry (2D-SI) and VAMPIRE.  
For each method SHG, THG, PG/TG and SD as well as cross-correlation and interferometric retrievals are implemented (with some exceptions).  

On top of the naive implementation of each algorithm, some allow the usage of standard nonlinear optimization approaches like nonlinear conjugate gradients or LBFGS. In other cases the pulses may be parametrized via analytic functions instead of a discrete grid.



The available algorithms are: 

| Algorithm | Citation |
|---|---|
| Generalized Projection | [K. W. DeLong et al., Opt. Lett. 19, 2152-2154 (1994)](https://doi.org/10.1364/OL.19.002152)  |
| Ptychographic Iterative Engine | [A. Maiden et al., Optica 4, 736-745 (2017)](https://doi.org/10.1364/OPTICA.4.000736) |
| Common Pulse Retrieval Algorithm | [N. C. Geib, Optica 6, 495-505 (2019)](https://doi.org/10.1364/OPTICA.6.000495) and [pypret](https://github.com/ncgeib/pypret) |
| Differential Evolution | [J. Qiang and C. Mitchell (2014)](https://www.osti.gov/biblio/1163659) and [I. Grigorenko and M.E. Garcia, Physica A 284 131–139 (2000)](https://doi.org/10.1016/S0378-4371(00)00218-1)|
| Linesearch-Frog-Algorithm (for all methods) | [C. O. Krook and V. Pasiskevicius, Opt. Express 33, 33258-33269 (2025)](https://doi.org/10.1364/OE.569606)  |
| AD-Solvers ([optax](https://github.com/google-deepmind/optax) and [optimistix](https://github.com/patrick-kidger/optimistix)) | [ DeepMind et al., *The DeepMind JAX Ecosystem* (2020)](https://github.com/google-deepmind/optax) and<br>[J. Rader, T. Lyons and P. Kidger, *Optimistix: modular optimisation in JAX and Equinox*, arXiv:2402.09983 (2024)](https://arxiv.org/abs/2402.09983) |
| Evolutionary solvers ([evosax](https://github.com/RobertTLange/evosax)) | [R. T. Lange, arXiv 2212.04180 (2022)](https://arxiv.org/abs/2212.04180) |


For some methods, specific additional algorithms are implemented:

| Method | Algorithms | Citation |
|---|---|---|
| FROG | Vanilla | [R. Trebino, 10.1007/978-1-4615-1181-6 (2000)](https://link.springer.com/book/10.1007/978-1-4615-1181-6) |
|  | LSGPA | [J. Gagnon et al., Appl. Phys. B 92, 25-32, 10.1007/s00340-008-3063-x (2008)](https://doi.org/10.1007/s00340-008-3063-x) |
|  |  CPCGPA | [D. J. Kane and A. B. Vakhtin, Prog. Quantum Electron. 81 (100364),  (2022)](https://doi.org/10.1016/j.pquantelec.2021.100364) |
|  |  |  |
| 2D-SI | Direct-Reconstruction | [J. R. Birge et al., Opt. Lett. 31, 2063-2065, 10.1364/OL.31.002063 (2006)](https://doi.org/10.1364/OL.31.002063) |  





If you end up using this code for a publication, please use the citation below as well as the appropriate citations for the algorithm used.

```bibtex
@software{pulseretrievalwithjax,
  title = {Pulse {R}etrieval with {JAX}},
  author = {T. J. Stehling},
  url = {https://github.com/matillda123/Pulse-Retrieval-with-Jax/}
  year = {2025}
}
```






