# SSJ Julia notebook

This repository provides notebooks demonstrating how one can solve a simple Heterogeneous Agents business cycle model using the Sequence Space Jacobian method by [Auclert et al. (2021)](https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA17434) in Julia.

How does this notebook differ from other resources?
* Everything is implemented from scratch, not relying on a blackbox package
* The Sequence Space Jacobians (SSJs) are obtained using Automatic Differentiation

Additionally, I added a notebook that demonstrates the global Repeated Transition Method (RTM) by [Lee (2025)](https://hanbaeklee.github.io/Webpage/Lee_AggRepTrans_2025.pdf).
(The author argues it to be a Sequence Space method but it doesn't use SSJs)

To view HTML-versions of the notebook, click the links below [here](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html).

1. [First SSJ notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html): Krusell and Smith (1998)-type economy from Auclert et al. (2021)

2. [Second SSJ notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook_2.html): Simple HANK model from Auclert et al. (2021), different methods for getting the GE SSJs

3. [RTM notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/ks_rtm_notebook.html): Global solution of a Krusell and Smith (1998)-type economy using the RTM.


