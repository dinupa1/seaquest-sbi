# Neural Posterior Estimation

## Abstract

Bayesian inference is a robust framework for parameter estimation and uncertainty quantification in complex physical processes. However, traditional methods often require approximating the likelihood function, which can be mathematically intractable in real-world applications. To address this challenge, classifiers can be utilized to approximate the likelihood-to-evidence ratio, enabling efficient evaluation of posterior distributions. This alternative approach simplifies the inference process while maintaining accuracy, making it particularly valuable in scenarios with complex or unknown likelihoods, such as those encountered in physics and other scientific domains.

## Bayesian Inference

For a given observation $x$ and model (or theory) parameters $\vartheta$, the posterior $p(\vartheta | x)$ can be defined using Bayes’ theorem,

$$
p(\vartheta | x) = \dfrac{p(x | \vartheta)}{p(x)} p(\vartheta)
$$


Let $s(x, \vartheta)$ be a binary classifier that distinguishes between samples drawn from the `joint` probability distribution $p(x, \vartheta)$ and the `marginal` probability distribution $p(x)p(\vartheta)$. Then ratio estimator can be defined as,

$$
\hat{r}(x, \vartheta) = \dfrac{s(x, \vartheta)}{1 - s(x, \vartheta)}
$$


### Using this package

These models are optimized for [Fermilab E-906/SeaQuest](https://www.phy.anl.gov/mep/drell-yan/) Monte Carlo (MC) simulations. It is recommended to use the [Fermilab Elastic Analysis Facility](https://eafjupyter.readthedocs.io/en/latest/) for training and testing the package. Use the following commands to set up the environment variables and compile the dynamic libraries.


```
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

root -b -q setup.cc
```

To run the forward simulation,

```
python simulation.py
```

To run the Bayesian inference,

```
python inference.py
```

Plot the results,

```
python plots.py
```

Note that E906 MC files are not included in this repository. Please use the [Discord channel](https://discord.gg/ycs3ary4WV) to request MC or report any issues.


## References

- A Guide to Constraining Effective Field Theories with Machine Learning [arXiv:1805.00020 [hep-ph]](https://arxiv.org/abs/1805.00020)
- Neural Networks for Full Phase-space Reweighting and Parameter Tuning [arXiv:1907.08209 [hep-ph]](https://arxiv.org/abs/1907.08209)
- Learning Likelihood Ratios with Neural Network Classifiers [arXiv:2305.10500 [hep-ph]](https://arxiv.org/abs/2305.10500)
- Constraining the Higgs Potential with Neural Simulation-based Inference for Di-Higgs Production [arXiv:2405.15847 [hep-ph]](https://arxiv.org/abs/2405.15847)
- Likelihood-free MCMC with Amortized Approximate Ratio Estimators [arXiv:1903.04057 [stat.ML]](https://arxiv.org/abs/1903.04057)
- Statistical guarantees for stochastic Metropolis-Hastings [arXiv:2310.09335 [stat.ML]](https://arxiv.org/abs/2310.09335)

