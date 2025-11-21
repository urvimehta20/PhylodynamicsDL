# Statistical Efficiency of Phylodeep Models vs. Maximum Likelihood Estimation by Tree Size

We want to estimate the statistical efficiency of phylodeep models. We start with the simple case of the birth death
model and compare it with likelihood.

## Setup

1) Generate birth-death trees under static $\theta = (\lambda, \psi)$. Generate trees of different tip sizes, $n \in \{10, 20, 50, 100, 200, 500\}$.
2) At each tip size, find maximum likelihood estimate of the parameters. Calculate the bias and variance of the estimates.
3) Estimate the parameters from on each tree using the phylodeep models. Calculate the bias and variance of the estimates.
4) Estimate the statistical efficiency of the phylodeep models vs. the maximum likelihood estimates as a ratio of these two variance estimates and its relationship to tree size.
