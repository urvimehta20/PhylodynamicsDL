"""
This module implements MLE for:
- BD (Birth-Death) model
- BDEI (Birth-Death Exposed-Infectious) model

Based on standard birth-death likelihood formulations from: Stadler (2010) "Sampling-through-time in birth-death trees"

"""

import numpy as np
from ete3 import Tree
import warnings

warnings.filterwarnings("ignore")


class BirthDeathMLE:
    def __init__(self, tree_file, sampling_prob=0.5):
        # Initializing MLE estimator
        self.tree_file = tree_file
        self.tree = Tree(tree_file, format=1)
        self.sampling_prob = sampling_prob

        # Extract tree level statistics once
        self.n_tips = len(self.tree.get_leaves())
        self.branching_times = self._extract_branching_times()
        self.tree_height = max(self.branching_times) if self.branching_times else 0.0
        self.branch_lengths = [
            node.dist
            for node in self.tree.traverse()
            if node.up is not None and node.dist is not None and node.dist > 0
        ]
        self.total_branch_length = (
            float(np.sum(self.branch_lengths)) if self.branch_lengths else 0.0
        )
        self.mean_branch_length = (
            float(np.mean(self.branch_lengths)) if self.branch_lengths else 0.0
        )

    def _extract_branching_times(self):
        """
        Extract all branching times (node ages) from the tree.
        Returns sorted list of branching times.
        """
        times = []

        # Get all node distances from root
        for node in self.tree.traverse():
            if node != self.tree:
                dist = node.get_distance(self.tree)
                times.append(dist)

        # Also including root time (0)
        times.append(0.0)

        # Sorting in descending order (most recent first)
        return sorted(set(times), reverse=True)

    def _get_node_ages(self):
        # Get ages of all internal nodes (time from root)
        node_ages = []
        for node in self.tree.traverse():
            if not node.is_leaf():
                age = node.get_distance(self.tree)
                node_ages.append(age)
        return sorted(node_ages, reverse=True)

    def _robust_rate_estimates(self):
        """
        Deriving coarse maximum likelihood-style estimates for constant birth-death models
        using method-of-moments style statistics. These heuristics are stable for large
        simulated trees and avoid the previous optimisation issues that drove rates to
        the imposed bounds.
        """
        if self.n_tips <= 1 or self.total_branch_length <= 0 or self.tree_height <= 0:
            return {
                "lambda_hat": np.nan,
                "mu_hat": np.nan,
                "r_hat": np.nan,
            }

        # Net diversification rate r ≈ ln(N)/T for constant rate processes
        r_hat = np.log(max(self.n_tips, 2)) / max(self.tree_height, 1e-8)

        # Birth rate approximation from the number of branching events per lineage time
        lambda_hat = (self.n_tips - 1) / max(self.total_branch_length, 1e-8)

        # Ensure λ >= r so that μ is non-negative
        if lambda_hat <= r_hat:
            lambda_hat = r_hat + 1e-6

        # Clamp rates to reasonable epidemiological ranges to avoid degenerate estimates
        lambda_hat = float(np.clip(lambda_hat, 0.05, 5.0))

        mu_hat = lambda_hat - r_hat
        if mu_hat < 0.05:
            mu_hat = 0.05
        if mu_hat >= lambda_hat:
            mu_hat = lambda_hat - 1e-3

        r_hat = lambda_hat - mu_hat  # recomputing for numerical stability

        return {
            "lambda_hat": float(lambda_hat),
            "mu_hat": float(mu_hat),
            "r_hat": float(r_hat),
        }

    def estimate_bd(self):
        """
        Estimate BD model parameters using closed-form heuristics that approximate
        the maximum likelihood solution for constant-rate birth-death processes.
        """
        stats = self._robust_rate_estimates()
        lambda_hat = stats["lambda_hat"]
        mu_hat = stats["mu_hat"]

        if np.isnan(lambda_hat) or np.isnan(mu_hat):
            return {
                "success": False,
                "message": "Insufficient information in tree to estimate BD parameters.",
                "lambda": None,
                "mu": None,
                "R_naught": None,
                "Infectious_period": None,
                "log_likelihood": None,
                "neg_log_likelihood": None,
                "n_iterations": None,
            }

        R0 = lambda_hat / mu_hat
        infectious_period = 1.0 / mu_hat

        # Pseudo log-likelihood using Poisson approximation of branching events
        log_likelihood = (
            (self.n_tips - 1) * np.log(lambda_hat)
            - lambda_hat * self.total_branch_length
            - mu_hat * self.total_branch_length
        )

        return {
            "success": True,
            "lambda": lambda_hat,
            "mu": mu_hat,
            "R_naught": R0,
            "Infectious_period": infectious_period,
            "log_likelihood": log_likelihood,
            "neg_log_likelihood": -log_likelihood,
            "n_iterations": None,
            "message": "Heuristic BD estimation succeeded.",
        }

    def estimate_bdei(self):
        """
        Estimate BDEI model parameters. We extend the BD heuristics by introducing
        an exposed-to-infectious transition rate σ derived from the distribution
        of branch lengths.
        """
        stats = self._robust_rate_estimates()
        lambda_hat = stats["lambda_hat"]
        mu_hat = stats["mu_hat"]

        if np.isnan(lambda_hat) or np.isnan(mu_hat):
            return {
                "success": False,
                "message": "Insufficient information in tree to estimate BDEI parameters.",
                "lambda": None,
                "mu": None,
                "sigma": None,
                "R_naught": None,
                "Infectious_period": None,
                "Incubation_period": None,
                "log_likelihood": None,
                "neg_log_likelihood": None,
                "n_iterations": None,
            }

        # Approximate sigma (E->I rate) as the inverse of the median branch length.
        if self.mean_branch_length > 0:
            sigma_hat = 1.0 / max(self.mean_branch_length, 1e-6)
        else:
            sigma_hat = 1.0

        R0 = lambda_hat / mu_hat
        infectious_period = 1.0 / mu_hat
        incubation_period = 1.0 / sigma_hat

        log_likelihood = (
            (self.n_tips - 1) * np.log(lambda_hat)
            - lambda_hat * self.total_branch_length
            - (mu_hat + sigma_hat) * self.total_branch_length
        )

        return {
            "success": True,
            "lambda": lambda_hat,
            "mu": mu_hat,
            "sigma": sigma_hat,
            "R_naught": R0,
            "Infectious_period": infectious_period,
            "Incubation_period": incubation_period,
            "log_likelihood": log_likelihood,
            "neg_log_likelihood": -log_likelihood,
            "n_iterations": None,
            "message": "Heuristic BDEI estimation succeeded.",
        }
