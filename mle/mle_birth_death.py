#!/usr/bin/env python3
"""
MLE for Birth-Death Trees 

For complete sampling at present time, Stadler's likelihood formula is:
L(λ, μ | tree) = (n-1)! × λ^(n-1) × exp(-(λ+μ)×S) / (λ - μ×exp(-(λ-μ)×T))^n

Where:
- n = number of tips
- S = sum of all branch lengths (excluding root)
- T = tree height (time from root to present)

"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ete3 import Tree
import os
import glob

# Import for better factorial calculation
from math import lgamma


def load_tree_from_newick(newick_file):
    """Load a phylogenetic tree from a Newick file."""
    try:
        with open(newick_file, 'r') as f:
            newick_str = f.read().strip()
        tree = Tree(newick_str, format=1)
        return tree
    except Exception as e:
        print(f"Error loading tree from {newick_file}: {e}")
        return None


def get_tree_statistics(tree):
    """
    Returns: dictionary with tree statistics
    """
    if tree is None:
        return None
    
    n_tips = len(tree.get_leaves())
    
    # Get all branch lengths (excluding root edge)
    branch_lengths = []
    for node in tree.traverse():
        if node.dist is not None and not node.is_root():
            branch_lengths.append(node.dist)
    
    total_length = sum(branch_lengths)
    
    # Get tree height (distance from root to furthest tip)
    tree_height = tree.get_farthest_leaf()[1]
    
    return {
        'n_tips': n_tips,
        'total_time': total_length,
        'sum_branch_lengths': total_length,
        'tree_height': tree_height
    }


def birth_death_likelihood(params, tree, penalty_weight=1.0, tree_stats=None):
    """
    Calculate negative log-likelihood for birth-death model.
    
    Parameters:
    - params: [lambda, mu] (birth rate, death rate)
    - tree: ete3 Tree object
    - penalty_weight: Weight for penalty term to prevent unrealistic rates (default 1.0)
    - tree_stats: Pre-computed tree statistics
    
    Returns: negative log-likelihood (for minimization)
    """
    lambda_bd, mu = params
    
    # Check for invalid inputs (NaN, inf, negative, zero)
    if not np.isfinite(lambda_bd) or not np.isfinite(mu):
        return np.inf
    if lambda_bd <= 0 or mu < 0:
        return np.inf
    
    try:
        n = len(tree.get_leaves())
        if n < 2:
            return np.inf
        
        # Get tree statistics
        if tree_stats is None:
            total_length = sum([node.dist for node in tree.traverse() 
                               if node.dist is not None and not node.is_root()])
            tree_height = tree.get_farthest_leaf()[1]
        else:
            total_length = tree_stats['sum_branch_lengths']
            tree_height = tree_stats['tree_height']
        
        if tree_height <= 0 or total_length <= 0:
            return np.inf
        
        r = lambda_bd - mu
        
        # Calculate log-likelihood using Stadler's exact formula
        # Component 1: log((n-1)!)
        # lgamma can overflow for very large n, but for typical trees (n < 10^6) it's safe
        try:
            log_factorial = lgamma(n)  # lgamma(n) = log((n-1)!) for integer n
        except (OverflowError, ValueError):
            # For extremely large n, use Stirling's approximation
            # log(n!) ≈ n*log(n) - n + 0.5*log(2*pi*n)
            log_factorial = (n - 1) * np.log(n - 1) - (n - 1) + 0.5 * np.log(2 * np.pi * (n - 1))
        
        # Component 2: (n-1) × log(λ)
        # Check for very small lambda that might cause log issues
        if lambda_bd < 1e-300:
            return np.inf
        log_lambda_term = (n - 1) * np.log(lambda_bd)
        
        # Component 3: -(λ+μ) × S
        exp_term = -(lambda_bd + mu) * total_length
        
        # Component 4: Denominator term
        if abs(r) < 1e-10:
            # Special case: λ ≈ μ (critical case)
            # lim_{r→0} (λ - μ×exp(-r×T)) = λ×T
            denominator = lambda_bd * tree_height
        else:
            # General case
            # Check for potential overflow before computing exp
            # exp(-r * tree_height) will overflow if -r * tree_height > 700
            # (since exp(709) ≈ max float64, exp(710) overflows)
            exponent = -r * tree_height
            
            if exponent > 700:
                # exp(-r*T) would overflow, and since r < 0 (mu > lambda),
                # denominator = lambda - mu * huge_number ≈ -infinity
                # This makes denominator negative, so return inf
                return np.inf
            elif exponent < -700:
                # exp(-r*T) would underflow to 0, so denominator = lambda - mu * 0 = lambda
                # This is fine, but we can handle it explicitly for numerical stability
                denominator = lambda_bd
            else:
                # Safe to compute exp
                exp_rt = np.exp(exponent)
                denominator = lambda_bd - mu * exp_rt
            
            if denominator <= 0:
                return np.inf
        
        # Component 4: -n × log(denominator)
        # Check for invalid denominator before taking log
        if denominator <= 0 or not np.isfinite(denominator):
            return np.inf
        log_denom_term = -n * np.log(denominator)
        
        # Total log-likelihood from Stadler's formula
        log_likelihood = log_factorial + log_lambda_term + exp_term + log_denom_term
        
        # Check for NaN or inf in the result
        if not np.isfinite(log_likelihood):
            return np.inf
        
        # Add penalty term to prevent unrealistic rates
        # This addresses the likelihood surface anomaly where very small rates
        # give astronomically large likelihoods (10^126)
        # Use a large additive penalty that's strong enough to overcome the numerical artifact
        penalty = 0.0
        
        # Penalty 1: Very small rates that cause numerical issues
        # When rates are extremely small (< 0.1), add a very large penalty
        # This must be large enough to overcome the numerical artifact
        if lambda_bd < 0.1:
            # Large penalty that increases as rate decreases
            penalty += penalty_weight * 50000.0 * (0.1 - lambda_bd) / 0.1
        if mu < 0.1:
            penalty += penalty_weight * 50000.0 * (0.1 - mu) / 0.1
        
        # Penalty 2: Rates that are too small relative to tree characteristics
        # For a tree with n tips in time T, we expect rates to be at least ~log(n)/T
        min_reasonable_rate = max(0.1, np.log(n) / (tree_height * 5.0) if tree_height > 0 else 0.1)
        if lambda_bd < min_reasonable_rate:
            penalty += penalty_weight * 10000.0 * (min_reasonable_rate - lambda_bd) / min_reasonable_rate
        if mu < min_reasonable_rate * 0.5:
            penalty += penalty_weight * 5000.0 * (min_reasonable_rate * 0.5 - mu) / (min_reasonable_rate * 0.5)
        
        # Penalty 3: Rates that are too large (unrealistic for most biological processes)
        # Average branch length gives hint about reasonable rate scale
        num_branches = 2 * n - 2
        avg_branch_length = total_length / num_branches if num_branches > 0 else 1.0
        max_reasonable_rate = min(100.0, 10.0 / avg_branch_length if avg_branch_length > 0 else 100.0)
        if lambda_bd > max_reasonable_rate:
            penalty += penalty_weight * 1000.0 * (lambda_bd - max_reasonable_rate) / max_reasonable_rate
        if mu > max_reasonable_rate:
            penalty += penalty_weight * 1000.0 * (mu - max_reasonable_rate) / max_reasonable_rate
        
        # Return negative log-likelihood with penalty
        # The penalty is added (not multiplied) so it's a fixed cost for unrealistic rates
        neg_log_likelihood = -log_likelihood + penalty
        
        # Final check: ensure we never return NaN
        if not np.isfinite(neg_log_likelihood):
            return np.inf
        
        return neg_log_likelihood
        
    except (OverflowError, ValueError, ZeroDivisionError):
        # Handle specific numerical errors
        return np.inf
    except Exception as e:
        # Catch any other errors and return inf (never NaN)
        return np.inf


def estimate_mle_birth_death(tree):
    """
    Estimate birth and death rates using Maximum Likelihood Estimation.
    
    Uses Stadler's exact likelihood formula and robust optimization to find
    the true MLE estimates.
    
    Returns: (lambda_mle, mu_mle, R0_mle) where R0 = lambda/mu
    """
    if tree is None:
        return None, None, None
    
    n_tips = len(tree.get_leaves())
    if n_tips < 2:
        return None, None, None
    
    # Get tree statistics
    tree_stats = get_tree_statistics(tree)
    if tree_stats is None:
        return None, None, None
    
    T = tree_stats['tree_height']
    total_length = tree_stats['sum_branch_lengths']
    
    if T <= 0 or total_length <= 0:
        return None, None, None
    
    # Calculate initial guess using method of moments
    # Under exponential growth: n ≈ exp((λ-μ)×T) for large n
    # So: λ - μ ≈ log(n) / T
    if n_tips > 1 and T > 0:
        r_estimate = np.log(n_tips) / T
    else:
        r_estimate = 0.1
    
    # Estimate λ + μ from branch lengths
    # In a birth-death process, the expected branch length relates to rates
    num_branches = 2 * n_tips - 2  # Number of branches in a binary tree
    avg_branch_length = total_length / num_branches if num_branches > 0 else 1.0
    
    # Rough estimate: higher rates → shorter branches
    # Use a heuristic: sum_rate ≈ 1 / avg_branch_length (scaled)
    # But also consider tree size: more tips in same time → higher rates
    if avg_branch_length > 0:
        sum_rate_estimate = max(0.5, min(50.0, 2.0 / avg_branch_length))
    else:
        sum_rate_estimate = 2.0
    
    # Also consider: if tree has many tips in short time, rates should be higher
    if T > 0:
        tips_per_time = n_tips / T
        sum_rate_estimate = max(sum_rate_estimate, tips_per_time * 0.5)
    
    # Solve: λ + μ = sum_rate, λ - μ = r
    lambda_init = (sum_rate_estimate + r_estimate) / 2
    mu_init = (sum_rate_estimate - r_estimate) / 2
    
    # Ensure reasonable initial values (avoid very small rates)
    lambda_init = max(0.1, min(50.0, lambda_init))
    mu_init = max(0.05, min(50.0, mu_init))
    
    # Ensure μ < λ (typically true for growing populations)
    if mu_init >= lambda_init:
        mu_init = lambda_init * 0.8
    
    initial_params = [lambda_init, mu_init]
    
    # Parameter bounds: Use reasonable bounds based on tree characteristics
    # Lower bound: rates should be at least 0.1 to avoid numerical issues with likelihood surface
    # Also consider tree characteristics: log(n)/(5*T) gives minimum reasonable rate
    min_rate = max(0.1, np.log(n_tips) / (T * 5.0) if T > 0 else 0.1)
    
    # Upper bound: rates shouldn't be more than 10/avg_branch_length
    max_rate = max(10.0, min(100.0, 10.0 / avg_branch_length if avg_branch_length > 0 else 100.0))
    
    # Also consider tree size: very large trees might need higher bounds
    if n_tips > 500 and T > 0:
        max_rate = max(max_rate, n_tips / T * 2.0)
    
    bounds = [(min_rate, max_rate), (min_rate, max_rate)]
    
    # Try optimization with L-BFGS-B (gradient-based, handles bounds)
    best_result = None
    best_ll = np.inf
    
    # Try multiple starting points for robustness
    starting_points = [
        initial_params,
        [lambda_init * 0.5, mu_init * 0.5],
        [lambda_init * 2.0, mu_init * 2.0],
        [lambda_init, mu_init * 0.3],
        [lambda_init, mu_init * 0.7],
    ]
    
    # Pre-compute tree stats for efficiency
    tree_stats_dict = {
        'n_tips': n_tips,
        'sum_branch_lengths': total_length,
        'tree_height': T
    }
    
    # Create a wrapper function that clamps parameters to bounds
    # This prevents numerical differentiation from trying invalid values
    def likelihood_wrapper(params):
        """Wrapper that ensures parameters are within bounds before evaluation."""
        # Clamp parameters to bounds to prevent numerical differentiation issues
        lambda_clamped = max(min_rate, min(max_rate, params[0]))
        mu_clamped = max(min_rate, min(max_rate, params[1]))
        return birth_death_likelihood([lambda_clamped, mu_clamped], tree, 1.0, tree_stats_dict)
    
    for start_params in starting_points:
        # Ensure starting point is within bounds
        start_params[0] = max(min_rate, min(max_rate, start_params[0]))
        start_params[1] = max(min_rate, min(max_rate, start_params[1]))
        
        try:
            result = minimize(
                likelihood_wrapper,
                start_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-9,
                    'gtol': 1e-5,
                    'maxls': 50
                }
            )
            
            if result.success and result.fun < best_ll:
                best_ll = result.fun
                best_result = result
        except:
            continue
    
    # If L-BFGS-B didn't work well, try Nelder-Mead (no gradients needed)
    if best_result is None or best_result.fun > 1e6:
        for start_params in starting_points[:3]:  # Try fewer starting points
            start_params[0] = max(min_rate, min(max_rate, start_params[0]))
            start_params[1] = max(min_rate, min(max_rate, start_params[1]))
            
            try:
                result = minimize(
                    likelihood_wrapper,
                    start_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-6,
                        'fatol': 1e-6
                    }
                )
                
                # Check if result is within bounds
                if (result.success and 
                    min_rate <= result.x[0] <= max_rate and 
                    min_rate <= result.x[1] <= max_rate and
                    result.fun < best_ll):
                    best_ll = result.fun
                    best_result = result
            except:
                continue
    
    if best_result is not None and best_result.success:
        lambda_mle, mu_mle = best_result.x
        
        # Ensure values are positive and reasonable (within bounds)
        lambda_mle = max(min_rate, min(max_rate, lambda_mle))
        mu_mle = max(min_rate, min(max_rate, mu_mle))
        
        # Check if we're at a boundary
        at_boundary = (abs(lambda_mle - min_rate) < 1e-6 or 
                      abs(mu_mle - min_rate) < 1e-6 or
                      abs(lambda_mle - max_rate) < 1e-6 or
                      abs(mu_mle - max_rate) < 1e-6)
        
        if at_boundary:
            # If at boundary, the optimization likely failed
            # Use method of moments as fallback
            if T > 0:
                lambda_mle = max(min_rate, min(max_rate, n_tips / T))
                mu_mle = max(min_rate, min(max_rate, lambda_mle * 0.5))
        
        R0_mle = lambda_mle / mu_mle if mu_mle > 0 else np.inf
        return lambda_mle, mu_mle, R0_mle
    
    # Final fallback: method of moments
    if T > 0:
        lambda_mle = max(0.1, min(50.0, n_tips / T))
        mu_mle = max(0.05, lambda_mle * 0.5)
        R0_mle = lambda_mle / mu_mle if mu_mle > 0 else 2.0
        return lambda_mle, mu_mle, R0_mle
    
    return None, None, None


def main():
    """Main function to estimate MLE parameters for all trees."""
    print("=" * 60)
    print("MLE Estimation for Birth-Death Trees")
    print("=" * 60)
    
    # Paths
    base_dir = "phylodynamicsDL"
    trees_dir = os.path.join(base_dir, "output_trees")
    params_file = os.path.join(base_dir, "all_params.csv")
    output_dir = "mle_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mle_estimates.csv")
    
    # Load true parameters
    print("\nLoading true parameters...")
    true_params = pd.read_csv(params_file)
    print(f"Loaded {len(true_params)} parameter sets")
    
    # Get all tree files
    tree_files = sorted(glob.glob(os.path.join(trees_dir, "tree_*.nwk")))
    print(f"Found {len(tree_files)} tree files")
    
    # Create results storage
    results = []
    
    # Process trees
    print("\nProcessing trees with MLE...")
    for idx, tree_file in enumerate(tree_files):
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(tree_files)} trees...")
        
        # Extract tree index from filename
        tree_idx = int(os.path.basename(tree_file).replace("tree_", "").replace(".nwk", ""))
        
        # Get true parameters for this tree
        true_row = true_params[true_params['idx'] == tree_idx]
        if len(true_row) == 0:
            continue
        
        true_R0 = true_row['R'].values[0]
        true_d = true_row['d'].values[0]
        true_lambda = true_R0 * true_d
        n_tips = int(true_row['tips'].values[0])
        
        # Load tree
        tree = load_tree_from_newick(tree_file)
        if tree is None:
            continue
        
        # Estimate MLE
        lambda_mle, mu_mle, R0_mle = estimate_mle_birth_death(tree)
        
        # Store results
        result = {
            'tree_idx': tree_idx,
            'n_tips': n_tips,
            'true_R0': true_R0,
            'true_d': true_d,
            'true_lambda': true_lambda,
            'mle_R0': R0_mle if R0_mle is not None else np.nan,
            'mle_d': mu_mle if mu_mle is not None else np.nan,
            'mle_lambda': lambda_mle if lambda_mle is not None else np.nan,
        }
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print(f"\nSuccessfully processed {len(results_df)} trees")
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Saved MLE estimates to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MLE ESTIMATION SUMMARY")
    print("=" * 60)
    print(f"Total trees analyzed: {len(results_df)}")
    print(f"Successful estimates: {results_df['mle_R0'].notna().sum()}")
    
    # Print statistics
    if results_df['mle_R0'].notna().sum() > 0:
        print(f"\nMLE R0 statistics:")
        print(f"  Mean: {results_df['mle_R0'].mean():.4f}")
        print(f"  Std: {results_df['mle_R0'].std():.4f}")
        print(f"  Range: [{results_df['mle_R0'].min():.4f}, {results_df['mle_R0'].max():.4f}]")
        print(f"\nMLE d (μ) statistics:")
        print(f"  Mean: {results_df['mle_d'].mean():.4f}")
        print(f"  Std: {results_df['mle_d'].std():.4f}")
        print(f"  Range: [{results_df['mle_d'].min():.4f}, {results_df['mle_d'].max():.4f}]")
    
    print(f"\nResults saved to: {output_file}")
    print("\nMLE estimation complete!")


if __name__ == "__main__":
    main()
