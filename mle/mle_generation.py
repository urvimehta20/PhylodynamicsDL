"""
MLE for the generated trees:

- Loads trees from phylodynamicsDL/output_trees/
- Groups trees by the tip size (n=50, 100, 200, 500)
- Performs MLE for BD model on each tree
- Aggregates and reports results for each tip size
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import mle_birth_death
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle_birth_death import BirthDeathMLE

print("=" * 80)
print("Maximum Likelihood Estimation for PhyloDynamicsDL Trees")
print("=" * 80)

# Get the directory where this script is located and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Analyzing all trees with tip size >= 50
TARGET_TIP_SIZES = None  # Set to None to analyze all trees >= 50 tips
MIN_TIP_SIZE = 50  # Minimum tip size for analysis

TREES_DIR = os.path.join(PROJECT_ROOT, "phylodynamicsDL", "output_trees")
PARAMS_FILE = os.path.join(PROJECT_ROOT, "phylodynamicsDL", "all_params.csv")
SAMPLING_PROBA = 0.5  # Sampling probability
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "mle_results")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the parameters file to get tree indices and tip sizes
print("\n[STEP 1] Loading tree metadata...")
print("-" * 80)
df_params = pd.read_csv(PARAMS_FILE)
print(f"Loaded {len(df_params)} tree records")
print(f"Tip size range: {df_params['tips'].min()} - {df_params['tips'].max()}")

# Determine which trees to analyze
if TARGET_TIP_SIZES is None:
    # Analyze all trees with tip size >= MIN_TIP_SIZE
    print(f"\nAnalyzing ALL trees with tip size >= {MIN_TIP_SIZE}")
    df_filtered = df_params[df_params['tips'] >= MIN_TIP_SIZE]
    print(f"Found {len(df_filtered)} trees with tip size >= {MIN_TIP_SIZE}")
    
    # Group by tip size for processing
    unique_sizes = sorted(df_filtered['tips'].unique())
    print(f"Tip sizes to process: {unique_sizes[:10]}{'...' if len(unique_sizes) > 10 else ''} ({len(unique_sizes)} unique sizes)")
    
    # Create a list of all tip sizes to process
    TARGET_TIP_SIZES = unique_sizes
else:
    # Original behavior: analyze specific tip sizes
    print("\nChecking availability of target tip sizes:")
    available_sizes = {}
    for size in TARGET_TIP_SIZES:
        count = len(df_params[df_params['tips'] == size])
        available_sizes[size] = count
        if count > 0:
            print(f"n={size}: {count} trees available")
        else:
            print(f"n={size}: No trees found")
    
    # Filter to only available sizes
    TARGET_TIP_SIZES = [s for s in TARGET_TIP_SIZES if available_sizes[s] > 0]

if not TARGET_TIP_SIZES:
    print("\n No trees found for analysis!")
    sys.exit(1)


print("\nPerforming Maximum Likelihood Estimation...")
print("-" * 80)
print(f"Using sampling probability: {SAMPLING_PROBA}")
print(f"Models: BD (Birth-Death)")  
# All trees are BD, so only BD model
# print(f"Models: BD (Birth-Death), BDEI (Birth-Death Exposed-Infectious)\n")

# Record start time
start_time = time.time()
all_results = []

for tip_size in TARGET_TIP_SIZES:
    print(f"\n{'='*80}")
    print(f"Analyzing trees with n={tip_size} tips")
    print(f"{'='*80}")
    
    # Get all tree indices for this tip size
    tree_indices = df_params[df_params['tips'] == tip_size]['idx'].values
    num_trees = len(tree_indices)
    
    print(f"Found {num_trees} trees with {tip_size} tips")
    
    # Process each tree
    results_for_size = []
    successful = 0
    failed = 0
    
    for i, tree_idx in enumerate(tree_indices):
        tree_file = os.path.join(TREES_DIR, f"tree_{tree_idx}.nwk")
        
        if not os.path.exists(tree_file):
            print(f"Tree {tree_idx}: File not found, skipping...")
            failed += 1
            continue
        
        try:
            print(f"\n Tree {i+1}/{num_trees} (idx={tree_idx}):", end=" ")
            
            # Initialize MLE estimator
            mle_estimator = BirthDeathMLE(tree_file, sampling_prob=SAMPLING_PROBA)
            
            # Estimate BD model
            print("\n BD model:", end=" ")
            bd_result = mle_estimator.estimate_bd()
            
            if bd_result['success']:
                result_bd = {
                    'tree_idx': tree_idx,
                    'tip_size': tip_size,
                    'model': 'BD',
                    'R_naught': bd_result['R_naught'],
                    'Infectious_period': bd_result['Infectious_period'],
                    'lambda': bd_result['lambda'],
                    'mu': bd_result['mu'],
                    'log_likelihood': bd_result['log_likelihood'],
                    'neg_log_likelihood': bd_result['neg_log_likelihood'],
                    'n_iterations': bd_result['n_iterations']
                }
                results_for_size.append(result_bd)
                print(f"R0={bd_result['R_naught']:.4f}, λ={bd_result['lambda']:.4f}, μ={bd_result['mu']:.4f}")
            else:
                print(f"Failed: {bd_result.get('message', 'Unknown error')}")
            
            # All trees in phylodynamicsDL are BD trees, so skip BDEI estimation
            # # Estimate BDEI model
            # print(" BDEI model:", end=" ")
            # bdei_result = mle_estimator.estimate_bdei()
            # 
            # if bdei_result['success']:
            #     result_bdei = {
            #         'tree_idx': tree_idx,
            #         'tip_size': tip_size,
            #         'model': 'BDEI',
            #         'R_naught': bdei_result['R_naught'],
            #         'Infectious_period': bdei_result['Infectious_period'],
            #         'Incubation_period': bdei_result['Incubation_period'],
            #         'lambda': bdei_result['lambda'],
            #         'mu': bdei_result['mu'],
            #         'sigma': bdei_result['sigma'],
            #         'log_likelihood': bdei_result['log_likelihood'],
            #         'neg_log_likelihood': bdei_result['neg_log_likelihood'],
            #         'n_iterations': bdei_result['n_iterations']
            #     }
            #     results_for_size.append(result_bdei)
            #     print(f"R0={bdei_result['R_naught']:.4f}, λ={bdei_result['lambda']:.4f}, μ={bdei_result['mu']:.4f}, σ={bdei_result['sigma']:.4f}")
            # else:
            #     print(f"Failed: {bdei_result.get('message', 'Unknown error')}")
            
            successful += 1
            
            # Progress update every 5 trees
            if (i + 1) % 5 == 0:
                print(f" Progress: {i+1}/{num_trees} trees processed")
                
        except Exception as e:
            print(f"Error processing tree {tree_idx}: {str(e)[:100]}")
            failed += 1
            continue
    
    print(f"\n Summary for n={tip_size}:")
    print(f"Successfully analyzed: {successful} trees")
    print(f"Failed: {failed} trees")
    
    # Aggregate results for this tip size
    if results_for_size:
        df_size = pd.DataFrame(results_for_size)
        all_results.append(df_size)
        
        # Save individual results for this tip size
        output_file = os.path.join(OUTPUT_DIR, f"mle_estimates_n{tip_size}.csv")
        df_size.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")


print("\n" + "=" * 80)
print("Aggregating MLE results across all tip sizes...")
print("=" * 80)

if not all_results:
    print("No results to aggregate!")
    sys.exit(1)

# Combine all results
df_all = pd.concat(all_results, ignore_index=True)

# Save combined results
combined_file = os.path.join(OUTPUT_DIR, "all_mle_estimates.csv")
df_all.to_csv(combined_file, index=False)
print(f"Saved combined results to: {combined_file}")

# Summary statistics by tip size and model
print("\n" + "-" * 80)
print("SUMMARY STATISTICS BY TIP SIZE AND MODEL (MLE)")
print("-" * 80)

summary_stats = []

for tip_size in TARGET_TIP_SIZES:
    df_size = df_all[df_all['tip_size'] == tip_size]
    
    if len(df_size) == 0:
        continue
    
    print(f"\nTip Size: n={tip_size}")
    num_trees = len(df_size[df_size['model'] == 'BD'].drop_duplicates('tree_idx'))
    print(f"  Total trees analyzed: {num_trees}")
    
    for model_name in df_size['model'].unique():
        df_model = df_size[df_size['model'] == model_name]
        
        print(f"\n Model: {model_name}")
        print(f"Number of estimates: {len(df_model)}")
        
        # Get parameter columns
        param_cols = [col for col in df_model.columns 
                     if col not in ['tree_idx', 'tip_size', 'model', 'log_likelihood', 
                                   'neg_log_likelihood', 'n_iterations', 'lambda', 'mu', 'sigma']]
        
        for param in param_cols:
            values = df_model[param].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                median_val = values.median()
                print(f"      {param}:")
                print(f"        Mean: {mean_val:.6f}")
                print(f"        Std:  {std_val:.6f}")
                print(f"        Median: {median_val:.6f}")
                
                summary_stats.append({
                    'tip_size': tip_size,
                    'model': model_name,
                    'parameter': param,
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'count': len(values)
                })
        
        # Log-likelihood statistics
        if 'log_likelihood' in df_model.columns:
            ll_values = df_model['log_likelihood'].dropna()
            if len(ll_values) > 0:
                print(f"    Log-likelihood:")
                print(f"        Mean: {ll_values.mean():.4f}")
                print(f"        Std:  {ll_values.std():.4f}")
                print(f"        Median: {ll_values.median():.4f}")

# Save summary statistics
if summary_stats:
    df_summary = pd.DataFrame(summary_stats)
    summary_file = os.path.join(OUTPUT_DIR, "mle_summary_statistics.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"\n Saved summary statistics to: {summary_file}")


# Calculate total time taken
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print("\n" + "=" * 80)
print("MLE ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n Total time taken: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f} seconds)")
print(f"\nOutput files saved in: {OUTPUT_DIR}/")
print("  - all_mle_estimates.csv: All MLE estimates for all trees")
print("  - mle_estimates_n{size}.csv: MLE estimates for each tip size")
print("  - mle_summary_statistics.csv: Summary statistics by tip size and model")
print("\n" + "=" * 80)
