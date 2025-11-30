"""
Estimation of parameters using PhyloDeep:

- Loads trees from phylodynamicsDL/output_trees/
- Groups trees by the tip size (n=50, 100, 200, 500)
- Uses PhyloDeep to estimate parameters for each tree
- Aggregates and reports results for each tip size

"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from phylodeep import BD, FULL 
# from phylodeep import BD, BDEI, BDSS, FULL  
from phylodeep.paramdeep import paramdeep
# from phylodeep.modeldeep import modeldeep  # Not needed since all trees are BD for now
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PhyloDynamicsDL Tree Analysis by Tip Size")
print("=" * 80)

# Configuration
TARGET_TIP_SIZES = [50, 100, 200, 500, 1000]

# Analyzing all trees with tip size >= 50 (PhyloDeep minimum requirement)
TARGET_TIP_SIZES = None  # Set to None to analyze all trees >= 50 tips
MIN_TIP_SIZE = 50  # PhyloDeep requires at least 50 tips

TREES_DIR = "phylodynamicsDL/output_trees"
PARAMS_FILE = "phylodynamicsDL/all_params.csv"
SAMPLING_PROBA = 0.5  # Default sampling probability
OUTPUT_DIR = "parameter_estimates"

# Creating theoutput directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the parameters file to get tree indices and tip sizes
print("\nLoading tree metadata...")
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
    print("\nNo trees found for analysis!!")
    exit(1)

print("\nAnalyzing trees and estimating parameters...")
print("-" * 80)
print(f"Using sampling probability: {SAMPLING_PROBA}")
print(f"Using FULL tree representation (most accurate)\n")

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
    
    # Determine which models are available
    # All trees in phylodynamicsDL are BD trees, so only analyze BD model
    # if tip_size < 200:
    #     available_models = [BD, BDEI]
    #     model_names = ['BD', 'BDEI']
    # else:
    #     available_models = [BD, BDEI, BDSS]
    #     model_names = ['BD', 'BDEI', 'BDSS']
    available_models = [BD]
    model_names = ['BD']
    
    print(f"Available models: {', '.join(model_names)}")
    
    # Processing each tree
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
            # Skip model selection since all trees are BD
            # print(f"\n  Tree {i+1}/{num_trees} (idx={tree_idx}):", end=" ")
            # 
            # model_selection = modeldeep(
            #     tree_file,
            #     SAMPLING_PROBA,
            #     vector_representation=FULL
            # )
            
            # Determine best model - can skip this process if needed
            print(f"\n  Tree {i+1}/{num_trees} (idx={tree_idx}):", end=" ")
            best_model = BD
            best_model_name = "BD"
            best_prob = 1.0  # All trees are BD
            
            print(f"Model: {best_model_name}")
            
            # Estimate parameters using the best model
            # We can also try all available models for comparison
            for model, model_name in zip(available_models, model_names):
                try:
                    params = paramdeep(
                        tree_file,
                        SAMPLING_PROBA,
                        model=model,
                        vector_representation=FULL,
                        ci_computation=False  # Faster, point estimates only
                    )
                    
                    # Storing theresults
                    result = {
                        'tree_idx': tree_idx,
                        'tip_size': tip_size,
                        'model': model_name,
                        'is_best_model': (model_name == best_model_name),
                        'model_selection_prob': best_prob if model_name == best_model_name else None
                    }
                    
                    # Add parameter estimates
                    for param_name in params.columns:
                        result[param_name] = params[param_name].iloc[0]
                    
                    results_for_size.append(result)
                    
                except Exception as e:
                    print(f"Error estimating {model_name} parameters: {str(e)[:50]}")
                    continue
            
            successful += 1
            
            # Progress update every 10 trees
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_trees} trees processed")
                
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
        output_file = os.path.join(OUTPUT_DIR, f"estimates_n{tip_size}.csv")
        df_size.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")

print("\n" + "=" * 80)
print("Aggregating results across all tip sizes...")
print("=" * 80)

if not all_results:
    print("No results to aggregate!!")
    exit(1)

# Combine all results
df_all = pd.concat(all_results, ignore_index=True)

# Save combined results
combined_file = os.path.join(OUTPUT_DIR, "all_estimates.csv")
df_all.to_csv(combined_file, index=False)
print(f"Saved combined results to: {combined_file}")

# Summary statistics by tip size and model
print("\n" + "-" * 80)
print("SUMMARY STATISTICS BY TIP SIZE AND MODEL")
print("-" * 80)

summary_stats = []

for tip_size in TARGET_TIP_SIZES:
    df_size = df_all[df_all['tip_size'] == tip_size]
    
    if len(df_size) == 0:
        continue
    
    print(f"\nTip Size: n={tip_size}")
    print(f"Total trees analyzed: {len(df_size[df_size['is_best_model'] == True].drop_duplicates('tree_idx'))}")
    
    for model_name in df_size['model'].unique():
        df_model = df_size[df_size['model'] == model_name]
        
        print(f"\n  Model: {model_name}")
        print(f"Number of estimates: {len(df_model)}")
        
        # Get parameter columns (exclude metadata columns)
        param_cols = [col for col in df_model.columns 
                     if col not in ['tree_idx', 'tip_size', 'model', 'is_best_model', 'model_selection_prob']]
        
        for param in param_cols:
            values = df_model[param].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                median_val = values.median()
                print(f"      {param}:")
                print(f"Mean: {mean_val:.6f}")
                print(f"Std:  {std_val:.6f}")
                print(f"Median: {median_val:.6f}")
                
                summary_stats.append({
                    'tip_size': tip_size,
                    'model': model_name,
                    'parameter': param,
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'count': len(values)
                })

# Save summary statistics
if summary_stats:
    df_summary = pd.DataFrame(summary_stats)
    summary_file = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"\n✓ Saved summary statistics to: {summary_file}")


# Calculate total time taken
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n Total time taken: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f} seconds)")
print(f"\nOutput files saved in: {OUTPUT_DIR}/")
print("all_estimates.csv: All parameter estimates for all trees")
print("estimates_n{size}.csv: Estimates for each tip size")
print("summary_statistics.csv: Summary statistics by tip size and model")
print("\n" + "=" * 80)
