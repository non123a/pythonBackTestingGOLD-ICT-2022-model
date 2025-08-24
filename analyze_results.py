import os
import json
import statistics
from collections import Counter

# --- Configuration ---
# Set the range of years you want to analyze
START_YEAR = 2009
END_YEAR = 2024

# A list to store the best parameters from each year
all_best_params = []

# --- Step 1: Read all results files ---
for year in range(START_YEAR, END_YEAR + 1):
    folder_name = f'out-{year}'
    file_path = os.path.join(folder_name, 'results.json')
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found for year {year} at {file_path}. Skipping.")
        continue
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            best_params = data['best_parameters']
            all_best_params.append(best_params)
            print(f"Successfully read parameters for year {year}")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if not all_best_params:
    print("No valid results found. Please check your folder names and file paths.")
else:
    print(f"\nAnalysis complete. Found valid results from {len(all_best_params)} years.")
    
    # --- Step 2: Analyze the collected parameters ---
    
    # A dictionary to hold all values for each parameter
    param_values = {
        'pivot': [],
        'sl_mult': [],
        'tp_mult': [],
        'ote': [],
        'tf_filter': []
    }
    
    for params in all_best_params:
        for key, value in params.items():
            if key in param_values:
                param_values[key].append(value)
    
    # --- Step 3: Calculate the most common/average values ---
    
    # For numerical parameters, we'll calculate the mean (average)
    # For the categorical parameter 'tf_filter', we'll find the mode (most common)
    
    final_parameters = {}
    
    # Analyze numerical parameters
    for key in ['pivot', 'sl_mult', 'tp_mult', 'ote']:
        if param_values[key]:
            # Use statistics.median to be more robust to outliers
            final_parameters[key] = statistics.median(param_values[key])
            
    # Analyze categorical parameter
    if param_values['tf_filter']:
        final_parameters['tf_filter'] = Counter(param_values['tf_filter']).most_common(1)[0][0]

    # --- Step 4: Print the final recommended settings ---
    print("\n--- Recommended Robust Parameters (across all years) ---")
    for key, value in final_parameters.items():
        print(f"{key}: {value}")