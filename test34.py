# Cell 6: Process MODEL_INPUT with parallel processing
# Cell 6: Process MODEL_INPUT with parallel processing
import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json

def flatten_value(v):
    """Convert complex values (lists, dicts) to simple types"""
    if isinstance(v, list):
        if len(v) == 0:
            return 0
        if all(isinstance(x, (int, float)) for x in v):
            return sum(v) / len(v)  # average for numeric lists
        return 1  # indicator that list exists
    elif isinstance(v, dict):
        return 1 if v else 0
    elif isinstance(v, (int, float)):
        return float(v)
    elif isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 1
    return 0

def process_json_data(data):
    """Process JSON data handling both dict and list formats"""
    if isinstance(data, list):
        return {
            'list_length': len(data),
            'has_content': 1 if len(data) > 0 else 0
        }
    elif isinstance(data, dict):
        return {k: flatten_value(v) for k, v in data.items()}
    else:
        return {}

def process_row(row):
    """Process a single row of MODEL_INPUT data"""
    try:
        data = json.loads(row) if isinstance(row, str) else row
        return process_json_data(data)
    except Exception as e:
        print(f"Error processing row: {str(e)}")
        return {}

# Configuration
num_cores = mp.cpu_count()
print(f"Number of CPU cores available: {num_cores}")
print(f"Total records to process: {len(ml_training_df)}")

# Process using parallel execution
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    print("Starting parallel processing...")
    results = list(tqdm.tqdm(
        executor.map(process_row, ml_training_df['MODEL_INPUT']), 
        total=len(ml_training_df)
    ))

# Convert results to DataFrame
print("Converting results to DataFrame...")
model_input_df = pd.DataFrame(results)

print("\nProcessing complete!")
print(f"Final shape: {model_input_df.shape}")
print("\nSample of processed data:")
print(model_input_df.head())
print("\nColumn names:")
print(model_input_df.columns.tolist())

#----------
# Cell 7: Verify and clean numeric data
print("Verifying numeric conversion...")

def verify_numeric_column(series):
    """Verify if a column can be converted to numeric and handle errors"""
    try:
        return pd.to_numeric(series, errors='raise')
    except:
        # If conversion fails, print sample of problematic values
        problematic = series[~series.apply(lambda x: isinstance(x, (int, float)))]
        if len(problematic) > 0:
            print(f"\nColumn {series.name} has non-numeric values. Sample:")
            print(problematic.head())
        # Return original series for further inspection
        return series

# Load the processed MODEL_INPUT if not in memory
if 'model_input_df' not in locals():
    print("Loading processed MODEL_INPUT...")
    with open(save_dir / 'model_input_processed_final.pkl', 'rb') as f:
        model_input_df = pickle.load(f)

# Verify each column
print("\nChecking column types...")
for col in model_input_df.columns:
    model_input_df[col] = verify_numeric_column(model_input_df[col])

# Print summary of column types
print("\nColumn types summary:")
print(model_input_df.dtypes.value_counts())

# Save verified numeric version
print("\nSaving verified MODEL_INPUT...")
with open(save_dir / 'model_input_numeric_verified.pkl', 'wb') as f:
    pickle.dump(model_input_df, f)

print("Verification complete!")
print(f"Final shape: {model_input_df.shape}")

# ---------------

# Cell 7: Convert MODEL_INPUT columns to numeric (using GPU if available)
print("Converting MODEL_INPUT columns to numeric...")

def convert_columns_to_numeric(df):
    """Convert columns to numeric, handling categorical ones"""
    result_df = df.copy()
    for col in tqdm.tqdm(df.columns):
        try:
            # Try GPU acceleration if available
            try:
                # Move to GPU
                col_gpu = cp.array(df[col].values)
                # Convert to numeric
                result_df[col] = cp.asnumeric(col_gpu).get()
            except:
                # Fallback to CPU
                result_df[col] = pd.to_numeric(df[col], errors='raise')
        except:
            print(f"\nColumn {col} contains non-numeric values, creating dummies...")
            dummies = pd.get_dummies(df[col], prefix=f'MODEL_INPUT_{col}')
            result_df = pd.concat([result_df.drop(columns=[col]), dummies], axis=1)
    
    return result_df

# Load the processed MODEL_INPUT if not in memory
if 'model_input_df' not in locals():
    print("Loading processed MODEL_INPUT...")
    with open(save_dir / 'model_input_processed_final.pkl', 'rb') as f:
        model_input_df = pickle.load(f)

# Convert to numeric
model_input_df = convert_columns_to_numeric(model_input_df)

# Save numeric version
print("\nSaving numeric MODEL_INPUT...")
with open(save_dir / 'model_input_numeric_final.pkl', 'wb') as f:
    pickle.dump(model_input_df, f)

print("Numeric conversion complete!")
print(f"Final shape: {model_input_df.shape}")
