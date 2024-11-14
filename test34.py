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
#--------

# Cell: Process History Columns in Parallel
import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json

def process_history_row(row_data):
    """Process a single history row"""
    try:
        data = json.loads(row_data) if isinstance(row_data, str) else row_data
        # Convert all values to float where possible
        return {k: float(v) if isinstance(v, (int, float)) else 1.0 
                for k, v in data.items()}
    except Exception as e:
        print(f"Error processing history row: {str(e)}")
        return {}

def process_history_column_parallel(column_data, column_name):
    """Process an entire history column in parallel"""
    num_cores = mp.cpu_count()
    print(f"\nProcessing {column_name} with {num_cores} cores...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm.tqdm(
            executor.map(process_history_row, column_data),
            total=len(column_data),
            desc=column_name
        ))
    
    # Convert results to DataFrame
    history_df = pd.DataFrame(results)
    
    # Add prefix to column names
    history_df.columns = [f'{column_name}_{col}' for col in history_df.columns]
    
    # Convert all columns to numeric, replacing non-numeric with 0
    for col in history_df.columns:
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce').fillna(0)
    
    return history_df

# List of history columns to process
history_columns = [
    'PTNT_GCN_HIST',
    'PTNT_HIC3_HIST',
    'DRUG_GCN_HIST',
    'DRUG_HIC3_HIST'
]

# Process each history column in sequence (but each column processes in parallel)
processed_dfs = {}
for column in history_columns:
    processed_dfs[column] = process_history_column_parallel(
        ml_training_df[column],
        column
    )
    print(f"\n{column} processed:")
    print(f"Shape: {processed_dfs[column].shape}")
    print(f"Sample columns: {list(processed_dfs[column].columns)[:5]}")

# Combine all processed history columns with main dataframe
print("\nCombining all processed columns...")
for column in history_columns:
    ml_training_df = pd.concat([ml_training_df, processed_dfs[column]], axis=1)
    print(f"Added {processed_dfs[column].shape[1]} columns from {column}")

# Drop original history columns
ml_training_df = ml_training_df.drop(columns=history_columns)
#------------

# Cell: Verify numeric columns and handle JSON/complex columns
import pickle
from pathlib import Path

# Load the processed dataframe from pickle
save_dir = Path('intermediate_results')
pickle_path = save_dir / 'final_processed_df.pkl'

print("Loading processed dataframe...")
if pickle_path.exists():
    with open(pickle_path, 'rb') as f:
        ml_training_df = pickle.load(f)
    print("Loaded successfully!")
else:
    print("No saved dataframe found, using current ml_training_df")

print("\nVerifying and processing columns...")

# Function to check if a column is numeric
def is_numeric_dtype(col):
    return pd.api.types.is_numeric_dtype(col)

# Handle HIC3_CD (categorical column)
if 'HIC3_CD' in ml_training_df.columns:
    print("\nProcessing HIC3_CD...")
    # Create dummy variables for HIC3_CD
    hic3_dummies = pd.get_dummies(ml_training_df['HIC3_CD'], prefix='HIC3_CD')
    ml_training_df = pd.concat([ml_training_df, hic3_dummies], axis=1)
    ml_training_df = ml_training_df.drop('HIC3_CD', axis=1)

# Handle TTLTIMESPAID
if 'TTLTIMESPAID' in ml_training_df.columns:
    print("\nProcessing TTLTIMESPAID...")
    ml_training_df['TTLTIMESPAID'] = pd.to_numeric(ml_training_df['TTLTIMESPAID'], errors='coerce').fillna(0)

# Handle JSON columns
json_columns = [
    'MODEL_INPUT_JSON',
    'PTNT_GCN_HIST_JSON',
    'PTNT_HIC3_HIST_JSON',
    'DRUG_GCN_HIST_JSON',
    'DRUG_HIC3_HIST_JSON'
]

for col in json_columns:
    if col in ml_training_df.columns:
        print(f"\nRemoving JSON column: {col}")
        ml_training_df = ml_training_df.drop(col, axis=1)

# Check remaining non-numeric columns
non_numeric_cols = []
for col in ml_training_df.columns:
    if col not in ['label', 'MESSAGE_ID']:
        try:
            if not is_numeric_dtype(ml_training_df[col]):
                print(f"\nChecking column: {col}")
                print(f"Current dtype: {ml_training_df[col].dtype}")
                print("Sample values:")
                print(ml_training_df[col].head())
                non_numeric_cols.append(col)
        except Exception as e:
            print(f"Error checking column {col}: {str(e)}")

if non_numeric_cols:
    print("\nFound non-numeric columns:", non_numeric_cols)
    print("Attempting to convert remaining non-numeric columns...")
    
    for col in non_numeric_cols:
        try:
            if col not in ['label', 'MESSAGE_ID'] + json_columns:
                ml_training_df[col] = pd.to_numeric(ml_training_df[col], errors='coerce').fillna(0)
                print(f"Successfully converted {col} to numeric")
        except Exception as e:
            print(f"Could not convert {col}: {str(e)}")

# Print summary of column types
print("\nColumn types summary:")
print(ml_training_df.dtypes.value_counts())

# Save verified dataframe
print("\nSaving verified dataframe...")
with open(save_dir / 'ml_training_df_verified.pkl', 'wb') as f:
    pickle.dump(ml_training_df, f)

# Print final statistics
print("\nVerification complete!")
print(f"Final shape: {ml_training_df.shape}")
print("\nSample of final columns:")
print(list(ml_training_df.columns)[:5])

# Print multi-label statistics
print("\nMulti-label statistics:")
print(f"Total number of samples:", len(ml_training_df))
print(f"Number of unique label combinations:", ml_training_df['label'].nunique())
print("\nSample counts per NBA:")
nba_list = ['NBA3', 'NBA4', 'NBA5', 'NBA5_CD', 'NBA7', 'NBA8', 'NBA12']
for nba in nba_list:
    count = ml_training_df[ml_training_df['label'].str.contains(f"{nba}_ATTEMPTED")].shape[0]
    print(f"{nba}_ATTEMPTED: {count}")
