# Cell 6: Process MODEL_INPUT with parallel processing
import tqdm
import pickle
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

def flatten_value(v):
    """Convert complex values (lists, dicts) to simple types"""
    if isinstance(v, list):
        # For lists, join elements with comma or take the first element
        if len(v) == 0:
            return 0
        if all(isinstance(x, (int, float)) for x in v):
            return sum(v) / len(v)  # average for numeric lists
        return 1  # indicator that list exists
    elif isinstance(v, dict):
        # For dictionaries, indicate presence
        return 1 if v else 0
    elif isinstance(v, (int, float)):
        return float(v)
    elif isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 1  # indicator that value exists
    return 0

def process_chunk(chunk_data):
    """Process a chunk of MODEL_INPUT data in parallel"""
    chunk_df = pd.DataFrame()
    for row in chunk_data:
        try:
            data = json.loads(row) if isinstance(row, str) else row
            # Flatten complex values
            flattened_data = {k: flatten_value(v) for k, v in data.items()}
            row_df = pd.DataFrame([flattened_data])
            chunk_df = pd.concat([chunk_df, row_df], ignore_index=True)
        except Exception as e:
            print(f"Error in chunk processing: {e}")
            chunk_df = pd.concat([chunk_df, pd.DataFrame([{}])], ignore_index=True)
    return chunk_df

# Configuration
num_cores = mp.cpu_count()  # Get number of CPU cores
chunk_size = 1000  # Size of each chunk
num_chunks = math.ceil(len(ml_training_df) / chunk_size)

print(f"Total records: {len(ml_training_df)}")
print(f"Number of CPU cores available: {num_cores}")
print(f"Number of chunks: {num_chunks}")

# Split data into chunks
model_input_chunks = np.array_split(ml_training_df['MODEL_INPUT'], num_chunks)

# Process chunks in parallel
model_input_df = pd.DataFrame()
processed_chunks = 0

with ProcessPoolExecutor(max_workers=num_cores) as executor:
    # Submit all chunks for processing
    future_to_chunk = {executor.submit(process_chunk, chunk): i 
                      for i, chunk in enumerate(model_input_chunks)}
    
    # Process completed chunks
    for future in tqdm.tqdm(as_completed(future_to_chunk), total=len(model_input_chunks)):
        chunk_idx = future_to_chunk[future]
        try:
            chunk_df = future.result()
            model_input_df = pd.concat([model_input_df, chunk_df], ignore_index=True)
            processed_chunks += 1
            
            # Save intermediate results every 10 chunks
            if processed_chunks % 10 == 0:
                print(f"\nProcessed {processed_chunks}/{num_chunks} chunks")
                print(f"Current shape: {model_input_df.shape}")
                with open(save_dir / f'model_input_processed_{processed_chunks}.pkl', 'wb') as f:
                    pickle.dump(model_input_df, f)
                
        except Exception as e:
            print(f'\nError processing chunk {chunk_idx}: {e}')

# Save final processed MODEL_INPUT
print("\nSaving final processed MODEL_INPUT...")
with open(save_dir / 'model_input_processed_final.pkl', 'wb') as f:
    pickle.dump(model_input_df, f)

print("MODEL_INPUT processing complete!")

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
