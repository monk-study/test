# Cell 6: Process MODEL_INPUT with parallel processing
import tqdm
import pickle
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import cupy as cp  # For GPU support
import math

# Create directory for intermediate results if it doesn't exist
save_dir = Path('intermediate_results')
save_dir.mkdir(exist_ok=True)

def process_chunk(chunk_data):
    """Process a chunk of MODEL_INPUT data in parallel"""
    chunk_df = pd.DataFrame()
    for row in chunk_data:
        try:
            data = json.loads(row) if isinstance(row, str) else row
            row_df = pd.DataFrame([data])
            chunk_df = pd.concat([chunk_df, row_df], ignore_index=True)
        except Exception as e:
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
                print(f"\nSaving after {processed_chunks} chunks...")
                with open(save_dir / f'model_input_processed_{processed_chunks}.pkl', 'wb') as f:
                    pickle.dump(model_input_df, f)
                
        except Exception as e:
            print(f'\nError processing chunk {chunk_idx}: {e}')

# Save final processed MODEL_INPUT
print("\nSaving final processed MODEL_INPUT...")
with open(save_dir / 'model_input_processed_final.pkl', 'wb') as f:
    pickle.dump(model_input_df, f)

print("MODEL_INPUT processing complete!")

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
