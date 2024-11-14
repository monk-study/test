# Cell 6: Combine Features and Create Labels with Type Conversion
import tqdm
import pickle
from pathlib import Path

# Create directory for intermediate results if it doesn't exist
save_dir = Path('intermediate_results')
save_dir.mkdir(exist_ok=True)

print("Starting feature processing...")

# Process MODEL_INPUT with progress bar and chunking
print("Processing MODEL_INPUT...")
chunk_size = 10000  # Process 10000 rows at a time
total_rows = len(ml_training_df)
model_input_df = pd.DataFrame()

# Check if we have a saved intermediate result
intermediate_file = save_dir / 'model_input_processed.pkl'
if intermediate_file.exists():
    print("Loading previously processed MODEL_INPUT data...")
    with open(intermediate_file, 'rb') as f:
        model_input_df = pickle.load(f)
    print("Loaded saved progress.")
else:
    for chunk_start in tqdm.tqdm(range(0, total_rows, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_df = pd.DataFrame()
        
        for idx in range(chunk_start, chunk_end):
            try:
                row = ml_training_df['MODEL_INPUT'].iloc[idx]
                data = json.loads(row) if isinstance(row, str) else row
                row_df = pd.DataFrame([data])
                chunk_df = pd.concat([chunk_df, row_df], ignore_index=True)
            except Exception as e:
                print(f"\nError processing row {idx}: {e}")
                chunk_df = pd.concat([chunk_df, pd.DataFrame([{}])], ignore_index=True)
        
        model_input_df = pd.concat([model_input_df, chunk_df], ignore_index=True)
        
        # Save intermediate result every 5 chunks
        if (chunk_start // chunk_size) % 5 == 0:
            print(f"\nSaving intermediate result at row {chunk_end}...")
            with open(intermediate_file, 'wb') as f:
                pickle.dump(model_input_df, f)

    # Save final result
    with open(intermediate_file, 'wb') as f:
        pickle.dump(model_input_df, f)

print("\nConverting MODEL_INPUT columns to numeric...")
for col in tqdm.tqdm(model_input_df.columns):
    try:
        model_input_df[col] = pd.to_numeric(model_input_df[col], errors='raise')
    except:
        print(f"\nColumn {col} contains non-numeric values, creating dummies...")
        dummies = pd.get_dummies(model_input_df[col], prefix=f'MODEL_INPUT_{col}')
        model_input_df = pd.concat([model_input_df.drop(columns=[col]), dummies], axis=1)

# Save processed MODEL_INPUT
print("\nSaving processed MODEL_INPUT...")
with open(save_dir / 'model_input_processed_final.pkl', 'wb') as f:
    pickle.dump(model_input_df, f)

# Process history columns with progress tracking
for prefix in ['PTNT_GCN_HIST', 'PTNT_HIC3_HIST', 'DRUG_GCN_HIST', 'DRUG_HIC3_HIST']:
    print(f"\nProcessing {prefix}...")
    history_file = save_dir / f'{prefix}_processed.pkl'
    
    if history_file.exists():
        print(f"Loading previously processed {prefix} data...")
        with open(history_file, 'rb') as f:
            history_df = pickle.load(f)
    else:
        history_df = pd.DataFrame()
        
        for idx in tqdm.tqdm(range(total_rows)):
            try:
                row = ml_training_df[prefix].iloc[idx]
                data = json.loads(row) if isinstance(row, str) else row
                processed_data = {k: float(v) if isinstance(v, (int, float)) else 1.0 
                                for k, v in data.items()}
                history_df = pd.concat([history_df, pd.DataFrame([processed_data])], 
                                     ignore_index=True)
                
                # Save intermediate result every 10000 rows
                if idx % 10000 == 0:
                    with open(history_file, 'wb') as f:
                        pickle.dump(history_df, f)
                        
            except Exception as e:
                print(f"\nError processing row {idx} in {prefix}: {e}")
                history_df = pd.concat([history_df, pd.DataFrame([{}])], 
                                     ignore_index=True)
        
        # Save final result
        with open(history_file, 'wb') as f:
            pickle.dump(history_df, f)
    
    # Add prefix to column names
    history_df.columns = [f'{prefix}_{col}' for col in history_df.columns]
    
    # Convert all columns to numeric
    print(f"Converting {prefix} columns to numeric...")
    for col in tqdm.tqdm(history_df.columns):
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce').fillna(0)
    
    # Concatenate with main dataframe
    ml_training_df = pd.concat([ml_training_df, history_df], axis=1)
    print(f"Added {len(history_df.columns)} features from {prefix}")

print("\nCreating labels...")
# Create labels
nba_list = ['NBA3', 'NBA4', 'NBA5', 'NBA5_CD', 'NBA7', 'NBA8', 'NBA12']
ml_training_df['label'] = ml_training_df.apply(
    lambda row: ','.join(
        [f"{nba}_ATTEMPTED" for nba in nba_list 
         if row[f"{nba}_ATTEMPTED"] == 1]
    ),
    axis=1
)

# Drop original JSON columns
columns_to_drop = [col for col in ml_training_df.columns 
                  if col in ['MODEL_INPUT', 'PTNT_GCN_HIST', 'PTNT_HIC3_HIST', 
                            'DRUG_GCN_HIST', 'DRUG_HIC3_HIST']]
ml_training_df = ml_training_df.drop(columns=columns_to_drop)

# Save final processed dataframe
print("\nSaving final processed dataframe...")
with open(save_dir / 'final_processed_df.pkl', 'wb') as f:
    pickle.dump(ml_training_df, f)

print("\nFeature processing complete.")
print(f"Final dataframe shape: {ml_training_df.shape}")
print("\nSample of columns and their types:")
print(ml_training_df.dtypes.head())
