# Cell 6: Combine Features and Create Labels with Type Conversion
# First ensure all numeric columns are properly typed
numeric_columns = [col for col in ml_training_df.columns 
                  if col.endswith('_ATTEMPTED')]
for col in numeric_columns:
    ml_training_df[col] = ml_training_df[col].astype(float)

print("Starting feature processing...")

# Process MODEL_INPUT separately first as it has different structure
print("Processing MODEL_INPUT...")
model_input_df = pd.DataFrame()
for idx, row in ml_training_df['MODEL_INPUT'].items():
    try:
        data = json.loads(row) if isinstance(row, str) else row
        # Convert to DataFrame row and handle non-numeric values
        row_df = pd.DataFrame([data])
        model_input_df = pd.concat([model_input_df, row_df], ignore_index=True)
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        # Add empty row to maintain index alignment
        model_input_df = pd.concat([model_input_df, pd.DataFrame([{}])], ignore_index=True)

# Convert numeric columns and handle categorical ones
for col in model_input_df.columns:
    try:
        model_input_df[col] = pd.to_numeric(model_input_df[col], errors='raise')
    except:
        print(f"Column {col} contains non-numeric values, creating dummies...")
        # Create dummies for categorical columns
        dummies = pd.get_dummies(model_input_df[col], prefix=f'MODEL_INPUT_{col}')
        model_input_df = pd.concat([model_input_df.drop(columns=[col]), dummies], axis=1)

# Process history columns
for prefix in ['PTNT_GCN_HIST', 'PTNT_HIC3_HIST', 'DRUG_GCN_HIST', 'DRUG_HIC3_HIST']:
    print(f"Processing {prefix}...")
    history_df = pd.DataFrame()
    
    for idx, row in ml_training_df[prefix].items():
        try:
            data = json.loads(row) if isinstance(row, str) else row
            # Convert all values to float where possible
            processed_data = {k: float(v) if isinstance(v, (int, float)) else 1.0 
                            for k, v in data.items()}
            history_df = pd.concat([history_df, pd.DataFrame([processed_data])], 
                                 ignore_index=True)
        except Exception as e:
            print(f"Error processing row {idx} in {prefix}: {e}")
            history_df = pd.concat([history_df, pd.DataFrame([{}])], 
                                 ignore_index=True)
    
    # Add prefix to column names
    history_df.columns = [f'{prefix}_{col}' for col in history_df.columns]
    
    # Convert all columns to numeric, replacing non-numeric with 0
    for col in history_df.columns:
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce').fillna(0)
    
    # Concatenate with main dataframe
    ml_training_df = pd.concat([ml_training_df, history_df], axis=1)
    print(f"Added {len(history_df.columns)} features from {prefix}")

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

print("\nFeature processing complete.")
print(f"Final dataframe shape: {ml_training_df.shape}")
print("\nSample of columns and their types:")
print(ml_training_df.dtypes.head())

# Verify all columns are numeric except label and MESSAGE_ID
non_numeric_cols = []
for col in ml_training_df.columns:
    if col not in ['label', 'MESSAGE_ID']:
        if not np.issubdtype(ml_training_df[col].dtype, np.number):
            non_numeric_cols.append(col)

if non_numeric_cols:
    print("\nWarning: Found non-numeric columns:", non_numeric_cols)
