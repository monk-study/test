# Cell 6: Combine Features and Create Labels with Type Conversion
# First ensure all numeric columns are properly typed
numeric_columns = [col for col in ml_training_df.columns 
                  if col.endswith('_ATTEMPTED')]
for col in numeric_columns:
    ml_training_df[col] = ml_training_df[col].astype(float)

def process_json_value(value):
    """Convert value to numeric, handling categorical values"""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            # For categorical values, return 1.0 to indicate presence
            return 1.0
    return 0.0

def extract_json_features(json_str):
    """Safely convert JSON string to features, handling categorical values"""
    try:
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        return {k: process_json_value(v) for k, v in data.items()}
    except:
        return {}

# Process JSON columns safely
print("Processing JSON columns...")
for prefix in ['MODEL_INPUT', 'PTNT_GCN_HIST', 'PTNT_HIC3_HIST', 'DRUG_GCN_HIST', 'DRUG_HIC3_HIST']:
    print(f"Processing {prefix}...")
    features_col = f"{prefix}_FEATURES"
    ml_training_df[features_col] = ml_training_df[prefix].apply(extract_json_features)
    
    # Convert features to columns
    temp_df = pd.json_normalize(ml_training_df[features_col])
    
    # Handle categorical columns
    for col in temp_df.columns:
        unique_values = temp_df[col].unique()
        # Check if column contains strings
        if any(isinstance(x, str) for x in unique_values if pd.notna(x)):
            # Create indicator columns for categorical values
            print(f"Creating indicators for categorical column: {col}")
            dummies = pd.get_dummies(temp_df[col], prefix=f"{prefix}_{col}")
            temp_df = pd.concat([temp_df.drop(columns=[col]), dummies], axis=1)
        else:
            # Convert numeric columns
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
    
    # Add prefix to avoid column name conflicts
    temp_df.columns = [f'{prefix}_{col}' for col in temp_df.columns]
    ml_training_df = pd.concat([ml_training_df, temp_df], axis=1)
    print(f"Added {len(temp_df.columns)} features from {prefix}")

# Create labels
nba_list = ['NBA3', 'NBA4', 'NBA5', 'NBA5_CD', 'NBA7', 'NBA8', 'NBA12']
ml_training_df['label'] = ml_training_df.apply(
    lambda row: ','.join(
        [f"{nba}_ATTEMPTED" for nba in nba_list 
         if row[f"{nba}_ATTEMPTED"] == 1]
    ),
    axis=1
)

# Drop original JSON columns and features
columns_to_drop = [col for col in ml_training_df.columns 
                  if col.endswith(('_JSON', '_FEATURES', '_HIST'))]
ml_training_df = ml_training_df.drop(columns=columns_to_drop)

# Verify all columns are numeric
for col in ml_training_df.columns:
    if col not in ['label', 'MESSAGE_ID']:
        try:
            ml_training_df[col] = pd.to_numeric(ml_training_df[col], errors='coerce').fillna(0)
        except:
            print(f"Warning: Could not convert {col} to numeric")

print("\nFeature processing complete.")
print(f"Final dataframe shape: {ml_training_df.shape}")
print("\nSample of columns and their types:")
print(ml_training_df.dtypes.head())
