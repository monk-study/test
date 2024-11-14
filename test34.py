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

def check_numeric(series):
    """Check if a pandas Series is numeric"""
    try:
        return pd.api.types.is_numeric_dtype(series.dtype)
    except:
        return False

# Get all columns except label and MESSAGE_ID
columns_to_check = [col for col in ml_training_df.columns 
                   if col not in ['label', 'MESSAGE_ID']]

# First, handle any remaining JSON columns
json_cols = [col for col in columns_to_check if 'JSON' in col or 'FEATURES' in col]
if json_cols:
    print("\nRemoving JSON/FEATURES columns:")
    for col in json_cols:
        print(f"Dropping {col}")
        ml_training_df = ml_training_df.drop(columns=[col])
    columns_to_check = [col for col in columns_to_check if col not in json_cols]

# Process HIC3_CD separately if it exists
if 'HIC3_CD' in columns_to_check:
    print("\nProcessing HIC3_CD...")
    hic3_dummies = pd.get_dummies(ml_training_df['HIC3_CD'], prefix='HIC3_CD')
    ml_training_df = pd.concat([ml_training_df, hic3_dummies], axis=1)
    ml_training_df = ml_training_df.drop('HIC3_CD', axis=1)
    columns_to_check.remove('HIC3_CD')

# Check and convert remaining columns
non_numeric_cols = []
for col in columns_to_check:
    try:
        if not check_numeric(ml_training_df[col]):
            print(f"\nNon-numeric column found: {col}")
            print(f"Current dtype: {ml_training_df[col].dtype}")
            print("Sample values:")
            print(ml_training_df[col].head())
            non_numeric_cols.append(col)
    except Exception as e:
        print(f"Error checking column {col}: {str(e)}")

# Convert non-numeric columns to numeric
if non_numeric_cols:
    print("\nConverting non-numeric columns to numeric...")
    for col in non_numeric_cols:
        try:
            ml_training_df[col] = pd.to_numeric(ml_training_df[col].astype(str).str.strip(), 
                                              errors='coerce').fillna(0)
            print(f"Successfully converted {col}")
        except Exception as e:
            print(f"Error converting {col}: {str(e)}")

# Verify final dtypes
print("\nFinal column types summary:")
dtype_counts = ml_training_df.dtypes.value_counts()
print(dtype_counts)

# Check for any remaining object columns
object_cols = ml_training_df.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print("\nWarning: Found remaining object columns:", list(object_cols))

# Save verified dataframe
print("\nSaving verified dataframe...")
with open(save_dir / 'ml_training_df_verified.pkl', 'wb') as f:
    pickle.dump(ml_training_df, f)

print("\nVerification complete!")
print(f"Final shape: {ml_training_df.shape}")

# Print multi-label statistics
print("\nMulti-label statistics:")
print(f"Total number of samples:", len(ml_training_df))
print(f"Number of unique label combinations:", ml_training_df['label'].nunique())
print("\nSample counts per NBA:")
nba_list = ['NBA3', 'NBA4', 'NBA5', 'NBA5_CD', 'NBA7', 'NBA8', 'NBA12']
for nba in nba_list:
    count = ml_training_df[ml_training_df['label'].str.contains(f"{nba}_ATTEMPTED")].shape[0]
    print(f"{nba}_ATTEMPTED: {count}")
