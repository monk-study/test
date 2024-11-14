# Cell: Clean up duplicated columns and convert to numeric
print("Cleaning up duplicated columns and converting to numeric...")

# First, let's see what columns are duplicated
duplicated_cols = ml_training_df.columns[ml_training_df.columns.duplicated()]
print("\nDuplicated columns found:")
print(duplicated_cols)

# Remove duplicated columns first
ml_training_df = ml_training_df.loc[:, ~ml_training_df.columns.duplicated()]

# Now convert the problematic hic3 columns
hic3_cols = [
    'times_coupon_hic3_missing',
    'times_paid_cash_hic3_missing',
    'prct_times_paid_cash_hic3_missing',
    'prct_ttl_times_sold_hic3_missing',
    'prct_times_cdc_hic3_missing'
]

for col in hic3_cols:
    if col in ml_training_df.columns:
        try:
            # Convert to numeric and fill NaN with 0
            ml_training_df[col] = pd.to_numeric(ml_training_df[col], errors='coerce').fillna(0)
            print(f"Successfully converted {col}")
        except Exception as e:
            print(f"Error converting {col}: {str(e)}")

# Check final dtypes
print("\nFinal column types summary:")
dtype_counts = ml_training_df.dtypes.value_counts()
print(dtype_counts)

# Verify remaining object columns
object_cols = ml_training_df.select_dtypes(include=['object']).columns
print("\nColumns still in object format:")
print(object_cols.tolist())

# Save the cleaned dataframe
print("\nSaving cleaned dataframe...")
with open(save_dir / 'ml_training_df_cleaned.pkl', 'wb') as f:
    pickle.dump(ml_training_df, f)

print("\nCleaning complete!")
