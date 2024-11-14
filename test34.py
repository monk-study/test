# Cell 9: Train First Stage Model (Updated)
# Get feature columns, excluding non-feature columns
feature_cols = [col for col in ml_training_df.columns 
               if col not in ['label', 'MESSAGE_ID', 'is_insurance'] 
               and not col.endswith('_ATTEMPTED')]

# Convert all feature columns to float
X_train = train_df[feature_cols].astype(float)

# Prepare insurance labels
train_df['is_insurance'] = train_df['label'].apply(
    lambda x: 1.0 if any(indicator in x 
                        for indicator in ['NBA4_ATTEMPTED', 'NBA5_CD_ATTEMPTED']) 
    else 0.0
)

print("Feature columns shape:", X_train.shape)
print("Sample of feature values:")
print(X_train.head())

# Train stage 1 model
stage1_model = xgb.XGBClassifier(**xgb_params)
stage1_model.fit(X_train, train_df['is_insurance'])

print("\nStage 1 model trained")
