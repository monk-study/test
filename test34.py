# Cell: Split and train stage-specific models
print("Splitting data based on insurance vs non-insurance...")

# Create insurance mask properly
insurance_mask = train_df['is_insurance'] == 1
print(f"Number of insurance cases: {insurance_mask.sum()}")
print(f"Number of non-insurance cases: {(~insurance_mask).sum()}")

# Split data ensuring we're using proper indexing
insurance_X = train_df[feature_cols].loc[insurance_mask]
insurance_y = train_df['label'].loc[insurance_mask]

non_insurance_X = train_df[feature_cols].loc[~insurance_mask]
non_insurance_y = train_df['label'].loc[~insurance_mask]

print("\nTraining specialized models...")
# Train insurance model
if len(insurance_X) > 0:
    print(f"\nTraining insurance model on {len(insurance_X)} samples")
    insurance_model = xgb.XGBClassifier(**xgb_params)
    insurance_model.fit(insurance_X, insurance_y)
    print("Insurance model trained")

# Train non-insurance model
if len(non_insurance_X) > 0:
    print(f"\nTraining non-insurance model on {len(non_insurance_X)} samples")
    non_insurance_model = xgb.XGBClassifier(**xgb_params)
    non_insurance_model.fit(non_insurance_X, non_insurance_y)
    print("Non-insurance model trained")

# Make predictions function
def make_predictions(X):
    """Make predictions using both stages"""
    # Stage 1: Predict insurance vs non-insurance
    insurance_pred = stage1_model.predict(X)
    predictions = np.empty(len(X), dtype=object)
    
    # Process insurance cases
    insurance_mask = insurance_pred == 1
    if any(insurance_mask):
        predictions[insurance_mask] = insurance_model.predict(X[insurance_mask])
    
    # Process non-insurance cases
    non_insurance_mask = ~insurance_mask
    if any(non_insurance_mask):
        predictions[non_insurance_mask] = non_insurance_model.predict(X[non_insurance_mask])
    
    return predictions
