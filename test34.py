# Cell 6: Combine Features and Create Labels
# Combine all features
ml_training_df = pd.concat([ml_training_df, ptnt_hist_df, drug_hist_df], axis=1)

# Create multi-label representation with correct NBA numbers
nba_list = ['NBA3', 'NBA4', 'NBA5', 'NBA5_CD', 'NBA7', 'NBA8', 'NBA12']

# Create multi-label representation
ml_training_df['label'] = ml_training_df.apply(
    lambda row: ','.join(
        [f"{nba}_ATTEMPTED" for nba in nba_list 
         if row[f"{nba}_ATTEMPTED"] == 1]
    ),
    axis=1
)

# Print label distribution
print("Label distribution:")
print(ml_training_df['label'].value_counts().head(10))

# Print multi-label statistics
print("\nMulti-label statistics:")
print("Total number of samples:", len(ml_training_df))
print("Number of unique label combinations:", ml_training_df['label'].nunique())
print("\nSample counts per NBA:")
for nba in nba_list:
    count = ml_training_df[ml_training_df['label'].str.contains(f"{nba}_ATTEMPTED")].shape[0]
    print(f"{nba}_ATTEMPTED: {count}")

#cell 9

# Cell 9: Train First Stage Model
# Function to identify insurance cases (NBA4 and NBA5_CD)
def is_insurance_case(label):
    insurance_indicators = ['NBA4_ATTEMPTED', 'NBA5_CD_ATTEMPTED']
    return any(indicator in label for indicator in insurance_indicators)

# Prepare stage 1 data
train_df['is_insurance'] = train_df['label'].apply(is_insurance_case)

# Print distribution of insurance vs non-insurance cases
print("Distribution of cases:")
print(train_df['is_insurance'].value_counts(normalize=True))

# Train stage 1 model
stage1_model = xgb.XGBClassifier(**xgb_params)
stage1_model.fit(train_df[feature_cols], train_df['is_insurance'])

print("\nStage 1 model trained")

# Quick validation on training data
train_insurance_pred = stage1_model.predict(train_df[feature_cols])
print("\nStage 1 Training Performance:")
print(classification_report(train_df['is_insurance'], train_insurance_pred))
