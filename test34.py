# Get number of available CPU cores
num_cores = mp.cpu_count()
print(f"Number of CPU cores available: {num_cores}")

# XGBoost parameters with multi-threading
xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'nthread': num_cores,  # Use all available cores
    'tree_method': 'hist',  # Faster tree construction method
    'predictor': 'cpu_predictor',
    'verbosity': 2  # To see more information about training
}
xgb_params['tree_method'] = 'gpu_hist'
xgb_params['predictor'] = 'gpu_predictor'
xgb_params.update({
    'parallel_tree': 'true',  # Build trees in parallel
    'sampling_method': 'gradient_based',  # More efficient sampling
    'grow_policy': 'lossguide'  # More efficient tree growing
})

print("Training stage 1 model using all cores...")
stage1_model = xgb.XGBClassifier(**xgb_params)
stage1_model.fit(train_df[feature_cols], train_df['is_insurance'])
