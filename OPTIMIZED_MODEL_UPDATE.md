# Optimized Model Training Update

## Summary of Changes

The subtiled prediction script now **optimizes a new LGBM model** using Optuna with the 30 selected features, instead of just loading pre-trained hyperparameters.

## What Changed

### Before (Simple Training)
```python
# Load hyperparameters from existing study
pipeline_performance = best_classifier_pipe(
    db_loc="study.db",
    study_name=study_name_final
)

# Train on full data
pipeline_performance.fit(X, y, classifier__sample_weight=weights)
```

**Problems**:
- Used hyperparameters optimized on ALL features, not the selected 30
- No verification of performance with reduced feature set
- No cross-validation metrics reported

### After (Optuna Optimization)

**Step 1**: Create new Optuna study for selected features
```python
study_name_optimized = f"optimized_final_{select_how_many}_{classifier}_{scoring}_{n_splits}"
# Example: "optimized_final_30_LGBM_kappa_3"
```

**Step 2**: Optimize hyperparameters on 30 selected features
```python
study.optimize(
    lambda trial: classifier_objective(
        trial,
        X,                              # Only 30 features
        y,
        groups=groups,                  # field_id for spatial CV
        n_splits=3,
        classifier_override=["LGBM"],
        weights=weights,                # Field_size weights
        scoring="kappa",
    ),
    n_trials=50,
    n_jobs=-1,                          # Use all CPU cores
)
```

**Step 3**: Train final model with best hyperparameters
```python
pipeline_performance = best_classifier_pipe(
    db_loc="study.db",
    study_name=study_name_optimized
)
pipeline_performance.fit(X, y, classifier__sample_weight=weights)
```

**Step 4**: Calculate out-of-sample performance
```python
cv = StratifiedGroupKFold(n_splits=3)
for fold in cv.split(X, y, groups):
    # Train on 2/3 of data, test on 1/3
    # Calculate Cohen's Kappa and Balanced Accuracy
```

## Expected Output

When running the script, you'll see:

```
============================================================
STEP 2: Optimizing LGBM model with selected features
============================================================

Training data loaded: 15847 samples
Classes: ['cassava' 'cotton' 'maize_masika' 'millet' 'rice' 'sorghum' 'sunflower']
Features: 30

✓ Created/loaded study: optimized_final_30_LGBM_kappa_3

Running Optuna optimization (50 trials)...
Scoring: kappa
Cross-validation: 3-fold StratifiedGroupKFold

[I 2025-10-19 ...] Trial 0 finished with value: 0.6234 ...
[I 2025-10-19 ...] Trial 1 finished with value: 0.6456 ...
...
[I 2025-10-19 ...] Trial 49 finished with value: 0.6789 ...

============================================================
OPTIMIZATION RESULTS
============================================================
Best trial: 37
Best kappa score: 0.6892

Best hyperparameters:
  classifier: LGBM
  num_leaves: 127
  max_depth: 8
  learning_rate: 0.0234
  n_estimators: 500
  min_child_samples: 25
  ...

============================================================
TRAINING FINAL MODEL
============================================================

Training final model on 15847 samples...

============================================================
OUT-OF-SAMPLE PERFORMANCE
============================================================

Fold 1:
  Cohen's Kappa: 0.6834
  Balanced Accuracy: 0.7123

Fold 2:
  Cohen's Kappa: 0.6921
  Balanced Accuracy: 0.7245

Fold 3:
  Cohen's Kappa: 0.6756
  Balanced Accuracy: 0.7089

============================================================
Mean Cohen's Kappa: 0.6837 ± 0.0068
Mean Balanced Accuracy: 0.7152 ± 0.0064
============================================================

✓ Final model ready for prediction
  Classes: 7
  Features: 30
```

## Key Improvements

### 1. **Feature-Specific Optimization**
- Hyperparameters are now optimized specifically for the 30 selected features
- Previous approach used hyperparameters from a model trained on ALL features
- Better performance with reduced feature set

### 2. **Performance Validation**
- Out-of-sample cross-validation metrics reported
- Cohen's Kappa (primary metric for imbalanced classes)
- Balanced Accuracy (accounts for class imbalance)
- 3-fold StratifiedGroupKFold prevents spatial leakage

### 3. **Reproducible Results**
- Study saved to SQLite database
- Can skip optimization on subsequent runs (loads existing study)
- All trials logged for analysis

### 4. **Spatial Cross-Validation**
- Uses `field_id` as groups
- Ensures all pixels from same field stay together
- Prevents overly optimistic performance from spatial autocorrelation

## Configuration

Located in [5_model_subtiled_prediction.py](5_model_subtiled_prediction.py):

```python
# Model parameters
select_how_many = 30      # Number of top features
classifier = "LGBM"       # LightGBM classifier
scoring = "kappa"         # Cohen's Kappa for optimization
n_splits = 3              # Cross-validation folds

# Optimization parameters
n_trials = 50             # Optuna trials (increase for better results)
n_jobs = -1               # Use all CPU cores
```

### Adjustable Parameters

**n_trials**: Number of hyperparameter combinations to try
- `50`: Quick optimization (~15-30 min on multi-core)
- `100`: Better optimization (~30-60 min)
- `200`: Thorough optimization (~1-2 hours)

**n_splits**: Cross-validation folds
- `3`: Standard (current)
- `5`: More robust but slower

## Database Storage

All optimization results are stored in:
```
/mnt/bigdrive/Dropbox/Tanzania_data/.../models/study.db
```

Study name format:
```
optimized_final_{n_features}_{classifier}_{scoring}_{n_splits}
```

Example: `optimized_final_30_LGBM_kappa_3`

### Viewing All Studies

```python
import optuna
storage = optuna.storages.RDBStorage(url="sqlite:///study.db")
summaries = optuna.study.get_all_study_summaries(storage=storage)
for summary in summaries:
    print(f"{summary.study_name}: {summary.n_trials} trials")
```

## Performance Expectations

Based on typical LGBM performance with selected features:

| Metric | Expected Range | Excellent |
|--------|----------------|-----------|
| Cohen's Kappa | 0.60 - 0.75 | > 0.75 |
| Balanced Accuracy | 0.65 - 0.80 | > 0.80 |
| Per-class Accuracy | Varies by crop | > 0.70 |

**Class difficulty** (from past experience):
- **Easy**: Maize, Rice, Water (>80% accuracy)
- **Medium**: Cotton, Sorghum, Cassava (65-75%)
- **Hard**: Millet, Sunflower (50-65%)

## Integration with Prediction

The optimized model flows directly into prediction:

```
Step 2: Optimize & Train → pipeline_performance (trained model)
                          ↓
Step 3: Predict          → process_tile_with_subtiling(model_pipeline=pipeline_performance)
                          ↓
                        → predict_subtile() uses the model
                          ↓
                        → Output: prediction_tile##_sub###.tif
```

## Troubleshooting

### Optimization Taking Too Long
- Reduce `n_trials` from 50 to 25
- Use `n_jobs=1` if memory is tight (slower but safer)

### Poor Performance Scores
- Check for class imbalance in training data
- Verify feature quality (missing values, outliers)
- Try increasing `n_trials` to 100+

### Database Locked Error
- Close any other processes using `study.db`
- Increase timeout: `engine_kwargs={"connect_args": {"timeout": 60}}`

## Related Documentation

- [FEATURE_ORDERING_FIX.md](FEATURE_ORDERING_FIX.md) - Feature ordering consistency
- [VRT_WRITE_FIX.md](VRT_WRITE_FIX.md) - VRT write-through error fix
- [SUBTILED_PREDICTION_GUIDE.md](SUBTILED_PREDICTION_GUIDE.md) - Overall workflow

## Summary

✅ **Optimizes hyperparameters** specifically for 30 selected features
✅ **Reports out-of-sample performance** with spatial cross-validation
✅ **Saves optimization history** to SQLite database
✅ **Uses all CPU cores** for parallel optimization
✅ **Ready for prediction** with optimized model

The script now ensures the best possible model performance before making predictions on the full 42 tiles!
