# Feature Selection and Column Name Mapping Fix

## Problem Description

The original code had a mismatch between the feature names saved for model predictions and the actual features the model expected:

1. **Before the fix**: Feature names were captured BEFORE the pipeline transformations (before one-hot encoding, outlier capping, etc.)
2. **The issue**: The feature selector (`SelectFromModel`) operates on features AFTER all transformations
3. **The mismatch**: When extracting selected features, the code tried to map the selector's mask to pre-transformation column names, but the selector worked on post-transformation column names (e.g., `num__koi_depth`, `cat__comment_str_CENTROID`)

## Solution

### 1. Added `get_transformed_feature_names()` function
**File**: `src/ml/data_prep/utils.py`

This function extracts the actual feature names after the ColumnTransformer preprocessing step but before feature selection. These are the names that the feature selector actually sees.

```python
def get_transformed_feature_names(pipeline, original_feature_names):
    """
    Get the feature names after preprocessing transformations (one-hot encoding, etc.)
    but before feature selection.
    """
```

### 2. Updated `extract_selected_features()` function
**File**: `src/ml/data_prep/utils.py`

Now uses transformed feature names instead of original names when mapping the feature selector's mask:

- Gets transformed feature names from the pipeline
- Maps the selector's boolean mask to these transformed names
- Returns the correct list of features the model expects

### 3. Updated `extract_feature_importance()` function
**File**: `src/ml/data_prep/utils.py`

Similarly updated to use transformed feature names for proper mapping.

### 4. Added `map_transformed_to_original_features()` function
**File**: `src/ml/data_prep/utils.py`

Creates a mapping from transformed feature names back to original feature names for better interpretability:

```python
{
  "num__koi_depth": {
    "original_feature": "koi_depth",
    "transformer_type": "num",
    "is_encoded": false
  },
  "cat__comment_str_CENTROID": {
    "original_feature": "comment_str_CENTROID",
    "transformer_type": "cat",
    "is_encoded": true
  }
}
```

### 5. Enhanced model saving
**File**: `src/ml/data_prep/models.py`

When saving selected features, the code now saves:

1. **Main file** (`{model}_selected_features.json`): Contains the exact transformed feature names and mapping
2. **Simplified file** (`{model}_original_features.json`): Contains just the original feature names for easier reading

## Files Modified

1. `src/ml/data_prep/utils.py`
   - Added: `get_transformed_feature_names()`
   - Added: `map_transformed_to_original_features()`
   - Updated: `extract_selected_features()`
   - Updated: `extract_feature_importance()`

2. `src/ml/data_prep/models.py`
   - Updated: `evaluate_models()` to save both transformed and original feature names
   - Added import: `map_transformed_to_original_features`

## How to Use

### For Training
No changes needed - the training process now automatically extracts and saves the correct feature names.

### For Prediction (router.py or inference)

**Before the fix**, you would load selected features like:
```python
# This was WRONG - using original column names
with open('selected_features.json') as f:
    features = json.load(f)['feature_names']
```

**After the fix**, the selected features JSON now contains:
```json
{
  "n_features": 30,
  "feature_names": [
    "num__koi_duration_err2",
    "num__koi_depth",
    "num__koi_ror",
    ...
  ],
  "feature_mapping": {
    "num__koi_duration_err2": {
      "original_feature": "koi_duration_err2",
      "transformer_type": "num",
      "is_encoded": false
    },
    ...
  },
  "note": "These are the exact transformed feature names required for prediction."
}
```

**For inference**, you need to:
1. Preprocess your input data using the SAME preprocessing pipeline
2. Use the preprocessed (transformed) data directly for prediction
3. The pipeline handles all the transformations internally

Example:
```python
# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    pipeline = model_data['trained_pipeline']

# Your raw input data
X_raw = pd.DataFrame({...})  # Original feature columns

# Preprocess using DataPreprocessor
preprocessor = DataPreprocessor(datatype=DATATYPE.KEPLER, data_dir='./data')
X_processed = preprocessor.prepare_data(X_raw)

# The pipeline handles everything from here
predictions = pipeline.predict(X_processed)
```

## Benefits

1. ✅ **Correct feature names**: Selected features now match what the model actually expects
2. ✅ **Better debugging**: Feature mapping shows the relationship between original and transformed features
3. ✅ **Easier interpretation**: Original features file provides a human-readable list
4. ✅ **Production-ready**: The pipeline handles all transformations internally
5. ✅ **Backward compatible**: Still saves feature information for understanding model behavior

## Testing

To verify the fix works:

1. Run training: `python launch.py`
2. Check the generated files in `models/{timestamp}/{dataset}/`:
   - `Logistic_Regression_selected_features.json` - Full details with mapping
   - `Logistic_Regression_original_features.json` - Simplified original names
3. Load a model and make predictions using the router

The feature names should now correctly represent the transformed features the model expects.
