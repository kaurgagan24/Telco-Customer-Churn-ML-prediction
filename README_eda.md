# Telco churn: cleaned and model-ready data

## Data preparation
- Whitespace stripped; blank strings recognized as missing.
- TotalCharges converted to numeric. The 11 missing values all occur at tenure=0.
  They are set to 0 under the explicit assumption that no charges have accrued yet.
  TotalChargesWasMissing preserves the original missingness.
- Churn: No=0, Yes=1. Binary service/household flags similarly mapped to 0/1.
- No duplicate rows/IDs or inconsistent phone/internet-service records were found.
- All 7,043 customers retained. No arbitrary outlier deletion or winsorization.
- customerID is retained for traceability but excluded from all model matrices.
- 'No internet service' and 'No phone service' remain distinct from 'No'.

## Engineered features
| Feature | Definition |
|---|---|
| TotalChargesWasMissing | Original total-charge value was missing |
| IsNewCustomer | tenure equals zero |
| AutomaticPayment | Automatic bank transfer or credit card |
| InternetAddonCount | Number of six internet add-ons marked Yes |
| SupportServiceCount | OnlineSecurity and TechSupport marked Yes |
| StreamingServiceCount | StreamingTV and StreamingMovies marked Yes |
| TotalServiceCount | Internet add-ons + phone service + internet service + multiple-line option |
| ContractMonths | Month-to-month=1, one year=12, two year=24; not remaining contract time |
| LogTenure | log(1+tenure) |
| LogTotalCharges | log(1+TotalCharges) |

These are candidate predictors, not demonstrated improvements. Some are redundant
with original features. Compare performance with/without them using training-only
cross-validation. No target-based aggregates or target encoding were used.

## Outputs
- cleaned_customers.csv: human-readable cleaned data, ID and target included.
- engineered_customers.csv: cleaned data plus features; still contains categorical
  columns, customerID and target. Do not pass the whole table directly to a model.
- X_train_raw.csv / X_test_raw.csv: original predictors, no ID/target.
- y_train.csv / y_test.csv: aligned labels; maintain row order.
- X_train_ready.csv / X_test_ready.csv: numeric matrices, aligned with labels.
- split_manifest.csv: customer IDs and train/test assignment.
- fitted_preprocessor.joblib: fitted training-only preprocessing pipeline.
- quality_report.json: checks, sizes and class balance.

## Split and transformations
80% training / 20% testing, stratified on Churn, random_state=42.
Imputation, numeric standardization, and categorical encoding are fitted only on
training rows. Binary indicators remain 0/1. All categorical levels are one-hot
encoded; unknown future levels are ignored rather than raising an error.
Median numeric and most-frequent categorical imputers are safeguards for future
missing values; the supplied cleaned data has no remaining missing values.

## Train a chosen algorithm (recommended for tuning)
Keep churn_preprocessing.py importable alongside this example.

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score
from churn_preprocessing import make_preprocessor

X_train = pd.read_csv('churn_prepared/X_train_raw.csv')
X_test = pd.read_csv('churn_prepared/X_test_raw.csv')
y_train = pd.read_csv('churn_prepared/y_train.csv')['Churn']
y_test = pd.read_csv('churn_prepared/y_test.csv')['Churn']

# Replace this classifier with the desired compatible supervised algorithm.
model = Pipeline([
    ('preprocess', make_preprocessor()),
    ('classifier', LogisticRegression(max_iter=2000))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_train, y_train, cv=cv,
                        scoring=['roc_auc', 'average_precision', 'balanced_accuracy'])
print({k: v.mean() for k, v in scores.items() if k.startswith('test_')})
model.fit(X_train, y_train)
# Evaluate once after model/threshold selection is complete:
p = model.predict_proba(X_test)[:, 1]
print('Test ROC-AUC:', roc_auc_score(y_test, p))
print('Test average precision:', average_precision_score(y_test, p))
```

The ready matrices are convenient for a fixed train/test experiment. For
cross-validation or hyperparameter tuning, use raw splits and the full pipeline
above so preprocessing is refitted within every fold. Do not cross-validate on
the already scaled training matrix if strict fold isolation is required.
For tree-based models, make_preprocessor(scale_numeric=False) skips scaling.
Full dummy encoding is convenient for regularized predictive models; remove a
reference level and redundant features if fitting an unregularized explanatory
regression that requires a full-rank design matrix.

## Evaluation and limitations
Churn is the minority class (~26.5%). No oversampling was performed. Compare
class weights or resampling within training folds, never on the test set.
Report ROC-AUC, average precision, precision, recall, F1 and a confusion matrix;
choose any decision threshold using training validation data, not test labels.
No classifier has been trained or its accuracy claimed in this preparation task.

This is a snapshot dataset without a prediction timestamp or churn horizon.
Random stratification is not evidence of prospective/temporal performance.
Confirm predictor availability before churn and use temporal evaluation when
timestamps become available. Gender and SeniorCitizen may need exclusion or
group-level fairness evaluation depending on intended use.

Tested environment: Python with pandas, numpy, scikit-learn 1.6.1 and joblib.
Serialized pipelines should be loaded in a matching trusted environment.
