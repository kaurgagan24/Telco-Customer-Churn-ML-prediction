# Telco Customer Churn Prediction

An end-to-end machine learning application that predicts customer churn
using an XGBoost classifier. The project covers the complete workflow:
dataset acquisition, exploratory data analysis, model development,
Python pipeline scripts, FastAPI inference API, Docker containerization,
an interactive HTML/CSS/JavaScript dashboard, and cloud deployment using
AWS.

------------------------------------------------------------------------
## Project overview

Customer churn occurs when customers stop using a company's products or
services. Predicting which customers are likely to leave can help a
business identify at-risk customers and plan retention activities.

This project uses customer demographic, service subscription, contract
and billing information to estimate churn probability.

### Objectives

-   Explore and understand customer churn data.
-   Clean and preprocess the dataset.
-   Perform exploratory data analysis (EDA).
-   Train and evaluate classification models.
-   Tune the XGBoost model and select a decision threshold.
-   Save the trained model and supporting metadata.
-   Serve predictions through a REST API built with FastAPI.
-   Build a dashboard for customer input, predictions and model/data
    analytics.
-   Package the application with Docker.
-   Deploy the container to AWS using Amazon ECR and Amazon ECS with
    Fargate.

### Technology stack

  Area                          Tools
  ----------------------------- -------------------------------------
  Language                      Python
  Data analysis                 Pandas, NumPy
  Visualization                 Matplotlib, Seaborn
  Machine learning              Scikit-learn, XGBoost
  Hyperparameter optimization   Optuna
  Experiment tracking           MLflow
  Model serialization           Joblib
  API                           FastAPI, Uvicorn, Pydantic
  Frontend                      HTML, CSS, JavaScript
  API communication             JavaScript Fetch API
  Containerization              Docker
  Cloud image registry          Amazon ECR
  Cloud deployment              Amazon ECS, AWS Fargate
  Cloud region used/planned     Asia Pacific (Mumbai), `ap-south-1`

------------------------------------------------------------------------

## Application workflow

``` text
Kaggle dataset
      |
      v
Data loading and understanding
      |
      v
Exploratory Data Analysis (EDA)
      |
      v
Data cleaning and preprocessing
      |
      v
Feature engineering and train/test split
      |
      v
Model training and evaluation
      |
      v
XGBoost + Optuna hyperparameter tuning
      |
      v
Threshold selection and final evaluation
      |
      v
Save model and metadata in artifacts/
      |
      v
Refactor notebook workflow into Python scripts
      |
      v
FastAPI loads model and exposes /predict
      |
      v
HTML + CSS + JavaScript dashboard
      |
      v
Docker image
      |
      v
Amazon ECR
      |
      v
Amazon ECS on AWS Fargate
      |
      v
Public dashboard URL
```

------------------------------------------------------------------------

## Repository structure

The project contains the notebook, reusable Python code, model
artifacts, API, frontend and Docker configuration.

``` text
Telco Customer Churn ML prediction/
│
├── app/
│   ├── main.py                  # FastAPI application and API endpoints                   # 
│   └── frontend/
│       ├── index.html           # Prediction dashboard
│       ├── style.css            # Dashboard styling
│       └── script.js            # Form handling and API requests
│
├── artifacts/
│   ├── xgb_model.joblib         # Trained XGBoost model
│   ├── feature_columns.json     # Expected model feature columns
│   ├── metrics.json             # Saved evaluation metrics
│   ├── best_params.json         # Best hyperparameters
│   ├── threshold_sweep.csv      # Metrics at different thresholds
│   └── train_info.json          # Training metadata
│
├── configs/
│   └── config.yaml              # Project and model configuration
│
├── data/
│   └── ...                      # Dataset files (not necessarily committed)
│
├── notebooks/
│   └── customer_churn_end_to_end_pipeline.ipynb               # EDA and model development notebook
|   └── mlruns/                      # Local MLflow experiment data (if retained)
│
├── src/
│   ├── config.py                # Configuration and path handling
│   ├── preprocess.py            # Input preprocessing
│   ├── train.py                 # Training/model-loading functions
│   └── ...                      # Other reusable project modules
│
├── tests/                       # Contain Test file
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

------------------------------------------------------------------------

## Dataset

The project uses a Telco Customer Churn dataset obtained from Kaggle.

**Dataset source:**(https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information such as:

-   Demographic details: gender, senior citizen status, partner and
    dependents.
-   Services: phone service, internet service, online security, online
    backup, device protection, tech support and streaming services.
-   Account details: tenure, contract type, paperless billing and
    payment method.
-   Billing details: monthly charges and total charges.
-   Target: `Churn`.

------------------------------------------------------------------------

## Exploratory data analysis

The notebook is the initial development environment for understanding
the dataset and testing ideas.

Typical EDA tasks include:

1.  Load the CSV using Pandas.
2.  Inspect rows, columns, data types and summary statistics.
3.  Check missing values and duplicate records.
4.  Inspect the target distribution and class imbalance.
5.  Explore numerical features such as tenure, MonthlyCharges and
    TotalCharges.
6.  Explore categorical features such as Contract, InternetService and
    PaymentMethod.
7.  Use charts and grouped statistics to investigate relationships with
    churn.
8.  Identify preprocessing requirements before model training.

Example commands:

``` python
import pandas as pd

df = pd.read_csv("'../data/raw/Telco-Customer-Churn_unprocessed.csv'")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["Churn"].value_counts(dropna=False))
```


------------------------------------------------------------------------

## Model development

The classification workflow is developed and tested in the Jupyter
notebook before being moved into reusable Python modules.

The project uses XGBoost for the final churn classifier and Optuna for
hyperparameter optimization.

The workflow includes:

1.  Separate the target column from the input features.
2.  Apply the same cleaning and preprocessing logic consistently.
3.  Split the data into training and holdout/test sets.
4.  Train candidate classification models as implemented in the
    notebook.
5.  Evaluate models using suitable classification metrics.
6.  Tune XGBoost hyperparameters with Optuna.
7.  Evaluate the selected model.
8.  Examine different classification thresholds.
9.  Save the selected model, feature-column information and evaluation
    metadata.

### Why use a decision threshold?

A binary classifier produces a probability. A threshold converts that
probability into a class prediction.

For example, with a threshold of `0.30`:

``` text
If churn_probability >= 0.30:
    predict Churn = Yes
Else:
    predict Churn = No
```

The threshold is a model decision setting, not the same thing as the
probability itself. The threshold should be selected based on validation
results and the business trade-off between false positives and false
negatives.

The project dashboard currently reports an evaluation threshold of
`0.3`. Confirm that the API uses the same configured threshold.

------------------------------------------------------------------------

## Moving from notebook to Python scripts

Once the notebook workflow is stable, reusable logic is moved into the
`src/` package.

The purpose of this step is to separate experimentation from application
code.

A typical separation is:

-   `src/preprocess.py`: reusable data cleaning, transformation and
    inference preprocessing.
-   `src/train.py`: model training, evaluation, saving and loading
    functions.
-   `src/config.py`: reading configuration and resolving project paths.
-   `configs/config.yaml`: configuration values such as paths and model
    settings.

------------------------------------------------------------------------

## Model artifacts

The final model and supporting files are stored in `artifacts/`.

  Artifact                 Purpose
  ------------------------ -------------------------------------------
  `xgb_model.joblib`       Serialized trained model
  `feature_columns.json`   Expected input feature names and ordering
  `metrics.json`           Saved model evaluation metrics
  `best_params.json`       Selected hyperparameters
  `threshold_sweep.csv`    Evaluation results across thresholds
  `train_info.json`        Training metadata

The API loads the model and feature metadata when the FastAPI
application starts. It should not retrain the model for every prediction
request.

------------------------------------------------------------------------

## FastAPI application

FastAPI provides the REST API used by the frontend.

The main application is located at:

``` text
app/main.py
```

The application uses a Pydantic request model to validate customer
details. It loads the configuration, trained model and feature columns
during application startup.

### Prediction flow

``` text
Customer fills dashboard form
          |
          v
JavaScript collects form values
          |
          v
POST /predict with JSON body
          |
          v
FastAPI validates request using Pydantic
          |
          v
Preprocessing transforms input
          |
          v
Loaded XGBoost model predicts probability
          |
          v
Threshold converts probability to class
          |
          v
API returns prediction response
          |
          v
JavaScript displays result
```

### Run FastAPI locally

Open PowerShell in the project root and activate the virtual
environment:

``` powershell
.venv\Scripts\Activate
```

Run the API:

``` powershell
uvicorn app.main:app --reload --port 8000
```

Open:

-   API documentation: `http://127.0.0.1:8000/docs`
-   Health check: `http://127.0.0.1:8000/health`
-   Dashboard: use the route configured in `app/main.py` (currently the
    static frontend is mounted at `/dashboard` in the observed
    application configuration).

If the frontend is mounted with `StaticFiles(..., html=True)` at
`/dashboard`, open:

``` text
http://127.0.0.1:8000/dashboard/
```

The exact dashboard URL depends on the route and static-file
configuration in `main.py`.

------------------------------------------------------------------------

## Frontend dashboard

The frontend is built with plain HTML, CSS and JavaScript.

### Prediction page

The prediction page collects the customer information expected by the
API, including:

-   Customer profile.
-   Subscribed services.
-   Account and billing information.

When the user clicks the prediction button, `script.js` sends the
entered details to the existing `/predict` endpoint using the Fetch API.

The dashboard displays the API response, such as:

-   Churn prediction.
-   Estimated churn probability.
-   Decision threshold.

### Analytics page/section

The analytics dashboard is intended to present model and dataset
information, including:

-   Model performance metrics.
-   Customer and churn distribution.
-   Contract distribution.
-   Other dataset-level patterns.
-   Threshold analysis.

Analytics must be based on saved evaluation files or actual aggregated
dataset statistics. Do not use hardcoded sample values and present them
as real project results.

### Frontend/API integration

For a same-origin deployment, the frontend can call the API using a
relative URL:

``` javascript
const response = await fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(customerData)
});

const result = await response.json();
```

The actual `customerData` object must contain the exact field names and
data types expected by the Pydantic model in `app/main.py`.

------------------------------------------------------------------------

## Run locally

### 1. Clone the repository

``` powershell
git clone <YOUR_GITHUB_REPOSITO>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create and activate a virtual environment

``` powershell
python -m venv .venv
.venv\Scripts\Activate
```

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use dependency versions compatible with the environment used to train
and serialize the model.

### 4. Place the dataset and model artifacts

Ensure the required dataset, configuration and model artifacts exist in
their expected locations. The inference API requires the saved model and
feature-column metadata.

### 5. Start FastAPI

``` powershell
uvicorn app.main:app --reload --port 8000
```

### 6. Open the application

Open the dashboard route configured in FastAPI, then use `/docs` to
inspect and test the API independently.

------------------------------------------------------------------------

## Docker

Docker packages the API, frontend files, configuration and trained model
into a portable image.

The project includes:

-   `Dockerfile`: defines the application image.
-   `.dockerignore`: excludes unnecessary files from the build context.
-   `requirements-docker.txt`: inference dependencies used by the Docker
    image.

### Build the image

Run from the project root:

``` powershell
docker build -t telco-churn-api:v1 .
```

### Run the container

``` powershell
docker run -d --name telco-churn-container -p 8000:8000 telco-churn-api:v1
```

### Check the running container

``` powershell
docker ps
```

### View logs

``` powershell
docker logs telco-churn-container
```

### Test the application

-   Dashboard: `http://localhost:8000/dashboard/` (if configured at
    `/dashboard`).
-   API docs: `http://localhost:8000/docs`.
-   Health: `http://localhost:8000/health`.

### Stop and remove the container

``` powershell
docker stop telco-churn-container
docker rm telco-churn-container
```

### Rebuild after changing code

Stop and remove the old container, rebuild the image and run a new
container:

``` powershell
docker stop telco-churn-container
docker rm telco-churn-container
docker build -t telco-churn-api:v2 .
docker run -d --name telco-churn-container -p 8000:8000 telco-churn-api:v2
```

If the container name does not exist, Docker may report an error for the
stop/remove command; check `docker ps -a` before proceeding.

------------------------------------------------------------------------

## AWS deployment: ECR + ECS Fargate

The planned AWS deployment uses:

-   **Amazon ECR** to store the Docker image.
-   **Amazon ECS** to manage the container.
-   **AWS Fargate** to run the container without managing EC2 servers.

The example below uses the Mumbai region, `ap-south-1`.

> **Deployment status:** Treat this section as a deployment guide until
> the AWS service has actually been created and tested. Add the real
> public URL and deployment date after successful deployment.

### 1. Prepare AWS

1.  Create/sign in to an AWS account.
2.  Enable MFA.
3.  Create an IAM identity with only the permissions required for ECR
    and ECS deployment.
4.  Install AWS CLI v2.
5.  Configure the CLI using the IAM credentials.

``` powershell
aws configure
```

Enter the region:

``` text
ap-south-1
```

Verify the identity:

``` powershell
aws sts get-caller-identity
```

Do not use root-account access keys or publish AWS credentials.

### 2. Create an ECR repository

Create a private ECR repository named `telco-churn-api` in the Mumbai
region using the AWS Console, or run:

``` powershell
aws ecr create-repository `
  --repository-name telco-churn-api `
  --region ap-south-1
```

Retrieve your AWS account ID:

``` powershell
$AWS_ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$AWS_REGION = "ap-south-1"
$ECR_URI = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/telco-churn-api"
```

### 3. Authenticate Docker with ECR

``` powershell
aws ecr get-login-password --region $AWS_REGION |
  docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
```

### 4. Build, tag and push the image

Build the latest application image from the project root:

``` powershell
docker build -t telco-churn-api:v1 .
```

Tag it for ECR:

``` powershell
docker tag telco-churn-api:v1 "${ECR_URI}:v1"
```

Push it:

``` powershell
docker push "${ECR_URI}:v1"
```

Confirm that the image appears in the ECR repository in the AWS Console.

### 5. Create an ECS cluster

In the AWS Console:

1.  Open Amazon ECS.
2.  Create a cluster.
3.  Give it a name such as `telco-churn-cluster`.
4.  Use AWS Fargate as the compute option when creating the service.

### 6. Create a Fargate task definition

Create a task definition with settings appropriate for the application:

-   Launch type/compute: AWS Fargate.
-   Operating system: Linux.
-   Architecture: match the Docker image architecture (for example,
    X86_64).
-   CPU and memory: start with a small supported combination and
    increase only if required.
-   Container image: the ECR image URI ending in `:v1`.
-   Container port: `8000`.
-   Protocol: TCP.
-   Container name: for example, `telco-churn-api`.

Use an ECS task execution role so ECS can pull the private ECR image and
write logs. Do not put AWS access keys inside the container image.

### 7. Create an ECS service

Create a service in the cluster:

-   Compute: Fargate.
-   Task definition: the one created above.
-   Desired tasks: 1 for a basic demonstration.
-   Networking: select a VPC and public subnets.
-   Public IP: enable assignment if using the simple direct-public-IP
    demonstration setup.
-   Security group: allow inbound TCP port `8000` only from the IP
    ranges that need access.

For a temporary demonstration, a public endpoint may be used, but an
internet-facing production application should use HTTPS, authentication
and an appropriately configured load balancer or other secure ingress.
Do not leave an unauthenticated API open to the entire internet with
unrestricted access.

### 8. Test the deployment

After the ECS task is running, find its public IP or the load balancer
DNS name, depending on the networking setup.

Test:

``` text
http://<PUBLIC_HOST>:8000/health
http://<PUBLIC_HOST>:8000/docs
```

Open the dashboard using the frontend route configured in FastAPI, for
example:

``` text
http://<PUBLIC_HOST>:8000/dashboard/
```

Test the prediction form and analytics page. Confirm the frontend uses
relative API paths so it calls the deployed backend rather than
`localhost`.

### 9. Monitor and clean up

-   Check ECS service events and task status.
-   Review container logs in CloudWatch if logging is configured.
-   Monitor AWS billing.
-   Stop or delete the ECS service and other billable resources when the
    demonstration is complete.
-   Delete unused ECR images/repositories if they are no longer needed.

A running Fargate task, public IPv4 address, load balancer and data
transfer can incur charges. Review the current AWS pricing for your
region before leaving resources running.

------------------------------------------------------------------------

## API endpoints

  -----------------------------------------------------------------------
  Method                  Endpoint                Purpose
  ----------------------- ----------------------- -----------------------
  GET                     `/health`               Checks API/model health

  POST                    `/predict`              Receives customer data
                                                  and returns churn
                                                  prediction

  GET                     `/docs`                 Interactive FastAPI
                                                  documentation

  GET                     `/dashboard/`           Frontend, if mounted at
                                                  `/dashboard`
  -----------------------------------------------------------------------

The actual routes are defined in `app/main.py`. If the frontend is
mounted at a different path, update this table accordingly.

### Example prediction request

Use the exact Pydantic schema from `app/main.py` when testing. The
following is only a structural illustration; replace or adjust fields
and accepted values to match the actual model:

``` json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 906.0
}
```

The actual response keys and values should be read from the
implementation of `/predict` in `app/main.py`.

------------------------------------------------------------------------

## Evaluation results

The current dashboard screenshot displays the following model evaluation
values:

  Metric                   Displayed value
  ---------------------- -----------------
  Accuracy                           71.8%
  Precision                          48.1%
  Recall                             81.3%
  F1-score                           60.4%
  ROC-AUC                            0.825
  Evaluation threshold                0.30
  Holdout size             1,409 customers

These values should be checked against the current
`artifacts/metrics.json` before publishing this README. The dashboard
indicates that accuracy was derived from saved precision and recall on
the configured holdout; verify the exact calculation and source metrics
before describing it as a directly measured metric.

### Interpreting the metrics

-   **Accuracy:** proportion of all predictions that are correct.
-   **Precision:** proportion of predicted churn cases that are actual
    churn cases.
-   **Recall:** proportion of actual churn cases that the model
    identifies.
-   **F1-score:** harmonic mean of precision and recall.
-   **ROC-AUC:** measures how well the model ranks positive cases above
    negative cases across classification thresholds.

The selected threshold affects the predicted class and the
precision/recall trade-off.

------------------------------------------------------------------------

## Known limitations and next steps

-   **Prediction consistency:** During local dashboard testing,
    different customer inputs appeared to return the same probability
    (`73.61%`) and churn label. This needs investigation before claiming
    that individual customer predictions are reliable. Test `/predict`
    independently through `/docs`, compare request payloads and inspect
    preprocessing and feature-column alignment.
-   **Analytics provenance:** Ensure every dashboard metric and chart is
    loaded from actual artifacts or calculated from the dataset. Do not
    present placeholder values as real results.
-   **Explainability:** Add SHAP or another explainability method only
    after validating that explanations are correctly computed for the
    final model and its preprocessing pipeline.
-   **Public API security:** Add HTTPS, authentication/rate limiting as
    appropriate, restrictive network access and safe logging before
    exposing the application to real customer data.
-   **Dependency consistency:** Pin and test compatible Python, XGBoost,
    scikit-learn and Joblib versions across local training, inference
    and Docker.
-   **AWS status:** Add the deployed URL only after ECS reports a
    healthy running task and the public dashboard and API have been
    tested.

------------------------------------------------------------------------

## Future improvements

-   Investigate and fix the repeated prediction probability.
-   Add automated tests for preprocessing and prediction.
-   Add SHAP-based global and individual explanations.
-   Improve model monitoring and logging.
-   Add authentication and HTTPS.
-   Add CI/CD for automated Docker builds and deployment.
-   Track model versions and experiments more systematically.
-   Add drift monitoring and a retraining workflow.

------------------------------------------------------------------------

## Author

**\[Gagandeep Kaur\]**

Add your GitHub profile and LinkedIn profile here:

-   GitHub: (https://github.com/kaurgagan24)
-   LinkedIn:(https://www.linkedin.com/in/gagandeep-kaur-cs)

------------------------------------------------------------------------

## Acknowledgements

-   Kaggle dataset contributor: BlastChar
-   Open-source libraries and tools used in the project: Pandas, NumPy,
    Scikit-learn, XGBoost, Optuna, MLflow, FastAPI and Docker.
