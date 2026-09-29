# Project Description

In this project you will build, track, and serve a trip-duration model on the January 2025 NYC Yellow Taxi dataset.

## Main Task

Train and locally deploy a Machine Learning model that predicts the duration of New York City yellow taxi trips.

Use:

- `scikit-learn` to train a `RandomForestRegressor`,
- `MLflow` to track experiments locally,
- `FastAPI` to serve predictions through an API,
- a local request client such as `curl` or `requests` to validate the endpoint.

### Stretch Tasks

- Test your data pipeline, model, or API.
- Use `GridSearchCV` or `Optuna` for hyperparameter tuning.
- Compare the tuned model against the baseline and explain whether the extra complexity was worth it.

## Dataset

Yellow Taxi trip records: [NYC TLC trip record data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- Year: `2025`
- Month: `01`
- Target: `duration`

## Suggested Workflow

1. Load the January 2025 taxi data and inspect the columns that influence trip duration.
2. Create a clean training dataset and define the target variable.
3. Split the data into train and validation sets.
4. Train a baseline `RandomForestRegressor` model.
5. Evaluate the model with RMSE and log the run to MLflow.
6. Refactor feature preparation and model loading into modules you can reuse from both training code and the API.
7. Build a FastAPI app with a `/predict` endpoint.
8. Run the API locally and send a sample request.
9. Document your results and tradeoffs in the README.

## Suggested Repository Layout

This repo starts lightweight on purpose. As you implement the project, a practical beginner-friendly structure is:

- `notebooks/`: EDA, feature checks, and experiment notes.
- `src/`: Reusable data preparation, feature engineering, and training code.
- `app/`: FastAPI application and prediction schema.
- `tests/`: Data, model, and API tests.
- `artifacts/`: Saved model files or exports that should not be committed if they are large.

## Local-First Deployment Guidance

Choose **one** of these local deployment paths:

- Run the API directly with `uvicorn`.
- Package the API in Docker and run it locally.

Keep the deployment workflow easy to build, run and review on a single machine.

## Local Data Services

- `MLflow`: Run experiment tracking and inspect runs locally, for example at `http://127.0.0.1:5000`.
- `FastAPI`: Serve predictions locally, for example at `http://127.0.0.1:8000`.
- `Docker`: Optional, if you want a containerized local workflow.

## Answer the following questions

1. What is the RMSE of your final model?
2. What would you do differently if you had more time?
