import warnings
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlflow.tracking import MlflowClient

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_path = ("wine-quality.csv")
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    EXPERIMENT_NAME = "wine_dataset_regression"

    mlflow_client = MlflowClient()

    experiment_details = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment_details is not None:
        experiment_id = experiment_details.experiment_id
    else:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(experiment_id=experiment_id, run_name="wine_elastic_net") as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, artifact_path="model")
        
    # Registra o modelo
    run_id = run.info.run_id
    model_uri = "runs:/{}/sklearn-model".format(run_id)
    mlflow.register_model(model_uri, "ElasticNetModel")