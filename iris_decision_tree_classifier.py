import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn import tree

if __name__ == '__main__':
    iris = load_iris()
    sk_model = tree.DecisionTreeClassifier()
    sk_model = sk_model.fit(iris.data, iris.target)

    EXPERIMENT_NAME = "iris_dataset_classification"

    mlflow_client = MlflowClient()

    experiment_details = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment_details is not None:
        experiment_id = experiment_details.experiment_id
    else:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=experiment_id, run_name="iris_decision_tree_classifier") as run:
        mlflow.log_param("criterion", sk_model.criterion)
        mlflow.log_param("splitter", sk_model.splitter)
        mlflow.sklearn.log_model(sk_model, artifact_path="model")