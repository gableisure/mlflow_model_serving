import warnings
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    iris_df['target'] = y
    train_df, test_df = train_test_split(iris_df, test_size=0.3, random_state=42, stratify=iris_df["target"])
    X_train = train_df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
    y_train = train_df["target"]
    X_test = test_df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
    y_test = test_df["target"]

    n_estimators = 100
    max_depth = 10
    random_state = 0

    EXPERIMENT_NAME = "iris_dataset_classification"

    mlflow_client = MlflowClient()

    experiment_details = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment_details is not None:
        experiment_id = experiment_details.experiment_id
    else:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=experiment_id, run_name="iris_dataset_api_rest") as run:
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_estimators", n_estimators)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        clf.fit(X_train, y_train)
        iris_predict_y = clf.predict(X_test)
        roc_auc_score_val = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
        mlflow.log_metric("test roc_auc_score", roc_auc_score_val)
        accuracy_score = accuracy_score(y_test, iris_predict_y)
        mlflow.log_metric("test accuracy_score", accuracy_score)
        mlflow.sklearn.log_model(clf, artifact_path="model")

    # Registra o modelo
    run_id = run.info.run_id
    model_uri = "runs:/{}/sklearn-model".format(run_id)
    mv = mlflow.register_model(model_uri, "RandomForestIrisModel")