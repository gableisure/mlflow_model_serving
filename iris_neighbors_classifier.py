# Importações de bibliotecas
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)
    
    test_size = 0.26
    n_neighbors = 9

    iris_data = load_iris()
    df_iris = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
    df_iris['class'] = iris_data.target

    x = df_iris.iloc[:, :-1].values
    sample_size = len(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, df_iris['class'], test_size = test_size)

    scaler = StandardScaler()  
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)  
    x_test = scaler.transform(x_test)
    
    classifier = KNeighborsClassifier(n_neighbors = n_neighbors)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    print(y_pred)
    report = classification_report(y_test, y_pred, output_dict = True)
    
    EXPERIMENT_NAME = "iris_dataset_classification"
    
    mlflow_client = MlflowClient()

    experiment_details = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment_details is not None:
        experiment_id = experiment_details.experiment_id
    else:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=experiment_id, run_name="iris_neighbors_classifier") as run:
        mlflow.log_param('n_neighbors', n_neighbors)
        mlflow.log_metric('test_size', test_size)
        mlflow.log_metric('sample_size', sample_size)
        mlflow.log_metric('accuracy', report['accuracy'])