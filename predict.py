import mlflow
import pandas as pd
logged_model = 'runs:/85cc21fde29445da961cd2b70dac6dec/model'

# Load model
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
res = loaded_model.predict(pd.DataFrame({
    "sepal length (cm)": 6.7,
    "sepal width (cm)": 3.0,
    "petal length (cm)": 5.2,
    "petal width (cm)": 2.3,
}, index=[0]))

print(len(res))