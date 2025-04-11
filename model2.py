import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Iris Classification Experiments")

# Define different hyperparameter sets
hyperparameter_sets = [
    {
        "name": "Basic Logistic Regression",
        "params": {
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "C": 1.0
        },
        "description": "Default logistic regression with LBFGS solver",
        "tags": {
            "model_type": "logistic_regression",
            "dataset": "iris",
            "baseline": "true"
        }
    },
    {
        "name": "Regularized Logistic Regression",
        "params": {
            "solver": "lbfgs",
            "max_iter": 1500,
            "random_state": 42,
            "C": 0.1,  # Stronger regularization
            "penalty": "l2"
        },
        "description": "Logistic regression with stronger L2 regularization",
        "tags": {
            "model_type": "logistic_regression",
            "dataset": "iris",
            "regularized": "true"
        }
    },
    {
        "name": "High Iteration Logistic Regression",
        "params": {
            "solver": "saga",
            "max_iter": 3000,
            "random_state": 42,
            "C": 1.0,
            "penalty": "l1"  # L1 regularization
        },
        "description": "Logistic regression with L1 penalty and more iterations",
        "tags": {
            "model_type": "logistic_regression",
            "dataset": "iris",
            "high_iterations": "true"
        }
    }
]

# Run experiments for each hyperparameter set
for hp_set in hyperparameter_sets:
    with mlflow.start_run(run_name=hp_set["name"]):
        # Log description and tags
        mlflow.set_tag("mlflow.note.content", hp_set["description"])
        for tag_key, tag_value in hp_set["tags"].items():
            mlflow.set_tag(tag_key, tag_value)
        
        # Train model
        lr = LogisticRegression(**hp_set["params"])
        lr.fit(X_train, y_train)
        
        # Predict and calculate metrics
        y_pred = lr.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Log parameters and metrics
        mlflow.log_params(hp_set["params"])
        mlflow.log_metrics(metrics)
        
        # Log model
        signature = infer_signature(X_train, lr.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path=f"iris_model_{hp_set['name'].replace(' ', '_')}",
            signature=signature,
            input_example=X_train,
            registered_model_name=f"iris_model_{hp_set['name'].replace(' ', '_')}",
        )

        # Print results
        print(f"\nResults for {hp_set['name']}:")
        print(f"Parameters: {hp_set['params']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

# Load the last model for demonstration
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print("\nSample predictions from last model:")
print(result[:4])