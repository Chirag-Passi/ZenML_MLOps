from zenml.steps import step
from zenml.pipelines import pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib
from typing_extensions import Annotated
from zenml.logger import get_logger
from zenml.enums import ArtifactType
from zenml import Model
from zenml.integrations.pandas.materializers.pandas_materializer import (
    PandasMaterializer,
)
import warnings

warnings.filterwarnings("ignore")


# Optimized data loading step
@step(output_materializers={"X": PandasMaterializer, "y": PandasMaterializer})
def load_data() -> tuple[Annotated[pd.DataFrame, "X"], Annotated[pd.Series, "y"]]:
    """Load the dataset and return the feature matrix X and target vector y as Pandas DataFrames."""
    data = pd.read_csv("./data/diabetes-dev-1.csv")
    X = data.drop(columns=["Diabetic", "PatientID"])  # Drop unnecessary columns
    y = data["Diabetic"]  # Target variable
    return X, y


# Data splitting step
@step(
    output_materializers={
        "X_train": PandasMaterializer,
        "X_test": PandasMaterializer,
        "y_train": PandasMaterializer,
        "y_test": PandasMaterializer,
    }
)
def split_data_train_test(X: pd.DataFrame, y: pd.Series) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Split the dataset into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test


# Preprocessing step
@step(
    output_materializers={
        "X_train_scaled": PandasMaterializer,
        "X_test_scaled": PandasMaterializer,
    }
)
def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, "X_train_scaled"],
    Annotated[pd.DataFrame, "X_test_scaled"],
]:
    """Preprocess the data by scaling."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled


# Model creation step
@step
def create_model() -> RandomForestClassifier:
    """Create a RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model


# Model training step
@step
def train_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train the model on the training set."""
    model.fit(X_train, y_train)
    return model


# Model fine-tuning step
@step
def fine_tune_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Fine-tune the model with hyperparameter optimization."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, 30],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


from zenml.environment import Environment


# Model evaluation step
@step
def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate the model performance on the test set."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


# Metrics storage step
@step
def store_metrics(metrics: dict):
    """Store the model metrics."""
    logger = get_logger(__name__)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")


# Model storage step
@step
def store_model(
    model: RandomForestClassifier,
) -> Annotated[RandomForestClassifier, ArtifactType.MODEL]:
    """Save the trained model to disk and ZenML will store it as an artifact."""
    model_path = "random_forest_model.pkl"
    joblib.dump(model, model_path)
    return model


# Define the pipeline
@pipeline(
    enable_cache=False,
    model=Model(
        name="Ml Ops",
        license="Zenon.ai",
        description="ML Ops Working Model for Zenon.",
        metadata={
            "framework": "scikit-learn",
            "type": "RandomForestClassifier",
        },
        tags=["production", "diabetes-prediction"],
    ),
)
def ml_ops_pipeline():
    """Training pipeline that includes data loading, training, fine-tuning, and storing results."""
    # Load and process the data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data_train_test(X, y)
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # Create, train, fine-tune the model
    model = create_model()
    model = train_model(model, X_train_scaled, y_train)
    model = fine_tune_model(model, X_train_scaled, y_train)

    # Evaluate and store metrics and model
    metrics = evaluate_model(model, X_test_scaled, y_test)
    store_metrics(metrics)
    store_model(model)


# Run the pipeline
def run_pipeline():
    """Define and run the entire pipeline."""
    training_pipeline_instance = ml_ops_pipeline()


# Execute the pipeline
if __name__ == "__main__":
    run_pipeline()
