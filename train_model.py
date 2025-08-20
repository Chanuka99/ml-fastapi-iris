# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

def main():
    # 1) Load dataset
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = list(data.target_names)

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Build a simple pipeline (scaler + logistic regression)
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # 4) Train
    pipe.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 6) Save everything we need for inference
    artifact = {
        "pipeline": pipe,
        "feature_names": feature_names,         # order matters!
        "target_names": target_names,           # index -> class label
        "metrics": {"accuracy": acc},
        "model_type": "LogisticRegression",
        "problem_type": "classification",
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }
    joblib.dump(artifact, "model.pkl")
    print(f"Saved model.pkl with test accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
