# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Dict, Any
import numpy as np
import joblib

app = FastAPI(
    title="Iris ML Model API",
    description="FastAPI service for Iris species prediction",
    version="1.0.0"
)

# ------- Load model on startup -------
ARTIFACT = {}  # will hold pipeline, feature_names, target_names, metrics, etc.

@app.on_event("startup")
def load_artifact():
    global ARTIFACT
    try:
        ARTIFACT = joblib.load("model.pkl")
        # quick sanity checks
        assert "pipeline" in ARTIFACT and "feature_names" in ARTIFACT
        assert "target_names" in ARTIFACT
        print("Model loaded:", ARTIFACT.get("model_type"))
    except Exception as e:
        print("FATAL: could not load model.pkl:", e)
        raise

# ------- Pydantic schemas (input/output) -------
class PredictionInput(BaseModel):
    # Iris expects 4 features in this exact order:
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length (cm)")
    sepal_width:  float = Field(..., ge=0, le=10, description="Sepal width (cm)")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length (cm)")
    petal_width:  float = Field(..., ge=0, le=10, description="Petal width (cm)")

class BatchPredictionInput(BaseModel):
    from pydantic import conlist

class BatchPredictionInput(BaseModel):
    items: conlist(PredictionInput, min_length=1)


class PredictionOutput(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None

# ------- Endpoints -------
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": ARTIFACT.get("model_type"),
        "problem_type": ARTIFACT.get("problem_type"),
        "trained_at": ARTIFACT.get("trained_at"),
        "test_accuracy": ARTIFACT.get("metrics", {}).get("accuracy"),
        "features": ARTIFACT.get("feature_names"),
        "classes": ARTIFACT.get("target_names"),
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to [ [features...] ] in the correct order
        ordered = [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]
        X = np.array(ordered, dtype=float).reshape(1, -1)

        pipe = ARTIFACT["pipeline"]
        class_names = ARTIFACT["target_names"]

        pred_idx = int(pipe.predict(X)[0])

        # If model supports probabilities, return confidence
        proba = None
        confidence = None
        if hasattr(pipe, "predict_proba"):
            proba_arr = pipe.predict_proba(X)[0]  # shape (n_classes,)
            proba = {class_names[i]: float(p) for i, p in enumerate(proba_arr)}
            confidence = float(np.max(proba_arr))

        return PredictionOutput(
            predicted_class=class_names[pred_idx],
            predicted_index=pred_idx,
            confidence=confidence,
            probabilities=proba
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Bonus: batch predictions
@app.post("/predict-batch", response_model=List[PredictionOutput])
def predict_batch(batch: BatchPredictionInput):
    try:
        pipe = ARTIFACT["pipeline"]
        class_names = ARTIFACT["target_names"]

        rows = []
        for item in batch.items:
            rows.append([
                item.sepal_length, item.sepal_width,
                item.petal_length, item.petal_width
            ])
        X = np.array(rows, dtype=float)

        preds = pipe.predict(X).astype(int)

        probas = None
        if hasattr(pipe, "predict_proba"):
            probas = pipe.predict_proba(X)  # shape (n, n_classes)

        outputs = []
        for i, pred_idx in enumerate(preds):
            confidence = None
            proba_dict = None
            if probas is not None:
                p = probas[i]
                proba_dict = {class_names[j]: float(p[j]) for j in range(len(class_names))}
                confidence = float(np.max(p))
            outputs.append(PredictionOutput(
                predicted_class=class_names[pred_idx],
                predicted_index=int(pred_idx),
                confidence=confidence,
                probabilities=proba_dict
            ))
        return outputs

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
