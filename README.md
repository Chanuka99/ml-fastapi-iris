# FastAPI Iris ML API

## Project Overview
This project is a simple web API built using **FastAPI** that serves a machine learning model for **Iris Flower classification**. Users can send feature inputs (sepal length, sepal width, petal length, petal width) and receive predictions for the Iris species (Setosa, Versicolor, Virginica).

The project demonstrates:
- Machine learning model development and serialization
- API development with FastAPI
- Model inference through REST endpoints
- Input validation using Pydantic

---

## Dataset
We used the **Iris dataset** from scikit-learn:
- 150 samples
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: Setosa, Versicolor, Virginica

---

## Model
- Algorithm: Logistic Regression (you can also use RandomForestClassifier)
- Train/Test split: 80% train, 20% test
- Model saved using `joblib` as `model.pkl`

---

## API Endpoints
1. **GET /**  
   Health check endpoint.  
   **Response:**  
   ```json
   {
     "status": "healthy",
     "message": "ML Model API is running"
   }
