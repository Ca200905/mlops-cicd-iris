from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# load model
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Iris ML API running"}

@app.post("/predict")
def predict(data: dict):

    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
