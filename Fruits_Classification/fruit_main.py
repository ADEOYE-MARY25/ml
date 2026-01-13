# 

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MARYMAY FRUITS PAGE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class Data(BaseModel):
    size: float
    shape: int
    weight: float
    avg_price: float
    color: int
    taste: int

@app.get("/")
def home():
    return {"message": "welcome to fruit classification API"}

@app.get("/health")
def home():
    return {"message": "welcome to fruit classification API"}

@app.post("/predict_fruit_name")
def get_predicted_fruit_name(input: Data):

    # create feature array
    features = np.array([[
        input.size,
        input.shape,
        input.weight,
        input.avg_price,
        input.color,
        input.taste
    ]])

    # scale inputs
    features_scaled = scaler.transform(features)

    # prediction
    prediction = model.predict(features_scaled)

    return {"prediction": prediction[0]}

