import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from pydantic import BaseModel
from random import randint



app=FastAPI()
model= joblib.load("model.pkl")
scalar=joblib.load("scaler.pkl")

class Data(BaseModel):
        Age:float
        City:float
        Product_Category:float
        Unit_Price:float
        Quantity:float
        Discount_Amount:float
        Total_Amount:float
        Payment_Method:float
        Device_Type:float
        Session_Duration_Minutes:float
        Pages_Viewed:float
        Is_Returning_Customer:float
        Delivery_Time_Days:float
        Customer_Rating:float
        Gender_Male:float
        Gender_Female:float
        Gender_Other:float

@app.get("/")
def home():
        return{'message', "welcome to customer's behaviour dataset"}

@app.post("/predict_customers_behaviour")
def get_predicted_Returning_customer(input:Data):
    features=np.array([[
        input.Age,
        input.City,
        input.Product_Category,
        input.Unit_Price,
        input.Quantity,
        input.Discount_Amount,
        input.Total_Amount,
        input.Payment_Method,
        input.Device_Type,
        input.Session_Duration_Minutes,
        input.Pages_Viewed,
        input.Is_Returning_Customer,
        input.Delivery_Time_Days,
        input.Customer_Rating,
        input.Gender_Male,
        input.Gender_Female,
        input.Gender_Other,
    ]])

    x_scaled= scalar.fit_transform(features)
    y_prediction = model.predict(x_scaled)
    return (y_prediction[0])