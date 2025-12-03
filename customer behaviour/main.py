# import numpy as np
# import pandas as pd
# import joblib
# from fastapi import FastAPI
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
# from pydantic import BaseModel
# from random import randint



# app=FastAPI()
# model= joblib.load("model.pkl")
# scalar=joblib.load("scaler.pkl")

# class Data(BaseModel):
#     Age: float
#     City: float
#     Product_Category: float
#     Unit_Price: float
#     Quantity: float
#     Discount_Amount: float
#     Total_Amount: float
#     Payment_Method: float
#     Device_Type: float
#     Session_Duration_Minutes: float
#     Pages_Viewed: float
#     Delivery_Time_Days: float
#     Customer_Rating: float
#     Gender_Male: float
#     Gender_Female: float
#     Gender_Other: float

# num_col= ['Age', 'Unit_Price', 'Quantity', 'Discount_Amount', 'Total_Amount',
#        'Session_Duration_Minutes', 'Pages_Viewed', 'Delivery_Time_Days',
#        'Customer_Rating']

# @app.get("/")
# def home():
#         return{'message', "welcome to customer's behaviour dataset"}

# @app.post("/predict_customers_behaviour")
# def get_predicted_Returning_customer(input:Data):
#     features=np.array([[
#         input.Age,
#         input.City,
#         input.Product_Category,
#         input.Unit_Price,
#         input.Quantity,
#         input.Discount_Amount,
#         input.Total_Amount,
#         input.Payment_Method,
#         input.Device_Type,
#         input.Session_Duration_Minutes,
#         input.Pages_Viewed,
#         input.Delivery_Time_Days,
#         input.Customer_Rating,
#         input.Gender_Male,
#         input.Gender_Female,
#         input.Gender_Other
#     ]])

# #     df= pd.DataFrame([input.dict()], columns=num_col)

# #     x_scaled= scalar.transform(df)
#     y_prediction = model.predict(features)
#     return (y_prediction[0])


import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class Data(BaseModel):
    Age: float
    City: float
    Product_Category: float
    Unit_Price: float
    Quantity: float
    Discount_Amount: float
    Total_Amount: float
    Payment_Method: float
    Device_Type: float
    Session_Duration_Minutes: float
    Pages_Viewed: float
    Delivery_Time_Days: float
    Customer_Rating: float
    Gender_Male: float
    Gender_Female: float
    Gender_Other: float

@app.get("/")
def home():
    return {"message": "welcome to customer's behaviour dataset"}

@app.post("/predict_customers_behaviour")
def get_predicted_returning_customer(input: Data):

    columns = [
        "Age", "City", "Product_Category", "Unit_Price", "Quantity",
        "Discount_Amount", "Total_Amount", "Payment_Method", "Device_Type",
        "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days",
        "Customer_Rating", "Gender_Male", "Gender_Female", "Gender_Other"
    ]

    # Convert input to DataFrame WITH COLUMN NAMES
    data= pd.DataFrame([input.model_dump()], columns=columns)
#     df = pd.DataFrame([input.dict()], columns=columns)

    # Scale using saved scaler
#     x_scaled = scaler.transform(df)

    # Predict
    y_prediction = model.predict(data)
    if int(y_prediction[0]) == 0:
        return {"prediction": "Customer is likely to churn"}
    else:
        return {"prediction": "Business is likely to retain customer."}