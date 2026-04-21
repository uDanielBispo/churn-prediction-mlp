import joblib
import numpy as np
import torch

dummy_model = joblib.load("src/models/churn_prediction_dummy_classifier_model.pkl")
logistic_model = joblib.load("src/models/churn_prediction_logistic_regression_model.pkl")
mlp_model = joblib.load("src/models/churn_prediction_mlp_pytorch_model.pkl")

def dummy_predict(data):
    features = np.array([[
        data.Tenure_Months,
        data.Monthly_Charges,
        data.Gender_Male,
        data.Partner_Yes,
        data.Dependents_Yes,
        data.Phone_Service_Yes,
        data.Multiple_Lines_No_phone_service,
        data.Multiple_Lines_Yes,
        data.Internet_Service_Fiber_optic,
        data.Internet_Service_No,
        data.Online_Security_No_internet_service,
        data.Online_Security_Yes,
        data.Online_Backup_No_internet_service,
        data.Online_Backup_Yes,
        data.Device_Protection_No_internet_service,
        data.Device_Protection_Yes,
        data.Tech_Support_No_internet_service,
        data.Tech_Support_Yes,
        data.Streaming_TV_No_internet_service,
        data.Streaming_TV_Yes,
        data.Streaming_Movies_No_internet_service,
        data.Streaming_Movies_Yes,
        data.Contract_One_year,
        data.Contract_Two_year,
        data.Paperless_Billing_Yes,
        data.Payment_Method_Credit_card_automatic,
        data.Payment_Method_Electronic_check,
        data.Payment_Method_Mailed_check
    ]])

    prediction = dummy_model.predict(features)
    return int(prediction[0])

def logistic_predict(data):
    features = np.array([[
        data.Tenure_Months,
        data.Monthly_Charges,
        data.Gender_Male,
        data.Partner_Yes,
        data.Dependents_Yes,
        data.Phone_Service_Yes,
        data.Multiple_Lines_No_phone_service,
        data.Multiple_Lines_Yes,
        data.Internet_Service_Fiber_optic,
        data.Internet_Service_No,
        data.Online_Security_No_internet_service,
        data.Online_Security_Yes,
        data.Online_Backup_No_internet_service,
        data.Online_Backup_Yes,
        data.Device_Protection_No_internet_service,
        data.Device_Protection_Yes,
        data.Tech_Support_No_internet_service,
        data.Tech_Support_Yes,
        data.Streaming_TV_No_internet_service,
        data.Streaming_TV_Yes,
        data.Streaming_Movies_No_internet_service,
        data.Streaming_Movies_Yes,
        data.Contract_One_year,
        data.Contract_Two_year,
        data.Paperless_Billing_Yes,
        data.Payment_Method_Credit_card_automatic,
        data.Payment_Method_Electronic_check,
        data.Payment_Method_Mailed_check
    ]])

    prediction = logistic_model.predict(features)
    return int(prediction[0])

def mlp_predict(data):
    features = np.array([[
        data.Tenure_Months,
        data.Monthly_Charges,
        data.Gender_Male,
        data.Partner_Yes,
        data.Dependents_Yes,
        data.Phone_Service_Yes,
        data.Multiple_Lines_No_phone_service,
        data.Multiple_Lines_Yes,
        data.Internet_Service_Fiber_optic,
        data.Internet_Service_No,
        data.Online_Security_No_internet_service,
        data.Online_Security_Yes,
        data.Online_Backup_No_internet_service,
        data.Online_Backup_Yes,
        data.Device_Protection_No_internet_service,
        data.Device_Protection_Yes,
        data.Tech_Support_No_internet_service,
        data.Tech_Support_Yes,
        data.Streaming_TV_No_internet_service,
        data.Streaming_TV_Yes,
        data.Streaming_Movies_No_internet_service,
        data.Streaming_Movies_Yes,
        data.Contract_One_year,
        data.Contract_Two_year,
        data.Paperless_Billing_Yes,
        data.Payment_Method_Credit_card_automatic,
        data.Payment_Method_Electronic_check,
        data.Payment_Method_Mailed_check
    ]])

    mlp_model.eval()

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        out = mlp_model(x)
        prob = torch.sigmoid(out)
        return int((prob > 0.5).item())