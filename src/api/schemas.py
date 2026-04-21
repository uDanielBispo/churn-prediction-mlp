from pydantic import BaseModel

class CustomerData(BaseModel):
    Tenure_Months: float
    Monthly_Charges: float

    Gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int

    Phone_Service_Yes: int

    Multiple_Lines_No_phone_service: int
    Multiple_Lines_Yes: int

    Internet_Service_Fiber_optic: int
    Internet_Service_No: int

    Online_Security_No_internet_service: int
    Online_Security_Yes: int

    Online_Backup_No_internet_service: int
    Online_Backup_Yes: int

    Device_Protection_No_internet_service: int
    Device_Protection_Yes: int

    Tech_Support_No_internet_service: int
    Tech_Support_Yes: int

    Streaming_TV_No_internet_service: int
    Streaming_TV_Yes: int

    Streaming_Movies_No_internet_service: int
    Streaming_Movies_Yes: int

    Contract_One_year: int
    Contract_Two_year: int

    Paperless_Billing_Yes: int

    Payment_Method_Credit_card_automatic: int
    Payment_Method_Electronic_check: int
    Payment_Method_Mailed_check: int