import pytest

from src.api.schemas import CustomerData


def test_schema_valid():
    data = CustomerData(
        Tenure_Months=12,
        Monthly_Charges=70.5,

        Gender_Male=1,
        Partner_Yes=1,
        Dependents_Yes=0,
        Phone_Service_Yes=1,

        Multiple_Lines_No_phone_service=0,
        Multiple_Lines_Yes=1,

        Internet_Service_Fiber_optic=1,
        Internet_Service_No=0,

        Online_Security_No_internet_service=0,
        Online_Security_Yes=1,

        Online_Backup_No_internet_service=0,
        Online_Backup_Yes=1,

        Device_Protection_No_internet_service=0,
        Device_Protection_Yes=1,

        Tech_Support_No_internet_service=0,
        Tech_Support_Yes=1,

        Streaming_TV_No_internet_service=0,
        Streaming_TV_Yes=1,

        Streaming_Movies_No_internet_service=0,
        Streaming_Movies_Yes=1,

        Contract_One_year=0,
        Contract_Two_year=1,

        Paperless_Billing_Yes=1,

        Payment_Method_Credit_card_automatic=0,
        Payment_Method_Electronic_check=1,
        Payment_Method_Mailed_check=0
    )

    assert data is not None


def test_schema_invalid():
    with pytest.raises(Exception):
        CustomerData(Tenure_Months="errado")
