import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        user_id: str,
        age: int,
        gender: str,
        education_level: str,
        employment_status: str,
        job_title: str,
        monthly_income_usd: float,
        monthly_expenses_usd: float,
        savings_usd: float,
        has_loan: str,
        loan_type: str,
        loan_amount_usd: float,
        loan_term_months: int,
        monthly_emi_usd: float,
        loan_interest_rate_pct: float,
        debt_to_income_ratio: float,
        savings_to_income_ratio: float,
        region: str,
        record_date: str):

        self.user_id = user_id
        self.age = age
        self.gender = gender
        self.education_level = education_level
        self.employment_status = employment_status
        self.job_title = job_title
        self.monthly_income_usd = monthly_income_usd
        self.monthly_expenses_usd = monthly_expenses_usd
        self.savings_usd = savings_usd
        self.has_loan = has_loan
        self.loan_type = loan_type
        self.loan_amount_usd = loan_amount_usd
        self.loan_term_months = loan_term_months
        self.monthly_emi_usd = monthly_emi_usd
        self.loan_interest_rate_pct = loan_interest_rate_pct
        self.debt_to_income_ratio = debt_to_income_ratio
        self.savings_to_income_ratio = savings_to_income_ratio
        self.region = region
        self.record_date = record_date

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "user_id":[self.user_id],
                "age":[self.age],
                "gender":[self.gender],
                "education_level":[self.education_level],
                "employment_status":[self.employment_status],
                "job_title":[self.job_title],
                "monthly_income_usd":[self.monthly_income_usd],
                "monthly_expenses_usd":[self.monthly_expenses_usd],
                "savings_usd":[self.savings_usd],
                "has_loan":[self.has_loan],
                "loan_type":[self.loan_type],
                "loan_amount_usd":[self.loan_amount_usd],
                "loan_term_months":[self.loan_term_months],
                "monthly_emi_usd":[self.monthly_emi_usd],
                "loan_interest_rate_pct":[self.loan_interest_rate_pct],
                "debt_to_income_ratio":[self.debt_to_income_ratio],
                "savings_to_income_ratio":[self.savings_to_income_ratio],
                "region":[self.region],
                "record_date":[self.record_date]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
