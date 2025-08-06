import pandas as pd
import sys
from dataclasses import dataclass
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for datatransformation
        '''
        try:

            # Define columns that should be processed
            numerical_columns = ['age',
            'monthly_income_usd',
            'monthly_expenses_usd',
            'savings_usd',
            'loan_amount_usd',
            'loan_term_months',
            'monthly_emi_usd',
            'loan_interest_rate_pct',
            'debt_to_income_ratio',
            'savings_to_income_ratio']

            categorical_columns = ['gender',
            'education_level',
            'employment_status',
            'job_title',
            'has_loan',
            'loan_type',
            'region']

            num_pipeline= Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                    ]
                )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transformer_object_with_columns(self, numerical_columns, categorical_columns):
        '''
        This function is responsible for datatransformation with specific columns
        '''
        try:
            num_pipeline= Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                    ]
                )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            target_column_name="credit_score"
            columns_to_drop = ["credit_score", "user_id", 'record_date']
            
            # Define the columns that should be processed
            numerical_columns = ['age',
            'monthly_income_usd',
            'monthly_expenses_usd',
            'savings_usd',
            'loan_amount_usd',
            'loan_term_months',
            'monthly_emi_usd',
            'loan_interest_rate_pct',
            'debt_to_income_ratio',
            'savings_to_income_ratio']

            categorical_columns = ['gender',
            'education_level',
            'employment_status',
            'job_title',
            'has_loan',
            'loan_type',
            'region']

            input_feature_train_df=train_df.drop(columns=columns_to_drop,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=columns_to_drop,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Input feature train df shape: {input_feature_train_df.shape}")
            logging.info(f"Input feature train df columns: {input_feature_train_df.columns.tolist()}")
            logging.info(f"Target feature train df shape: {target_feature_train_df.shape}")
            
            # Check which columns actually exist in the data
            available_numerical = [col for col in numerical_columns if col in input_feature_train_df.columns]
            available_categorical = [col for col in categorical_columns if col in input_feature_train_df.columns]
            
            logging.info(f"Available numerical columns: {available_numerical}")
            logging.info(f"Available categorical columns: {available_categorical}")
            
            # Create preprocessing object with only available columns
            preprocessing_obj = self.get_data_transformer_object_with_columns(available_numerical, available_categorical)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Transformed train array shape: {input_feature_train_arr.shape}")
            logging.info(f"Target train array shape: {target_feature_train_df.shape}")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
