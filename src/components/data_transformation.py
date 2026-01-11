import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            num_cols = ['age', 'avg_glucose_level', 'bmi']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns : {num_cols} [Scaling completed!]")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
            logging.info(f"Categorical columns : {cat_cols} [Encoding completed!]")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data!")

            logging.info("Geting preprocessor object")

            preprocesssor = self.get_data_transformer_object()
            target_col = "stroke" 
            cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            num_cols = ['age', 'avg_glucose_level', 'bmi']

            input_train_df = train_df.drop(columns=[target_col], axis=1)
            input_test_df = test_df.drop(columns=[target_col], axis=1)

            target_train_df = train_df[target_col]
            target_test_df = test_df[target_col]


            logging.info("Applying preprocessor object on training and testing data")

            input_train_arr = preprocesssor.fit_transform(input_train_df)
            input_test_arr = preprocesssor.transform(input_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocesssor
            )

            logging.info("Saved preprocessor object!")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

