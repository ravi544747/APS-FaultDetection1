
from sensor.entity import artifact_entity
from sensor.entity import config_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from typing import Optional
from sensor import utils
from sensor.config import TARGET_COLUMN

class DataValidation:
    def __init__(self,
                     data_validation_config:config_entity.DataValidationConfig,
                     data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise SensorException(e,sys)
    
    
    def drop_missing_values_columns(self,df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        """
            this function will drop column which contains missing values more than specified threshold

            df: Accepts Pandas data frame
            threshold:percentage criteria for drop column
            ================================================================================================
            Returns: pandas dataframe if atleast signle column is available after missing columns drop , else none

        """
        try:
            threshold=self.data_validation_config.missing_threshold
            null_report=df.isna().sum()/df.shape[0]
            #selecting the columns name which contains the null values

            logging.info(f"selecting the columns name which contains the null values above to {threshold} ")
            drop_columns_names=null_report[null_report>threshold].index

            logging.info(f"columns to drop: {list(drop_columns_names)} ")
            
            self.validation_error[report_key_name]=list(drop_columns_names)
            df.drop(list(drop_columns_names), axis=1,inplace=True)
            # returns NULL if no columns left:
            
            if len(df.columns)==0:                
                return None
            else:
                return df
        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame, current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            base_columns= base_df.columns
            current_columns=current_df.columns

            missing_columns=[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column:[{base}is not in not available]")
                    missing_columns.append(base_column)
            
            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False
            return True

        except Exception as e:
            raise SensorException(e,sys)

    def data_drift(self,base_df:pd.DataFrame, current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()
            base_columns= base_df.columns
            current_columns=current_df.columns
            for base_column in base_columns:
                base_data, current_data= base_df[base_column], current_df[base_column]
                # Null_hypothesis is both columns data drawn from same distribution
                logging.info(f"Hypoyhesis{base_column}:{base_data.dtype},{current_data.dtype}")
                same_distribution=ks_2samp(base_data, current_data)

                if same_distribution.pvalue>0.05:
                    # we are accepting null hypothesis:
                    drift_report[base_column]={
                        "pvalues": float(same_distribution.pvalue),
                        "Same_distribution": True
                    }
                else:
                    drift_report[base_column]={
                        "pvalues": float(same_distribution.pvalue),
                        "Same_distribution":False
                    }
                    # different distribution

            self.validation_error[report_key_name]=drift_report




        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_validation(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Reading Base data Frame")
            base_df=pd.read_csv(self.data_validation_config.base_file_path)
            logging.info(f"Replace Null Values in Base data frame")
            base_df.replace({"na":np.NAN},inplace=True)
            # base_df  has na as null
            logging.info(f"droping the null values columns from base data frame")
            base_df=self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_within _base_dataset")

            logging.info(f"Reading Train data Frame")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Reading Test data Frame")
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"droping the null values columns from train data frame")
            train_df=self.drop_missing_values_columns(df=train_df, report_key_name="missing_values_within_train_dataset")
            logging.info(f"droping the null values columns from test data frame")           

            test_df=self.drop_missing_values_columns(df=test_df, report_key_name="missing_values_within_test_dataset")
            exclude_columns=[TARGET_COLUMN]

            base_df= utils.convert_columns_float(df=base_df, exclude_column=exclude_columns)
            train_df= utils.convert_columns_float(df=train_df, exclude_column=exclude_columns)
            test_df= utils.convert_columns_float(df=test_df, exclude_column=exclude_columns)

            logging.info(f"is all required columns present in train data Frame")          
            train_df_column_status=self.is_required_columns_exists(base_df=base_df, current_df=train_df,report_key_name="missing_columns_within _train_dataset")
            logging.info(f"is all required columns present in test data Frame")  
            test_df_column_status=self.is_required_columns_exists(base_df=base_df, current_df=test_df, report_key_name="missing_columns_within _test_dataset")

            if train_df_column_status:
                logging.info(f"As all the columns are available in train df hence detecting the data drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="datadrift_within _train_dataset")
            if test_df_column_status:
                logging.info(f"As all the columns are available in test df hence detecting the data drift")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="missing_values_within _test_dataset")

            # write the report:
            logging.info(f"writing a report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                 data=self.validation_error)
            
            data_validation_artifact=artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f" Data Validation artifact :{ data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)