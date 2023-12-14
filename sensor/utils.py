import pandas as pd
from sensor.config import mongo_client
from sensor.logger import logging
from sensor.exception import SensorException
import os,sys
import yaml
import dill
import numpy as np

def get_collection_as_dataframe(database_name:str, collection_name:str)->pd.DataFrame():
    
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise SensorException(e, sys) from e

def write_yaml_file(file_path, data:dict):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)

    except Exception as e:
        raise SensorException(e, sys)

def convert_columns_float(df:pd.DataFrame, exclude_column:list):
    try:
        for column in df.columns:
            if column not in exclude_column:
                df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise SensorException(e, sys) from e

def save_object(file_path:str, obj:object):
    try:
        logging.info("entering the save Object method of main Utils class ")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        logging.info("exited the save Object method of main Utils class")
    except Exception as e:
        raise SensorException(e, sys) from e

def load_object(file_path:str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The File:{file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SensorException(e, sys) from e

def save_numpy_array_data(file_path:str, array:np.array):
    """
    Save Numpy array data to file
    file_path:str location of the file to save
    array: np.arry data to save
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise SensorException(e,sys) from e
def load_numpy_array_data(file_path:str)->np.array:
    """
    load the numpy array data from file
    file_path: str location to load the file
    return:np.array data loaded
    """ 
    try:
        with open (file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorException(e,sys)








   




