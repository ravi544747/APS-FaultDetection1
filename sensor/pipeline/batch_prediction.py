from sensor.exception import SensorException
from sensor.logger import logging
from sensor.predictor import ModelResolver
import pandas as pd
from datetime import datetime
import os, sys
from sensor.utils import load_object
import numpy as np

PREDICTION_DIR = "prediction"
PREDICTION_FILE_NAME=f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"creating the model Resolver object ")
        Model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file: {input_file_path}")
        df=pd.read_csv(input_file_path)
        df.replace({"na":np.NAN}, inplace = True)

        # Validation for the prediction data set    
        
        logging.info(f" Loading the transformer to transformer dataset")      
        transformer = load_object(file_path = Model_resolver.get_latest_transformer_path())

        input_feature_name = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_name])

        
        logging.info(f" Loading the model to make prediction")
        model = load_object(file_path = Model_resolver.get_latest_models_path())
        prediction = model.predict(input_arr)

        logging.info(f" Target Encoder to convert predicted columns into Categorical")
        model = load_object(file_path = Model_resolver.get_latest_target_encoder_path())

        logging.info(f" Target Encoder to convert the predicted column to Categorial")

        target_encoder = load_object(file_path= Model_resolver.get_latest_target_encoder_path())

        cat_prediction = target_encoder.inverse_transform(prediction)
        df["prediction"] = prediction
        df["cat_pred"] = cat_prediction

        prediction_file_name = os.path.basename(input_file_path.replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"))

        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header = True)
        return prediction_file_path
    except Exception as e:
        raise SensorException(e, sys)