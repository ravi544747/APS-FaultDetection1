from sensor.logger import logging
from sensor.exception import SensorException
import os,sys
from sensor.utils import get_collection_as_dataframe
from sensor.entity import config_entity
from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from sensor.components import data_ingestion
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation

if __name__=="__main__":
     try:
          # Data Ingestion
          training_pipeline_config=config_entity.TrainingPipelineConfig()
          data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
          data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
          
          # Data Validation
          data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
          data_validation=DataValidation(data_validation_config=data_validation_config,
                          data_ingestion_artifact=data_ingestion_artifact)
          data_validation_artifact=data_validation.initiate_data_validation()
          # data_ Transfor maion 
          data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
          data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
          data_transformation_artifact = data_transformation.initiate_data_transformation()
          
          # model Training:

          model_trainer_config= config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
          model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
          model_trainer_artifact=model_trainer.initiate_model_trainer()

          model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config)
          model_eval = ModelEvaluation(model_eval_config=model_eval_config , 
                                        data_ingestion_artifact=data_ingestion_artifact,
                                         data_transformation_artifact=data_transformation_artifact,
                                          data_trainer_artifact=model_trainer_artifact)
          model_eval_artifact = model_eval.initiate_model_evaluation()
     except Exception as e:
          print(e)

