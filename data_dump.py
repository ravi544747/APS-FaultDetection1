import pymongo
import pandas as pd
import json

from sensor.config import mongo_client

#assigning the name for the database and collection:
DATA_FILE_PATH='/config/workspace/aps_failure_training_set1.csv'
DATABASE_NAME="aps"
COLLECTION_NAME='sensor'

if __name__=="__main__":
    df=pd.read_csv(DATA_FILE_PATH)
    print(f'ROws and columns{df.shape}')

# convert the data frame to the Jason format , sotht we can dump the record in the mango DB

    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    # insert converted jason record to mongo DB
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)  