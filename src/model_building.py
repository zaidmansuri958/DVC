import pandas as pd 
import os
import numpy as np 
import pickle
import yaml 
import logging
from sklearn.ensemble import RandomForestClassifier


logs_dir="logs"
os.makedirs(logs_dir,exist_ok=True)

logger=logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(logs_dir,"model_building.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")


formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path:str)->dict:
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',params_path)
        return params 
    except FileNotFoundError:
        logger.error('File not found %s',params_path)
        raise 
    except yaml.YAMLError as e:
        logger.error('Yaml error : %s',e)
        raise 
    except Exception as e:
        logger.error('unexpected error : %s',e)
        raise 

def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s",file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file : %s",e)
        raise 
    except FileNotFoundError as e:
        logger.error("File not found : %s",e)
        raise 
    except Exception as e :
        logger.error("Unexpected error occured while loading the data %s",e)
        raise 

def train_model(X_train,y_train,params):
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of sample in x train and y train must be the same")
        
        logger.debug("Initializing random forest model with parameters : %s",params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug("Model training started with %d samples",X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug("Model training completed")

        return clf 
    except ValueError as e:
        logger.error("Value Error during model training")
        raise 
    except Exception as e :
        logger.error("Error during model training %s",e)
        raise 


def save_model(model,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("Model saved to %s",file_path)
    except FileNotFoundError as e :
        logger.error("file not found : %s",e)
        raise 
    except Exception as e :
        logger.error("Error occured while saving the model : %s",e)
        raise 

def main():
    try:
        params=load_params('params.yaml')['model_building']
        train_data=load_data("./data/processed/train_tfidf.csv")
        X_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values 

        clf=train_model(X_train,y_train,params)

        model_save_path='models/model.pkl'
        save_model(clf,model_save_path)
    
    except Exception as e:
        logger.error("Failed to complete the model building process %s",e)
        print(f"Error {e}")

if __name__=="__main__":
    main()
