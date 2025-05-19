import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml 


log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s- %(name)s -%(levelname)s - %(message)s')
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

def load_data(data_url:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('data loaded from %s',data_url)
        return df 
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data : %s",e)
        raise 

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe : %s",e)
        raise 
    except Exception as e:
        logger.error("Unexpected error during preprocessing : %s",e)
        raise 

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str):
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug("Train and test data saved to %s",raw_data_path)
    except Exception as e :
        logger.error("Unexpected error occured while saving the data : %s",e)
        raise 

def main():
    try:
        params=load_params(params_path='params.yaml')
        test_size=params['data_ingestion']['test_size']
        data_path="https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process : %s',e)
        print(f"Error : {e}")

if __name__ == "__main__":
    main()

