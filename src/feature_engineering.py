import pandas as pd 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml 

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("feature_engineering")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'feature_engineering.log')
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

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('data loaded from %s',file_path)
        return df 
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data : %s",e)
        raise 

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int):
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)

        X_train=train_data['text'].values
        y_train=train_data['target'].values 
        X_test=train_data['text'].values 
        y_test=train_data['target'].values


        X_train_bow=vectorizer.fit_transform(X_train)
        X_test_bow=vectorizer.transform(X_test)


        train_df=pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train 

        test_df=pd.DataFrame(X_test_bow.toarray())
        test_df['label']=y_test 

        logger.debug('tfidf applied and data transformed')
        return train_df,test_df
    except Exception as e :
        logger.error("error during transformation : %s",e)
        raise

def save_data(df:pd.DataFrame,file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(os.path.join(file_path),index=False)
        logger.debug("data saved to %s",file_path)
    except Exception as e :
        logger.error("Unexpected error occured while saving the data : %s",e)
        raise 

def main():
    try:
        params=load_params(params_path='params.yaml')
        max_features=params['feature_engineering']['max_features']

        train_data=load_data("./data/interim/train_processed.csv")
        test_data=load_data("./data/interim/test_processed.csv")

        train_df,test_df=apply_tfidf(train_data,test_data,max_features)

        save_data(train_df,os.path.join("./data","processed","train_tfidf.csv"))
        save_data(test_df,os.path.join("./data","processed","test_tfidf.csv"))

    except Exception as e:
        logger.error('Failed to complete the features engineering process: %s',e)
        print(f"Error : {e}")

if __name__ == "__main__":
    main()