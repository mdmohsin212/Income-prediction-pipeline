import os
import pandas as pd

def load_data(url : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    
    except pd.errors.ParserError as e:
        print(f'Error : Failed to parse the csv file from {url}')
        print(e)
        raise
    
def save_data(df : pd.DataFrame, path : str) -> None:
    try:
        data_path = os.path.join(path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        df.to_csv(os.path.join(data_path, 'raw_data.csv'), index=False)
    
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise
    

def main():
    try:
        df = load_data('https://raw.githubusercontent.com/mdmohsin212/Machine-Learning/refs/heads/main/dataset/income.csv')
        save_data(df, "data")
    
    except Exception as e:
        print(f'Error : {e}')
        print("Failed to complete the data ingestion process")
        
if __name__ == '__main__':
    main()