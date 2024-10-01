import pandas as pd
import os

class CSVFileManager:

    def __init__(self, file_path):
       
        self.file_path = file_path



    def read(self):
      
        if self.exists():
            return pd.read_csv(self.file_path)

        else:
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")


    def write(self, data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")

        data.to_csv(self.file_path, index=False)


    def exists(self):
        
        return os.path.exists(self.file_path)


    def delete(self):

        if self.exists():
            os.remove(self.file_path)

        else:
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")


    

    def get_file_path(self):  

        return self.file_path