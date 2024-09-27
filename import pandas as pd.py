import pandas as pd

class CSVHandler:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)


    def read_csv(self):
        return self.df
    


    