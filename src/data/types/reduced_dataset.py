from src.data.dataset import Dataset
import numpy as np
import pandas as pd


class ReducedDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        super(ReducedDataset, self).__init__(config, verbose)
        self.data = pd.read_csv(config['path'], index_col=0)

    def get_raw_data(self) -> (np.ndarray, np.ndarray):
        return self.get_preprocessed_data()

    def get_raw_dataframe(self) -> pd.DataFrame:
        return self.get_preprocessed_dataframe()

    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        values = self.data.iloc[:,:-1].to_numpy()
        labels = self.data.iloc[:,-1:].to_numpy()
        return values, labels.reshape(labels.shape[0])

    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        return self.data

    # Auxiliary methods
