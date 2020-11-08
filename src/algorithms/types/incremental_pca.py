import numpy as np
from sklearn.decomposition import IncrementalPCA
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm


class IncrementalPCAlgorithm(FactorAnalysisAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.number_factors = config['number_factors']
        self.batch_size = config['batch_size']
        self.verbose = verbose

    def find_factors(self, values: np.ndarray) -> np.ndarray:
        algorithm = IncrementalPCA(n_components=self.number_factors, batch_size=self.batch_size)
        for index in range(0, values.shape[0], self.batch_size):
            algorithm.partial_fit(values[index:(index + self.batch_size)])
        if self.verbose:
            print('Sorted eigen values')
            print(algorithm.singular_values_)
            print('Sorted eigen vectors')
            print(algorithm.components_)
            print('Explained variance')
            print(algorithm.explained_variance_)
            print('Explained variance ratio')
            print(algorithm.explained_variance_ratio_)
        return algorithm.transform(values)

    # Auxiliary methods
