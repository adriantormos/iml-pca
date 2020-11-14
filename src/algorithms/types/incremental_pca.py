import numpy as np
from sklearn.decomposition import IncrementalPCA
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm


class IncrementalPCAlgorithm(FactorAnalysisAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.params = config['params']
        if 'batch_size' not in self.params:
            raise Exception('The param batch_size is mandatory')
        self.batch_size = self.params['batch_size']
        self.verbose = verbose

    def find_factors(self, values: np.ndarray) -> (np.ndarray, np.ndarray):
        algorithm = IncrementalPCA(**self.params)
        algorithm.fit(values)
        if self.verbose:
            print('Sorted eigen values')
            print(algorithm.singular_values_)
            print('Sorted eigen vectors')
            print(algorithm.components_)
            print('Explained variance')
            print(algorithm.explained_variance_)
            print('Explained variance ratio')
            print(algorithm.explained_variance_ratio_)
        transformed_values = algorithm.transform(values)
        reconstructed_values = algorithm.inverse_transform(transformed_values)
        return transformed_values, reconstructed_values

    # Auxiliary methods
