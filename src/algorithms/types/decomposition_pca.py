import numpy as np
from sklearn.decomposition import PCA
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm


class DecompositionPCAlgorithm(FactorAnalysisAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.params = config['params']
        self.verbose = verbose

    def find_factors(self, values: np.ndarray) -> np.ndarray:
        algorithm = PCA(**self.params)
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
        return algorithm.transform(values)

    # Auxiliary methods
