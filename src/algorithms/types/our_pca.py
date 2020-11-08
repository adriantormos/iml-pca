import numpy as np
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm


class OurPCAlgorithm(FactorAnalysisAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.number_factors = config['number_factors']
        self.verbose = verbose

    def find_factors(self, values: np.ndarray) -> np.ndarray:
        dimensions_mean = values.mean(axis=0)
        if self.verbose:
            print('Dimensions mean')
            print(dimensions_mean)
        values = values - dimensions_mean
        covariance_matrix = np.cov(values.T)
        if self.verbose:
            print('Covariance matrix')
            print(covariance_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        if self.verbose:
            print('Eigen values')
            print(eigen_values)
            print('Eigen vectors')
            print(eigen_vectors)
        sorted_indexes = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sorted_indexes]
        eigen_vectors = eigen_vectors[:, sorted_indexes]
        if self.verbose:
            print('Sorted eigen values')
            print(eigen_values)
            print('Sorted eigen vectors')
            print(eigen_vectors)
        if self.number_factors > 0:
            eigen_vectors = eigen_vectors[:, :self.number_factors]
        return values.dot(eigen_vectors)

    # Auxiliary methods
