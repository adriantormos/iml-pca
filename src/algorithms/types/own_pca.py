import numpy as np
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm
from src.auxiliary.visualize_methods import compare_multiple_lines


class OurPCAlgorithm(FactorAnalysisAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.n_components = config['params']['n_components']
        self.show_variance_plots = config['show_variance_plots']
        self.verbose = verbose
        self.output_path = output_path

    def find_factors(self, values: np.ndarray) -> (np.ndarray, np.ndarray):
        # Compute the d-dimensional mean vector
        dimensions_mean = values.mean(axis=0)
        if self.verbose:
            print('Dimensions mean')
            print(dimensions_mean)

        # Center all means in 0
        values = values - dimensions_mean

        # Compute the covariance matrix of the whole data set
        covariance_matrix = np.cov(values.T)
        if self.verbose:
            print('Covariance matrix')
            print(covariance_matrix)

        # Calculate eigenvectors and their corresponding eigenvalues of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        if self.verbose:
            print('Eigen values')
            print(eigen_values)
            print('Eigen vectors')
            print(eigen_vectors)

        # Sort the eigenvectors by decreasing eigenvalues
        sorted_indexes = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sorted_indexes]
        eigen_vectors = eigen_vectors[:, sorted_indexes]
        if self.verbose:
            print('Sorted eigen values')
            print(eigen_values)
            print('Sorted eigen vectors')
            print(eigen_vectors)

        # Transform eigenvectors to the new feature space
        values = values.dot(eigen_vectors)

        # Compute the explained variances of each eigenvector
        variances = np.var(values, axis=0)
        total_variance = np.sum(variances)
        explained_variances = variances / total_variance
        accumulated_variances = np.cumsum(explained_variances)
        accumulated_variances[-1] = 1
        if self.verbose:
            print('Explained variance')
            print(explained_variances)
            print('Accumulated explained variance')
            print(accumulated_variances)

        if self.show_variance_plots:
            lines = [(explained_variances, 'explained variance by each dimension'),
                     (accumulated_variances, 'accumulated variance')]
            x_array = list(range(1, eigen_vectors.shape[0] + 1))
            compare_multiple_lines(x_array, lines, 'Variance explained by dimensions', self.output_path + '/explained_variance', legend=True, xlabel='Dimension', ylabel='Variance', ylim=None, legend_position='center right')

        # Choose n_components eigenvectors with the largest eigenvalues
        if self.n_components > 0:
            values = values[:, :self.n_components]
            eigen_vectors = eigen_vectors[:self.n_components]

        reconstructed_values = values.dot(eigen_vectors)

        return values, reconstructed_values

    # Auxiliary methods
