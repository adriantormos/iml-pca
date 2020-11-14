import numpy as np
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm
from sklearn.manifold import TSNE


class TSNEAlgorithm(FactorAnalysisAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.params = config['params']
        self.algorithm = TSNE(**self.params)

    def find_factors(self, values: np.ndarray) -> np.ndarray:
        return self.algorithm.fit_transform(values)

    # Auxiliary methods

    def get_kdl(self):
        return self.algorithm.kl_divergence_
