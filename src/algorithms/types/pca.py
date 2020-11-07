import numpy as np
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm


class PCAlgorithm(FactorAnalysisAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)

    def find_factors(self, values: np.ndarray) -> np.ndarray:
        return values

    # Auxiliary methods
