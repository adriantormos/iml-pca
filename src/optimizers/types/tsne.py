from src.optimizers.optimizer import Optimizer
from src.algorithms.types.tsne import TSNEAlgorithm


class TSNEOptimizer(Optimizer):

    def __init__(self, optimizer_config, algorithm_config, output_path, verbose):
        super().__init__(optimizer_config, algorithm_config, output_path, verbose)

    def get_algorithm_score(self, algorithm_config, values, labels):
        best_score = None
        for _ in range(self.n_runs):
            algorithm: TSNEAlgorithm = TSNEAlgorithm(algorithm_config, self.output_path, True)
            algorithm.find_factors(values)
            score = algorithm.get_kdl()
            if best_score is None or score < best_score:
                best_score = score
        return best_score

    def run_algorithm(self, algorithm_config, values, labels):
        best_score = None
        best_values = None
        for _ in range(self.n_runs):
            algorithm: TSNEAlgorithm = TSNEAlgorithm(algorithm_config, self.output_path, True)
            values = algorithm.find_factors(values)
            score = algorithm.get_kdl()
            if best_score is None or score < best_score:
                best_score = score
                best_values = values
        return best_values
