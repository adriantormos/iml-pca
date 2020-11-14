from src.optimizers.optimizer import Optimizer
from src.factory.algorithm import AlgorithmFactory
from src.auxiliary.evaluation_methods import run_evaluation_metric
from src.algorithms.unsupervised_algorithm import UnsupervisedAlgorithm


class UnsupervisedOptimizer(Optimizer):

    def __init__(self, optimizer_config, algorithm_config, output_path, verbose):
        super().__init__(optimizer_config, algorithm_config, output_path, verbose)
        self.metrics = optimizer_config['metrics']



    def run_algorithm(self, algorithm_config, values, labels):
        best_score = None
        best_labels = None
        for _ in range(self.n_runs):
            algorithm: UnsupervisedAlgorithm = AlgorithmFactory.select_unsupervised_algorithm(algorithm_config, self.output_path, True) # TODO resolve this disorder
            output_labels = algorithm.find_clusters(values, labels)
            score = self.evaluate_algorithm(values, output_labels)
            if best_score is None or score < best_score:
                best_score = score
                best_labels = output_labels
        return best_labels

    def get_algorithm_score(self, algorithm_config, values, labels=None):
        best_score = None
        for _ in range(self.n_runs):
            algorithm: UnsupervisedAlgorithm = AlgorithmFactory.select_unsupervised_algorithm(algorithm_config, self.output_path, True) # TODO resolve this disorder
            output_labels = algorithm.run(values)
            score = self.evaluate_algorithm(values, output_labels)
            if best_score is None or score < best_score:
                best_score = score
        return best_score

    def evaluate_algorithm(self, values, output_labels):
        sum = 0
        for metric in self.metrics:
            sum += run_evaluation_metric(metric, values, None, output_labels)
        return sum / len(self.metrics)
