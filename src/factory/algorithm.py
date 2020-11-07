from src.algorithms.unsupervised_algorithm import UnsupervisedAlgorithm
from src.algorithms.factor_analysis_algorithm import FactorAnalysisAlgorithm
from src.algorithms.types.kmeans import KMeansAlgorithm
from src.algorithms.types.pca import PCAlgorithm
from src.algorithms.types.tsne import TSNEAlgorithm

class AlgorithmFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_unsupervised_algorithm(config, output_path, verbose) -> UnsupervisedAlgorithm:
        name = config['name']
        if name == 'kmeans':
            algorithm = KMeansAlgorithm(config, output_path, verbose)
        else:
            raise Exception('The unsupervised algorithm with name ' + name + ' does not exist')
        if issubclass(type(algorithm), UnsupervisedAlgorithm):
            return algorithm
        else:
            raise Exception('The unsupervised algorithm does not follow the interface definition')

    @staticmethod
    def select_factor_analysis_algorithm(config, output_path, verbose) -> FactorAnalysisAlgorithm:
        name = config['name']
        if name == 'pca':
            algorithm = PCAlgorithm(config, output_path, verbose)
        elif name == 't-sne':
            algorithm = TSNEAlgorithm(config, output_path, verbose)
        else:
            raise Exception('The factor analysis algorithm with name ' + name + ' does not exist')
        if issubclass(type(algorithm), FactorAnalysisAlgorithm):
            return algorithm
        else:
            raise Exception('The factor analysis algorithm does not follow the interface definition')
