import abc
import numpy as np


class FactorAnalysisAlgorithm:

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'find_factors') and
                callable(subclass.find_factors) or
                NotImplemented)

    # Main methods

    def __init__(self, config, output_path, verbose):
        pass

    @abc.abstractmethod
    def find_factors(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in interface class')

    # Auxiliary methods
