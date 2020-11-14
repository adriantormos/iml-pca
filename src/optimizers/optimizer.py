import abc
from prettytable import PrettyTable
from src.auxiliary.file_methods import print_pretty_json, save_csv
from src.auxiliary.visualize_methods import compare_multiple_lines


class Optimizer(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'get_algorithm_score') and
                callable(subclass.get_algorithm_score) and
                hasattr(subclass, 'run_algorithm') and
                callable(subclass.run_algorithm) or
                NotImplemented)

    # Main methods

    def __init__(self, optimizer_config, algorithm_config, output_path, verbose):
        self.parameters = optimizer_config['params']
        self.use_best_parameters = optimizer_config['use_best_parameters']
        self.n_runs = optimizer_config['n_runs']
        self.verbose = verbose
        self.output_path = output_path
        self.algorithm_config = algorithm_config
        self.optimizer_config = optimizer_config

    def run(self, values, labels):
        if self.verbose:
            print('Starting optimizer.', 'Algorithm: ' + self.algorithm_config['name'] + '.', 'Optimizer config:')
            print_pretty_json(self.optimizer_config)

        if len(self.parameters) > 0:

            best_params = self.algorithm_config['params'].copy()
            for parameter in self.parameters:
                parameter_name = parameter['name']
                parameter_values = parameter['values']
                algorithm_config = self.algorithm_config.copy()

                scores = []
                best_score = None
                best_value = None
                for value in parameter_values:
                    algorithm_config[parameter_name] = value
                    score = self.get_algorithm_score(algorithm_config, values, labels)
                    scores.append(score)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_value = value
                self.do_optimization_visualization(parameter_name, parameter_values, scores)

                if self.verbose:
                    print('Best value for param', parameter_name, ':', str(best_value))

                best_params[parameter_name] = best_value

            if self.use_best_parameters:
                self.algorithm_config['params'] = best_params

        if self.verbose:
            print('Final params: ', self.algorithm_config['params'])
        return self.run_algorithm(self.algorithm_config, values, labels)

    # Auxiliary methods

    @abc.abstractmethod
    def get_algorithm_score(self, algorithm_config, values, labels):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def run_algorithm(self, algorithm_config, values, labels):
        raise NotImplementedError('Method not implemented in interface class')

    def do_optimization_visualization(self, parameter_name, parameter_values, scores):
        self.do_table_results(parameter_name, parameter_values, scores)
        self.do_plot_results(parameter_name, parameter_values, scores)
        self.do_cluster_plots(parameter_name, parameter_values, scores)

    def do_table_results(self, parameter_name, parameter_values, scores):
        rows = [['parameter_value_' + parameter_name, 'score']]
        x = PrettyTable()
        x.field_names = ['parameter_value_' + parameter_name, 'score']
        for index, parameter_value in enumerate(parameter_values):
            rows.append([parameter_value, scores[index]])
            x.add_row([parameter_value, scores[index]])
        if self.output_path is not None:
            save_csv(self.output_path + '/table_scores_parameter_' + parameter_name, rows)
        print(x)

    def do_plot_results(self, parameter_name, parameter_values, scores):
        x_array = parameter_values
        lines = [(scores, None)]
        print(scores)
        print(x_array)
        compare_multiple_lines(x_array, lines, 'Scores for different values for the parameter ' + parameter_name, self.output_path + '/plot_scores_parameter_' + parameter_name, legend=False, xlabel='Different ' + parameter_name + ' values', ylabel='Score', ylim=None)

    def do_cluster_plots(self, parameter_name, parameter_values, scores):
        pass
