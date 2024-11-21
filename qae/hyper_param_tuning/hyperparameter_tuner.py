from typing import Callable, Any

from qae.hyper_param_tuning.spsa_experiment import SPSAExperiment

class HyperparameterTuner:
    """
    A class to manage hyperparameter optimization by running multiple instances
    of SPSAExperiment with varying hyperparameters and evaluating their performance.
    """
    def __init__(self, param_grid: list[dict[str, Any]], target_value: float) -> None:
        """
        Initializes the HyperparameterTuner.

        Parameters
        ----------
        param_grid : list of dict
            A list of hyperparameter dictionaries to test. Each dictionary contains
            the parameters for one run of the SPSAExperiment.
        target_value : float
            The target value for convergence.
        """
        self.param_grid = param_grid  # A list of different hyperparameter configurations
        self.target_value = target_value
        self.results = []  # To store performance metrics for each configuration
        self.experiments: list[SPSAExperiment] = []

    def run(
        self, 
        cost_function: Callable, 
        initial_point: list, 
        cost_next: Callable | None = None,
        num_runs_per_config: int = 10,
        iterations: int = 350,
        tol: float = 0.004,
        stagnation_tol: float = 0.005,
        target_value: float | int = 1
        ):
        """
        Runs the optimization across all hyperparameter configurations in the grid.

        Parameters
        ----------
        cost_function : Callable
            The cost function to be minimized.
        initial_point : np.ndarray
            The initial point for the optimization.
         cost_next : Callable | Optional
            The function with which f(x_next) is calculated.
        num_runs_per_config : int, optional
            Number of times to run each experiment configuration, by default 10.
        iterations : int, optional
            Maximum number of iterations for SPSA.
        tol : float, optional
            Tolerance for target value.
        stagnation_tol : float, optional
            Tolerance for stagnation.
        target_value : float, optional
            Target value for optimization termination.

        Returns
        -------
        None
        """
        print(f"Running {num_runs_per_config} of optimizations per hyperparameter set")
        for params in self.param_grid:
            print(f"Running for set: {params}")
            experiment = SPSAExperiment(**params, 
                                       target_value=target_value,
                                       tol=tol,
                                       stagnation_tol=stagnation_tol,
                                       iterations=iterations)
            self.experiments.append(experiment)
            for i in range(num_runs_per_config):
                print(f"Running optimization {i}/{num_runs_per_config}")
                experiment.run_optimization(cost_function, initial_point, cost_next)

    def set_performance(self, tolerance: float) -> None:
        """
        Evaluates and stores the performance of each experiment.
        
        Parameters
        ----------
        tolerance : float
            Tolerance for determining successful convergence.

        Returns
        -------
        None
        """
        self.results = []
        for experiment in self.experiments:
            performance = experiment.evaluate_performance(self.target_value, tolerance)
            self.results.append(performance)

    def get_best_config(self, metrics: list[tuple[str, bool, float]]) -> dict[str, Any]:
        """
        Returns the best hyperparameter configuration based on weighted, normalized metrics, where weights 
        control each metric's influence and normalization ensures fair comparison across different scales.

        Parameters
        ----------
        metrics : list of tuples
            Each tuple defines a metric to optimize and contains:
                - The metric name (str) to optimize (e.g., 'success_rate', 'mean_convergence_rate').
                - A boolean indicating if the metric should be minimized (True for ascending, False for descending).
                - A weight (float) indicating this metric's importance (higher = more important).

        Returns
        -------
        dict
            The best hyperparameter configuration based on the specified weighted criteria.
        """
        self._check_performance_set()
        # First, calculate min and max for each metric across all results to normalize values
        metric_ranges = {}
        for metric, _, _ in metrics:
            values = [result[metric] for result in self.results]
            metric_ranges[metric] = (min(values), max(values))

        # Define a composite score function that normalizes, weights, and applies direction
        def compute_composite_score(result):
            score = 0
            for metric, ascending, weight in metrics:
                min_val, max_val = metric_ranges[metric]
                # Normalize to [0, 1]; if min == max, we assign it a constant 0.5
                normalized_value = ((result[metric] - min_val) / (max_val - min_val)
                                    if min_val != max_val else 0.5)
                # Apply direction and weight
                contribution = (1 - normalized_value if ascending else normalized_value) * weight
                score += contribution
            return score

        # Sort by composite score and select the configuration with the highest score
        best_result = max(self.results, key=compute_composite_score)

        return best_result
    
    def get_best_experiment(self, metrics: list[tuple[str, bool, float]]) -> SPSAExperiment:
        """
        Finds and returns the SPSAExperiment object corresponding to the best hyperparameters.

        Parameters
        ----------
        metrics : list of tuples
            Each tuple defines a metric to optimize, including:
                - The metric name (str) to optimize (e.g., 'success_rate', 'mean_convergence_rate').
                - A boolean indicating if the metric should be minimized (True for ascending, False for descending).
                - A weight (float) indicating this metric's importance.

        Returns
        -------
        SPSAExperiment
            The SPSAExperiment instance corresponding to the best hyperparameter configuration.
        """
        self._check_performance_set()
        # Use `get_best_config` to find the best configuration based on metrics
        best_config = self.get_best_config(metrics)
        best_hyperparameters = best_config['hyperparameters']

        # Find the matching experiment based on hyperparameters
        for experiment in self.experiments:
            if experiment.get_hyperparameters() == best_hyperparameters:
                return experiment

        raise ValueError("Best experiment not found. Make sure experiments have been run.")

    def summarize_results(self):
        """
        Summarizes and prints the performance of all hyperparameter configurations tested.
        """
        self._check_performance_set()
        for result in self.results:
            print(f"Hyperparameters: {result['hyperparameters']}")
            print(f"Mean Final Value: {result['mean_final_value']} (from successful runs)")
            print(f"Std Final Value: {result['std_final_value']}")
            print(f"Mean Convergence Rate: {result['mean_convergence_rate']} iterations")
            print(f"Std Convergence Rate: {result['std_convergence_rate']}")
            print(f"Success Rate: {result['success_rate']}")
            print("-----------")
            
    def _check_performance_set(self) -> None:
        """
        Helper function to check if performance has been set.
        
        Raises
        ------
        RuntimeError
            If performance has not been set yet.
        """
        if not self.results:
            raise RuntimeError("Performance has not been set. Call 'set_performance' first.")
