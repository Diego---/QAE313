from typing import Callable
import numpy as np

from qae.optimization.my_spsa import SPSA, powerseries
from qae.utils.termination_and_callback import TerminationChecker, create_live_plot_callback

class SPSAExperiment:
    """
    A class for managing and running multiple optimization experiments using Simultaneous Perturbation Stochastic Approximation (SPSA) 
    for hyperparameter tuning.

    This class is designed to handle multiple optimization runs with a set of hyperparameters, track the performance of each run, 
    and evaluate the overall success of the optimization process.

    Used for:
    - Running multiple optimization experiments with a sets of hyperparameters.
    - Storing and managing data from each optimization run, including parameter values, cost function values, and iteration counts.
    - Evaluating the performance of the optimization process based on success rate, convergence speed, and other relevant metrics.
    - Facilitating easy comparison of training behaviors across different hyperparameter configurations.
    """
    def __init__(self, a: int, alpha: float, A: int, c: int, gamma: float, resample: int, 
                 target_value=1, tol=0.004, stagnation_tol=0.005, iterations=350) -> None:
        """
        Parameters
        ----------
        a : int
            Initial scaling factor for the learning rate sequence, controlling the initial step size in parameter updates.
        alpha : float
            Decay rate for the learning rate, determining the rate of decrease in learning rate over iterations.
        A : int
            Offset for the learning rate sequence, modifying the starting index of the learning rate decay.
        c : int
            Initial scaling factor for the perturbation sequence, setting the magnitude of initial perturbations.
        gamma : float
            Decay rate for the perturbation sequence, determining how quickly perturbations decrease.
        resample : int
            Number of resampling steps per iteration, enhancing the accuracy of gradient estimation by averaging over multiple noisy samples.
        target_value : float, optional
            Target value for optimization termination.
        tol : float, optional
            Tolerance for target value.
        stagnation_tol : float, optional
            Tolerance for stagnation.
        iterations : int, optional
            Maximum number of iterations for SPSA.
        """
        self.a = a
        self.alpha = alpha
        self.A = A
        self.c = c
        self.gamma = gamma
        self.resample = resample
        self.iterations = iterations
        
        # Initialize arrays so keep track of many training runs.
        self.counts = []
        self.values = []
        self.params = []
        self.results = []
        
        # Initialize auxiliary arrays to keep track of each training run.
        self._counts = []
        self._values = []
        self._params = []

        # Initialize learning rate and perturbation generators
        self.learn_rate = self._init_learning_rate()
        self.perturbation = self._init_perturbation()

        # Set up termination checker and optimizer
        self.term_check = self._init_termination_checker(target_value, tol, stagnation_tol)
        self.spsa = self._init_spsa_optimizer(self.iterations, self.term_check)

    def _init_learning_rate(self) -> Callable:
        """Initializes the learning rate sequence generator."""
        learning_rate_gen = powerseries(self.a, self.alpha, self.A)
        learning_rate = np.array([next(learning_rate_gen) for _ in range(self.iterations)])
        return learning_rate

    def _init_perturbation(self) -> Callable:
        """Initializes the perturbation sequence generator."""
        perturbation_gen = powerseries(self.c, self.gamma)
        perturbation  = np.array([next(perturbation_gen) for _ in range(self.iterations)])
        return perturbation

    def _init_termination_checker(self, target_value, tol, stagnation_tol) -> TerminationChecker:
        """Initializes the termination checker."""
        return TerminationChecker(target_value=target_value, tol=tol, stagnation_tol=stagnation_tol, number_past_iterations=40)

    def _init_spsa_optimizer(self, iterations, termination_checker) -> SPSA:
        """Initializes the SPSA optimizer."""
        
        callback = create_live_plot_callback(self._counts, self._values, self._params, [], plot = False)
        
        return SPSA(
            maxiter=iterations,
            termination_checker=termination_checker,
            callback=callback,
            perturbation=self.perturbation,
            learning_rate=self.learn_rate,
            resamplings=self.resample,
        )
        
    def get_hyperparameters(self):
        """
        Returns the current hyperparameters of the SPSA optimizer.

        Returns
        -------
        dict
            A dictionary containing the current values of the hyperparameters:
            - 'a': initial learning rate scaling factor
            - 'alpha': decay rate for learning rate
            - 'A': offset for learning rate sequence
            - 'c': initial perturbation scaling factor
            - 'gamma': decay rate for perturbation
            - 'resample': number of resampling steps per iteration
        """
        return {
            'a': self.a,
            'alpha': self.alpha,
            'A': self.A,
            'c': self.c,
            'gamma': self.gamma,
            'resample': self.resample
        }
        
    
    def run_optimization(
        self, 
        cost_function: Callable, 
        initial_point: list, 
        cost_next: Callable | None = None) -> None:
        """
        Runs the optimization using the `minimize` method of the SPSA optimizer.

        Parameters
        ----------
        cost_function : Callable
            A callable function representing the cost function to be minimized. This function should take 
            a set of parameters as input and return a scalar value that the optimizer aims to minimize.
            
        initial_point : list
            A list representing the initial point for the optimization. This is the starting 
            set of parameters from which the optimization will begin.
            
        cost_next : Callable | Optional
            The function with which f(x_next) is calculated at each new parameter set in the optimization
            via the callback. Defaults to None, in which case cost_function is used for this.

        Returns
        -------
        The method does not return anything, but the result of the optimization is saved to the `self.result` attribute.
        You can access the optimization result through the `result` attribute after calling this method.
        """
        
        # Re-initialize spsa object to get a clean optimization
        self.spsa = self._init_spsa_optimizer(self.iterations, self.term_check)
        
        # Running the optimization using the minimize method from SPSA
        result = self.spsa.minimize(
            fun=cost_function, 
            x0=initial_point, 
            fun_next=cost_next
            )
        
        # Set the optimization results
        self.results.append(result)
        self.counts.append(self._counts)
        self.values.append(self._values)
        # Transform params into a format readily usable by plotting module
        spsa_params = [[] for i in range(len(self._params[0]))]
        for _, params_at_i in enumerate(self._params):
            for j in range(len(params_at_i)):
                spsa_params[j].append(params_at_i[j])
        self.params.append(spsa_params)
        
        # Reset temporary arrays
        self._counts = []
        self._values = []
        self._params = []
        
    def _find_best_iteration(self, values: list[float]) -> tuple[int, float]:
        """
        Finds the iteration with the best (lowest) value for the cost function from the list of values.
        
        Parameters
        ----------
        values : list[float]
            A list of values representing the fidelity or energy (the cost) at each iteration during 
            the optimization.

        Returns
        -------
        tuple[int, float]
            A tuple where:
            - The first element is the index of the iteration with the best cost.
            - The second element is the value at that iteration.
        """
        fidelity_values = [-values[i] for i in range(len(values))] 
        ind_max = np.argmax(fidelity_values)
        
        return ind_max, fidelity_values[ind_max]
    
    def evaluate_performance(self, target_value: float = 1.0, tolerance: float = 0.01) -> dict:
        """
        Evaluates the performance of the optimization process.

        Parameters
        ----------
        target_value : float, optional
            The target cost value (default is -1.0, for use in the error correcting QAE).
        tolerance : float, optional
            The tolerance range within which the final fidelity must lie to be considered successful 
            (default is 0.01).

        Returns
        -------
        dict
            A dictionary containing the performance metrics:
            - 'success_rate': The fraction of runs that reached the target value within the tolerance.
            - 'avg_convergence_time': The average number of iterations it took for successful runs to converge.
            - 'mean_final_value': The mean final fidelity value for successful runs.
            - 'std_final_value': The standard deviation of final fidelity values.
            - 'mean_convergence_rate': The average number of iterations it took for convergence.
            - 'std_convergence_rate': The standard deviation of convergence times.
        """

        if not self.results:
            raise RuntimeError("Optimization must be run before evaluating performance.")

        print(f"{len(self.results)} optimizations ran for hyperparameters: {self.get_hyperparameters()}.")

        # Variables to track performance
        successful_runs = 0
        iterations_for_successful_run = []
        final_values_for_successful_run = []

        for values_list in self.values:
            best_iteration, best_value = self._find_best_iteration(values_list)

            # Check if the best value is within the desired range (target +/- tolerance)
            if abs(best_value - target_value) <= tolerance:
                successful_runs += 1
                iterations_for_successful_run.append(best_iteration)
                final_values_for_successful_run.append(best_value)

        success_rate = successful_runs / len(self.results)

        # Calculate mean and std for final values and convergence rates
        mean_final_value = np.mean(final_values_for_successful_run) if successful_runs else 0
        std_final_value = np.std(final_values_for_successful_run) if successful_runs else 0
        mean_convergence_rate = np.mean(iterations_for_successful_run) if successful_runs else 0
        std_convergence_rate = np.std(iterations_for_successful_run) if successful_runs else 0

        # Return the performance metrics
        return {
            'hyperparameters': self.get_hyperparameters(),
            'success_rate': success_rate,
            'mean_final_value': mean_final_value,
            'std_final_value': std_final_value,
            'mean_convergence_rate': mean_convergence_rate,
            'std_convergence_rate': std_convergence_rate
        }

    def __repr__(self):
        """
        Custom string representation of the SPSAExperiment object for better display.
        
        Returns
        -------
        str
            A string describing the experiment with its hyperparameters.
        """
        return f"SPSA Experiment with hyperparameters {self.get_hyperparameters()}"        
