import json
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections.abc import Callable

logger = logging.getLogger(__name__)

def create_live_plot_callback(
    counts: list[int], 
    values: list[float], 
    params: list[list[float]], 
    stepsize: list[float], 
    json_file_path: str | None = None,
    store_data: bool = False,
    extra_eval_freq: int | None = None,
    cost_extra: Callable | None = None,
    values_extra: list[float] | None = None,
    plot: bool = True
) -> Callable[[int, list[float], float, float, bool], None]:
    """
    Create a callback function that stores intermediate results, updates plots, and optionally writes data to a JSON file.
    
    Parameters
    ----------
    counts : list[int]
        The list where iteration counts will be stored.
    values : list[float]
        The list where mean values (e.g., cost function values) will be stored.
    params : list[list[float]]
        The list where parameter vectors will be stored.
    stepsize : list[float]
        The list where step sizes will be stored.
    json_file_path : str
        The path to the JSON file where data will be saved.
    store_data : bool, optional
        Whether to store the data in the json file provided. Defaults to false.
    extra_eval_freq : int, optional
        Number of iterations that pass between evaluations of an extra cost function.
        Used for evaluations in Hardware of the next point.
    cost_extra : Callable, optional
        The extra cost function to be evaluated every extra_eval_freq iterations.
    values_extra : list[float], optional
        The array to store the values from the extra cost function.
    plot : bool, optional
        Whether to do a live plot. Defaults to True.

    cost_extra : Callable, optional
    values_extra : list[float], optional
    
    Returns
    -------
    callback : Callable
        A function that accepts the iteration count, parameters, mean value, step size, and acceptance status,
        and stores this information.
    """
    
    if extra_eval_freq:
        assert cost_extra and values_extra, "Must provide extra cost function and array in which to store extra values."
    
    # Initialize the function with default behavior
    def store_intermediate_result_plot_live(eval_count: int, parameters: list[float], mean: float, stp_size: float, accepted: bool):
        if plot:
            clear_output(wait=True)  # Clears the previous output in the notebook
        
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)
        stepsize.append(stp_size)
        
        if extra_eval_freq:
            if len(counts) % extra_eval_freq == 0:
                print("Extra cost evaluation with provided function.")
                logger.info("Extra cost evaluation with provided function.")
                value_extra = cost_extra(params)
        
            values_extra.append(value_extra)
        
        # Store data in the file if global `store_data` flag is True
        if store_data:
            with open(json_file_path, 'a') as json_file:
                data = {
                    "Iteration": len(counts),
                    "Fidelity": -mean,
                    "Params": tuple(parameters),
                    "Time": str(datetime.datetime.now())
                }
                json.dump(data, json_file)
                json_file.write('\n')
        
        if plot:
            # Real-time plot
            plt.title("Cost Evolution")
            plt.xlabel("Iterations")
            plt.ylabel(r'$F$')
            plt.plot(range(len(values)), values, "b.")
            plt.show()

    return store_intermediate_result_plot_live

class TerminationChecker:
    """
    A termination checker class for optimization algorithms.
    
    This class allows checking whether the optimization should terminate based on either a target value
    or stagnation (flat evolution) over a number of iterations. Stagnation checking is optional and can be enabled
    by providing a `stagnation_tol` and `number_past_iterations`.

    Parameters
    ----------
    target_value : float
        The target value the optimization aims to reach. The optimization terminates when the value is within
        the given tolerance of the target value.
    tol : float
        The tolerance level for the target value, typically for convergence.
    stagnation_tol : float | None, optional
        The tolerance level for stagnation checking. The optimization terminates if the value does not
        change beyond this tolerance over the last `number_past_iterations` iterations. Default is None (no stagnation check).
    number_past_iterations : int | None, optional
        The number of past iterations to consider when checking for stagnation. Default is None (no stagnation check).

    Attributes
    ----------
    target_value : float
        The target value the optimization aims to reach.
    tol : float
        The tolerance level for the target value.
    stagnation_tol : float | None
        The tolerance level for stagnation checking, or None if no stagnation check is enabled.
    number_past_iterations : int | None
        The number of past iterations to consider for stagnation, or None if no stagnation check is enabled.
    values : list[float]
        The list of function values over iterations, used for stagnation checking.
    """

    def __init__(
        self, 
        target_value: float, 
        tol: float, 
        stagnation_tol: float | None = None, 
        number_past_iterations: int | None = None
        ):
        """
        Initialize the TerminationChecker instance.

        Parameters
        ----------
        target_value : float
            The target value the optimization aims to reach.
        tol : float
            The tolerance for convergence to the target value (default: required).
        stagnation_tol : float | None, optional
            The threshold for stagnation checking (default: None).
        number_past_iterations : int | None, optional
            The number of past iterations to track for stagnation checking (default: None).
        """
        self.target_value = target_value
        self.tol = tol
        self.stagnation_tol = stagnation_tol  # Default to None (no stagnation check)
        self.number_past_iterations = number_past_iterations  # Default to None (no stagnation check)
        self.values: list[float] = []

    def __call__(self, nfev: int, parameters: list[float], value: float, stepsize: float, accepted: bool) -> bool:
        """
        Check whether the optimization should terminate based on current optimization step.

        Parameters
        ----------
        nfev : int
            The number of function evaluations.
        parameters : list[float]
            The current parameters of the optimization.
        value : float
            The current value of the objective function.
        stepsize : float
            The current step size used in the optimization.
        accepted : bool
            Whether the current iteration was accepted.

        Returns
        -------
        bool
            True if termination criteria are met (either convergence or stagnation), False otherwise.
        """
        self.values.append(value)
        
        # If stagnation check is enabled, calculate the average of the last `number_past_iterations` values
        if self.stagnation_tol is not None and self.number_past_iterations is not None:
            stagnation_tol_std = self.stagnation_tol * 0.5
            if len(self.values) >= 0:
                last_values = self.values[-self.number_past_iterations:]
                last_few_av = np.mean(last_values)
                std_dev = np.std(last_values)
            
            # Check for stagnation over the last `number_past_iterations` values
            if len(self.values) > self.number_past_iterations and abs(value - last_few_av) < self.stagnation_tol:
                if std_dev < stagnation_tol_std:
                    logger.info("Stagnating Optimization. Average of last few iterations is " + 
                                f"{last_few_av}, standard deviation of last few values is: {std_dev}")
                    logger.info(f"Current value is {value}")
                    print("Stagnating Optimization. Average of last few iterations is " + 
                          f"{last_few_av}, standard deviation of last few values is: {std_dev}")
                    print(f"Current value is {value}")
                    return True
        
        # Check if the target value is reached within tolerance
        if abs(self.target_value - value) < self.tol:
            logger.info(f"Reached target value within tolerance: {self.target_value}")
            print(f"Reached target value within tolerance: {self.target_value}")
            return True

        return False
