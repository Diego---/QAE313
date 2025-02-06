import random
import logging
import json
import numpy as np
from datetime import datetime, time

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import TranspilerError

logger = logging.getLogger(__name__)

def sample_from_groups(original, n):
    # Ensure n is between 1 and 4
    n = max(1, min(n, 4))
    
    # Split the original list into groups of 4
    groups = [original[i:i+4] for i in range(0, len(original), 4)]
    
    # Sample 'n' elements from each group
    sampled_elements = []
    for group in groups:
        sampled_elements.extend(random.sample(group, n))
    
    return sampled_elements

def is_pure_state(rho):
    trace_rho_sq = np.trace(np.dot(rho, rho))
    print(trace_rho_sq)
    return np.isclose(trace_rho_sq, 1)

def cliffordize_circuit(circ: QuantumCircuit, skip_transpilation: bool = True) -> QuantumCircuit:
    """
    Adapt circuit to make all its gates Clifford gates.

    Parameters
    ----------
    circ : QuantumCircuit
        The quantum circuit to convert.
    skip_transpilation : bool
        Flag to skip transpilation. Defaults to True.

    Returns
    -------
    QuantumCircuit
        The quantum circuit only containing Clifford compatible gates.

    Raises
    ------
    TranspilerError
        A possible error when trying to transpile the circuit to another basis gate set.
    """
    
    new_angles = []
    if not skip_transpilation:
        try:
            transpiled_circuit = transpile(circ, basis_gates=['rx', 'ry', 'rz', 'cx'])
        except TranspilerError as e:
            logger.error("Cannot compile circuit to the selected native gate set.", exc_info=e)
            raise e
    else:
        transpiled_circuit = circ.copy()
    
    for gate in transpiled_circuit:
        operation = gate.operation
        
        # go through all gates
        for i in range(len(operation.params)):
            # Adjust the angle to be within the range (-π, π)
            angle = operation.params[i]
            
            # Round angles to the nearest multiple of π/2 to make gates Clifford
            rounded_angle = round(angle / (np.pi / 2)) * (np.pi / 2)
            operation.params[i] = rounded_angle
            new_angles.append(rounded_angle)
    
    return transpiled_circuit #, new_angles
    
def closest_pi_multiple(angle: float, target_values: list[float]) -> float:
    """
    Find the value from the given list of target values that is closest to a multiple of pi.

    Parameters
    ----------
    angle : float 
        The angle in radians.
    target_values : list[float] 
        A list of target values.

    Returns
    ----------
    float
        The value from the target values list that is closest to a multiple of pi.

    Description
    -----------
    This function calculates the value from the given list of target values that is closest to a multiple of pi. 
    It evaluates the absolute difference between the angle and each target value, selecting the target value 
    with the smallest absolute difference.
    """
    closest_value = min(target_values, key=lambda x: abs(angle - x))
    return closest_value

def round_to_pis_circuit(circ: QuantumCircuit, target_vals: list[float], skip_transpilation: bool = True) -> QuantumCircuit:
    """
    Adapt circuit to make all its gates have some multiples of pi in the angles.

    Parameters
    ----------
    circ : QuantumCircuit
        The quantum circuit to convert.
    target_vals : list[float]
        List of target values to which the parameters should be rounded.
    skip_transpilation : bool
        Flag to skip transpilation. Defaults to True.

    Returns
    -------
    QuantumCircuit
        The quantum circuit only containing Clifford compatible gates.

    Raises
    ------
    TranspilerError
        A possible error when trying to transpile the circuit to another basis gate set.
    """
    
    if not skip_transpilation:
        try:
            transpiled_circuit = transpile(circ, basis_gates=['rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'crx', 'cry', 'crz'])
        except TranspilerError as e:
            logger.error("Cannot compile circuit to the selected native gate set.", exc_info=e)
            raise e
    else:
        transpiled_circuit = circ.copy()
    
    for gate in transpiled_circuit:
        operation = gate.operation
        
        # go through all gates
        for i in range(len(operation.params)):
            angle = operation.params[i]
            
            # Round angles to the nearest multiple of π.
            rounded_angle = closest_pi_multiple(angle, target_vals)
            operation.params[i] = rounded_angle
    
    return transpiled_circuit
    
target_values = [0]
for n in [1,2,3,4,5,6,7]:
    target_values.append(np.pi * n/2)
    target_values.append(-np.pi * n/2)
    target_values.append(np.pi * n)
    target_values.append(-np.pi * n)
    target_values.append(np.pi * n/4)
    target_values.append(-np.pi * n/4)
    target_values.append(np.pi * n/4)
    target_values.append(-np.pi * n/4)
    # target_values.append(np.pi * n/np.sqrt(2))
    # target_values.append(-np.pi * n/np.sqrt(2))

    
def get_data_list_by_date_and_field(field: str, date: str | list[str], filename: str, time_window: tuple| None = None):
    """
    Extracts a list of values for a specified field from a JSON file, filtered by date and optionally by time window,
    ordered by iteration.

    Parameters
    ----------
    field : str
        The name of the field to extract from each JSON entry (e.g., 'Params').
    date : str | list[str]
        The date or list of dates to filter entries by, in 'YYYY-MM-DD' format.
    filename : str
        The name of the JSON file to read from.
    time_window : tuple, optional
        A tuple specifying the start and end times in 'HH:MM:SS' format to further filter entries by time.
        Use None for open-ended intervals (e.g., ('20:00:00', None) for all data after 8 PM).
        Default is None, which means no time filtering.

    Returns
    -------
    list
        A list of values corresponding to the specified field for the given date and time window, sorted by the 'Iteration' field.

    Notes
    -----
    Each line in the file should be a valid JSON object containing the fields 'Iteration', 'Fidelity', 'Params', and 'Time'.
    The function assumes that the 'Time' field contains date and time in the format 'YYYY-MM-DD HH:MM:SS.ssssss'.

    Examples
    --------
    >>> get_data_list_by_date_and_field('Params', '2024-11-24', 'data.json', time_window=('20:00:00', None))
    [[1.276764886887416, 0.3690630007917085], [1.2646764261894023, 0.25623736761024724]]
    """
    result = []

    # Parse the time window if provided
    start_time = None
    end_time = None
    if time_window:
        try:
            # Handle None for open-ended intervals
            if time_window[0] is not None:
                start_time = datetime.strptime(time_window[0], "%H:%M:%S").time()
            else:
                start_time = time.min  # Earliest possible time
            if time_window[1] is not None:
                end_time = datetime.strptime(time_window[1], "%H:%M:%S").time()
            else:
                end_time = time.max  # Latest possible time
        except ValueError:
            raise ValueError("Time window must be in the format ('HH:MM:SS', 'HH:MM:SS') or None for open intervals.")

    # Open the file and read it line by line
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Parse each line as a JSON object
                data = json.loads(line)
                
                # Check if the 'Time' field exists
                if 'Time' not in data:
                    continue  # Skip this line if 'Time' field is missing
                
                # Extract the date and time from the 'Time' field
                record_datetime = datetime.strptime(data['Time'], "%Y-%m-%d %H:%M:%S.%f")
                record_date = record_datetime.date()
                record_time = record_datetime.time()

                # Filter by date
                if record_date.isoformat() in (date if isinstance(date, list) else [date]):
                    # If a time window is provided, filter by time
                    if time_window:
                        if not (start_time <= record_time <= end_time):
                            continue

                    # Determine whether to use 'Iteration' or 'Epoch' key
                    iteration_key = 'Iteration' if 'Iteration' in data else 'Epoch' if 'Epoch' in data else None
                    if iteration_key is None:
                        continue

                    # Append the relevant field and the iteration number to the result list
                    result.append((data[iteration_key], data[field]))
                    
            except (json.JSONDecodeError, KeyError):
                continue
            
    # Check if no data was found for the specified date and time window
    if not result:
        print(f"No data recorded for {date} within the specified time window.")
        return []
    
    # Sort the result by the iteration number (first element in the tuple)
    result.sort(key=lambda x: x[0])
    
    # Extract only the field data (second element of each tuple) and return it
    return [item[1] for item in result]
