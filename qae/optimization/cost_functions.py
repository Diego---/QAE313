from typing import Callable

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

from qae.circuits.fidelity_measurement import create_av_fidelity

Num = float | int

def create_cost_func(backend: Backend, 
                     ansatz: QuantumCircuit | None = None, 
                     num_shots: int = 200,
                     encoding_circ: QuantumCircuit | None = None,
                     measurement_circ: QuantumCircuit | None = None,
                     init_states: list[QuantumCircuit] | None = None, 
                     final_rotations: list[QuantumCircuit] | None = None,
                     error_list: list[int] |list[QuantumCircuit] | None = None,
                     initializer_dictionaries: list[dict] | None = None,
                     mini_batch: int | None = None,
                     use_tomo: bool = False,
                     noise_model: NoiseModel | None = None,
                     return_errors: bool = False,
                     ) -> Callable:
    """
    Create a cost function to evaluate the fidelity of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    ansatz : QuantumCircuit, optional 
        The QuantumCircuit object representing the ansatz circuit. Defaults to None, in which case the cost
        function is assumed to take as an input the ansatz.
    num_shots : int, optional
        The number of shots for each circuit executio, optional.
    encoding_circ: QuantumCircuit, optional
        The type of circuit used for encoding/decoding the initial states into the codespace. Defaults
        to 3 qubit repetition code encoding.
    measurement_circ: QuantumCircuit, optional
        The measurements to be performed provided as a QuantumCircuit. Defaults to measuring
        qubit 0 into classical bit 0.
    init_states : list[QuantumCircuit], optional 
        A list of QuantumCircuit objects representing initial states. Defaults to init_states_complete.
    final_rotations : list[QuantumCircuit], optional
        A list of QuantumCircuit objects representing the final rotations required by the initial states.
    initializer_dictionaries : list[dict], optional
        List of dictionaries to assign to a completely parametrized circuit representation of the QAE problem.
        The ansaztz is interpreted as the fully parametrized representation.
    mini_batch : int, optional
        The size of mini-batch to use for evaluating circuits. Defaults to None.
    use_tomo : bool, optional
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    noise_model : NoiseModel
        Optional NoiseModel to be used in the simulation results.
    return_errors : bool
        Whether the final cost should be returned with its associated encertainty as an AffineScalarFunc object.
        Defaults to False.

    Returns
    ----------
    function
        A cost function that evaluates the fidelity of circuits with a given ansatz.

    Description
    -----------
    This function creates a closure that generates another function `cost`. The `cost` function calculates the cost 
    associated with the average fidelity of circuits with a given ansatz. It utilizes the `create_av_fidelity` function 
    to create a function `av_fid`, which calculates the average fidelity. The `cost` function then evaluates the average 
    fidelity using the provided ansatz, parameters, initial states, and backend. The cost represents the negative 
    average fidelity.
    """
    
    av_fid = create_av_fidelity(
        backend=backend, 
        mini_batch=mini_batch,
        noise_model=noise_model,
        use_tomo=use_tomo,
        return_errors=return_errors,
        )
    
    def cost(params: list[Num] | dict[str | Parameter, Num], **kwargs):
        used_circs_indices = kwargs.get('used_circs_indices')
        return av_fid(
            ansatz,
            params,
            num_shots,
            init_states,
            final_rotations,
            error_list,
            initializer_dictionaries,
            encoding_circ,
            measurement_circ,
            used_circs_indices=used_circs_indices
            )
    
    return cost
