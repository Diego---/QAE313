import random
import logging
import numpy as np
from typing import Callable
from uncertainties.core import AffineScalarFunc
from uncertainties import unumpy as unp

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_experiments.library import StateTomography
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers import JobV1 as Job
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import state_fidelity

from qae.circuits.circuit_building import build_aux_circs
from qae.circuits.circuit_constants import circ_labels

Num = float | int | np.number | AffineScalarFunc
logger = logging.getLogger(__name__)

def evaluate_final_state_fidelity(
        param_dict: dict[Parameter, float], 
        backend: Backend, 
        ansatz: QuantumCircuit | None = None,
        encoding_circ: QuantumCircuit | None = None,
        measurement_circ: QuantumCircuit | None = None,
        init_states: list[QuantumCircuit] | None = None,
        final_rotations: list[QuantumCircuit] | None = None,
        error_list: list[int] | list[QuantumCircuit] | None = None,
        initializer_dictionaries: list[dict] | None = None,
        num_shots: int = 200,
        mini_batch: int | None = None,
        noise_model: NoiseModel | None = None,
        use_tomo: bool = False,
        expected_final_state: Statevector | DensityMatrix | QuantumCircuit | None = None,
        return_errors: bool = False,
        **kwargs):
    """
    Calculate the fidelity of each circuit with a given ansatz.

    Parameters
    ----------
    param_dict : dict[Parameter, Num] 
        A dictionary mapping Parameters to numerical values for binding parameters in the ansatz circuit.
    backend : Backend 
        The backend to execute the circuits on.
    ansatz : QuantumCircuit 
        The QuantumCircuit object representing the ansatz circuit.
    encoding_circ: QuantumCircuit, optional
        The type of circuit used for encoding/decoding the initial states into the codespace. Defaults
        to 3 qubit repetition code encoding.
    measurement_circ: QuantumCircuit, optional
        The measurements to be performed provided as a QuantumCircuit. Defaults to measuring
        qubit 0 into classical bit 0.
    init_states : list[QuantumCircuit] 
        A list of QuantumCircuit objects representing initial states.
    final_rotations : list[QuantumCircuit] 
        A list of QuantumCircuit objects representing the final rotations required by the initial states.
    error_list : list[int], list[QuantumCircuit], optional
        List of errors to be used. Defaults to single qubit flips in each qubit as a list of QuantumCircuits.
    initializer_dictionaries : list[dict], optional
        List of dictionaries to assign to a completely parametrized circuit representation of the QAE problem.
        The ansaztz is interpreted as the fully parametrized representation.
    num_shots : int 
        The number of shots for each circuit execution.
    mini_batch : int, optional 
        The size of mini-batch to use for evaluating circuits. Defaults to None.
    noise_model : NoiseModel
        Noise model to be used in the simulation.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    expected_final_state : Statevector | DensityMatrix | QuantumCircuit, optional
        The expected final state against which the final fidelity is measured. Defaults to the single qubit
        |0> state.
    return_errors : bool
        Whether the final fidelities should be returned with their associated uncertainties as an AffineScalarFunc
        object. Defaults to False.

    Returns
    ----------
    list[float] 
        A list of fidelity values for each evaluated circuit.

    Description
    -----------
    This function calculates the fidelity of each circuit in relation to a given ansatz. It first builds auxiliary circuits
    using the `build_aux_circs` function. If a mini-batch size is specified and non-zero, a subset of circuits is selected 
    from the auxiliary circuits. Each selected circuit is then executed on the specified backend with the given number 
    of shots. The fidelity of each executed circuit is calculated based on the resulting measurement statistics, 
    compared to the target state |0⟩.

    The fidelity is calculated using state fidelity, where the resulting state is compared to the target state |0⟩.
    """

    fids = []
    jobs = []
    
    # If initializer_dictionaries is set, aux_circs will be a list of dictionaries with
    # the different initialization parameter maps (parameters which have to be set to get initial
    # states and other circuit settings) joined with the parameter_map to evaluate the final state
    # fidelity. Otherwise, aux circs are QuantumCircuits.
    aux_circs = build_aux_circs(ansatz=ansatz, 
                                init_states=init_states, 
                                final_rotations=final_rotations,
                                measurement_circ=measurement_circ,
                                initializer_dictionaries=initializer_dictionaries,
                                encoding_circ=encoding_circ,
                                error_list=error_list,
                                param_dict=param_dict,
                                )
    
    # Set the circuits that will be used to evaluate the fidelity
    uci = kwargs.get('used_circs_indices')

    if not (uci is None):
        used_circs: list[QuantumCircuit | dict] = [aux_circs[i] for i in uci]
        logger.info(f"Used circuits: {', '.join(circ_labels[i] for i in uci)}.")
        used_circ_labels = [circ_labels[i] for i in uci]
    else:
        if mini_batch:
            num_circs_per_group = min(mini_batch, 4)
            inds = random.sample([0,1,2,3], num_circs_per_group)
            used_circs: list[QuantumCircuit] = []
            for i in inds:
                inds.append(i+4)
                inds.append(i+8)
            used_circs = [aux_circs[i] for i in inds]
            logger.info(f"Used circuits: {', '.join(circ_labels[i] for i in inds)}")
            used_circ_labels = [circ_labels[i] for i in inds]
        else:
            used_circs: list[QuantumCircuit] = aux_circs.copy()
            used_circ_labels = circ_labels.copy()

    logger.info(f"{len(used_circs)} circuits were used to calculate the averaged fidelity.")
    if noise_model:
        logger.info("A noise model was used.")
    if use_tomo:
        if expected_final_state is None:
            final_state = Statevector.from_label('0')
        elif isinstance(expected_final_state, QuantumCircuit):
            final_state = Statevector(expected_final_state)
        elif not isinstance(expected_final_state, (Statevector, DensityMatrix)):
            raise TypeError(f"Expected Statevector, Density Matrix, or QuantumCircuit object, not {type(expected_final_state)}.")
        else:
            final_state = expected_final_state.copy()
        logger.info("Performing state tomography on resulting qubit state against provided final state.")
        
    # Set up the jobs to be ran
    for i, circ in enumerate(used_circs):
        if use_tomo:
            tomography = StateTomography(
                circuit = circ.remove_final_measurements(inplace = False),
                backend = backend,
                measurement_indices = [0]
                )
            jobs += [tomography.run(backend=backend, job_name = f'Circuit_{used_circ_labels[i]}')]
        else:
            if noise_model:
                job : Job = backend.run(
                    [circ], shots = num_shots, noise_model = noise_model, job_name = f'Circuit_{used_circ_labels[i]}'
                    )
            else:
                # The ansatz is the full parametrized circuit and circ = aux_circ[i] is the parameter dictionary 
                # for circuit i with the parameters given by param_dict.
                if backend.name in ['red_trap_backend', 'umz_simulator_backend'] and not initializer_dictionaries is None:
                    job : Job = backend.run(
                        [ansatz],
                        shots = num_shots,
                        job_name = f'Circuit_{used_circ_labels[i]}',
                        execution_settings = {'parameters': circ}
                    )
                else:
                    # aux_circs are already the circuits that need to be run in the not fully parametrized implementation. 
                    job : Job = backend.run([circ], shots = num_shots, job_name = f'Circuit_{used_circ_labels[i]}')
            jobs += [job]

    # Run the circuits, either by calculating the probability of getting 0 
    # or by doing final state tomography against expexted_final_state
    for i, job in enumerate(jobs):
        if use_tomo:
            resulting_density_matrix = job.analysis_results("state").value
            fid = state_fidelity(resulting_density_matrix, final_state)
            fids += [fid]
        else:
            counts =  job.result().get_counts()
            logger.info(f"Resulting counts are: {counts}")
            count_labels = list(counts.keys())
            count_labels.sort()
            if len(counts) == 2:
                frequencies = [counts[c] / num_shots for c in count_labels]
            else:
                # Counts is keyed as '0x0' or '0x1', or as '0' or '1', so the last 
                # letter of the string gives which was the only result one gets.
                bit = count_labels[0][-1]
                freq_to_append = [1, 0] if bit == '0' else [0, 1]
                frequencies = freq_to_append

            # As discussed above, the resulting state should always be zero. The only state
            # that produces the measurement statistics (when measuring sigma_Z) of the 0 state 
            # is the 0 state, so comparing using only the frequencies (populations) should be 
            # sufficient.
            # The target state is always the zero state. This is effectively just the probability
            # of measuring 0, approximated by the first element of frequencies.
            fids += [frequencies[0]]

    errs = [fid * (1 - fid) / num_shots for fid in fids]
    ufids = unp.uarray(fids, errs)
    
    logger.info("The calculated fidelities with errors are:")
    for ufid in ufids:
        logger.info(f"{ufid}.")        

    if return_errors:
        return ufids
    
    return fids

def create_av_fidelity(backend: Backend, 
                       mini_batch: int | None = None, 
                       noise_model: NoiseModel | None = None,
                       use_tomo: bool = False,
                       return_errors: bool = False,
                       ) -> Callable:
    """
    Create a function to calculate the average fidelity of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    mini_batch : int
        The size of mini-batch to use for evaluating circuits.
    noise_model : NoiseModel, optional
        Optional NoiseModel to be used in the simulation results.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    return_errors : bool
        Whether the function produced by this closure should return an uncertainties object which includes
        errors. 
    
    Returns
    ----------
    function
        A function that calculates the average fidelity of circuits with a given ansatz.

    Description
    -----------
    This function creates a closure that generates another function `averaged_fidelity`. The `averaged_fidelity` function 
    calculates the average fidelity of circuits with a given ansatz. It utilizes the `evaluate_final_state_fidelity` function to 
    evaluate fidelity for each circuit, using the provided ansatz, parameters, initial states, and backend. The average fidelity 
    is computed as the negative average of the fidelity values obtained from `evaluate_final_state_fidelity`.
    """
    def averaged_fidelity(
        ansatz: QuantumCircuit,
        parameters: list[Num] | dict[str | Parameter, Num | Parameter],
        num_shots: int, 
        init_states: list[QuantumCircuit] | None, 
        final_rotations: list[QuantumCircuit] | None,
        error_list: list[int] | list[QuantumCircuit] | None,
        initializer_dictionaries: list[dict] | None,
        encoding_circ: QuantumCircuit | None,
        measurement_circ: QuantumCircuit | None,
        **kwargs
        ) -> Num:
        
        if isinstance(parameters, dict):
            param_dict = parameters
        elif initializer_dictionaries is None:
            param_dict = {ansatz_parameter : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}
        else:
            actual_ansatz_parameters = [param.name for param in ansatz.parameters if param.name not in initializer_dictionaries[0]]
            param_dict = {ansatz_parameter : parameters[i] for i, ansatz_parameter in enumerate(actual_ansatz_parameters)}

        uci = kwargs.get('used_circs_indices')

        fids = evaluate_final_state_fidelity(
            param_dict, 
            backend, 
            ansatz,
            encoding_circ,
            measurement_circ,
            init_states, 
            final_rotations,
            error_list,
            initializer_dictionaries,
            num_shots, 
            mini_batch, 
            noise_model=noise_model,  
            use_tomo=use_tomo,
            used_circs_indices=uci,
            return_errors=return_errors,
            )

        average = np.average(fids)
        return -average
    
    return averaged_fidelity
