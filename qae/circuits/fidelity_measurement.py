import random
import logging
import time
import numpy as np
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimator
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_experiments.library import StateTomography
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers import JobV1 as Job
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import state_fidelity

from qae.circuits.circuit_building import build_aux_circs
from qae.circuits.circuit_constants import zero, init_states_complete, circ_labels

Num = float | int | np.number
logger = logging.getLogger(__name__)

def evaluate_final_state_fidelity(
        param_dict: dict[Parameter, float], 
        backend: Backend, 
        ansatz: QuantumCircuit | None = None, 
        init_states: list[QuantumCircuit] | None = None,
        final_rotations: list[QuantumCircuit] | None = None,
        num_shots: int = 200,
        mini_batch: int | None = None,
        noise_model: NoiseModel | None = None,
        use_tomo: bool = False,
        do_continous_evaluation: bool = False,
        use_pseudo_fidelity: bool = True,
        skip_compilation: bool = False,
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
    init_states : list[QuantumCircuit] 
        A list of QuantumCircuit objects representing initial states.
    final_rotations : list[QuantumCircuit] 
        A list of QuantumCircuit objects representing the final rotations required by the initial states.
    num_shots : int 
        The number of shots for each circuit execution.
    mini_batch : int, optional 
        The size of mini-batch to use for evaluating circuits. Defaults to None.
    noise_model : NoiseModel
        Noise model to be used in the simulation.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    do_continous_evaluation : bool
        Whether every time the fidelity of the resulting measurement statistics are close to the ideal simulation.
    use_pseudo_fidelity : bool
        Whether one should calculate a pseudo-fidelity only out of the measurement statistics. Defaults to True.
    skip_compilation : bool
        Whether to skip compilation.

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

    if not use_pseudo_fidelity:
        raise NotImplementedError("This functionality has not been implemented.")

    fids = []
    jobs = []
    aux_circs = build_aux_circs(ansatz=ansatz, 
                                init_states=init_states, 
                                final_rotations=final_rotations, 
                                param_dict=param_dict,
                                use_resets=False)
    
    if do_continous_evaluation:
        aux_circs_eval = aux_circs.copy
        ideal_states: list[Statevector] = []
    
    uci = kwargs.get('used_circs_indices')
    
    if not (uci is None):
        used_circs: list[QuantumCircuit] = [aux_circs[i] for i in uci]
        logger.info(f"Used circuits: {', '.join(circ_labels[i] for i in uci)}.")
        if do_continous_evaluation:
            used_circs_eval: list[QuantumCircuit] = [aux_circs_eval[i] for i in uci]
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
            if do_continous_evaluation:
                used_circs_eval = [aux_circs_eval[i] for i in inds]
        else:
            used_circs: list[QuantumCircuit] = aux_circs.copy()
            if do_continous_evaluation:
                used_circs_eval: list[QuantumCircuit] = aux_circs_eval.copy()

    logger.info(f"{len(used_circs)} circuits were used to calculate the averaged fidelity.")
    if noise_model:
        logger.info("A noise model was used.")
    if use_tomo:
        logger.info("Performing state tomography on resulting qubit state.")
    for i, circ in enumerate(used_circs):

        if do_continous_evaluation:
            circ_no_measurements = used_circs_eval[i].remove_final_measurements(inplace = False)
            ideal_state = Statevector(circ_no_measurements)
            ideal_state = partial_trace(ideal_state, [1,2])
            ideal_states += [ideal_state]

        if use_tomo:
            tomography = StateTomography(
                circuit = circ.remove_final_measurements(inplace = False),
                backend = backend,
                measurement_indices = [0]
                )
            jobs += [tomography.run(backend=backend)]
        else:
            if noise_model:
                job : Job = backend.run([circ], shots = num_shots, noise_model = noise_model, job_name = f'Circuit_{i}')
            else:
                if backend.name in ['red_trap_backend', 'umz_simulator_backend'] and skip_compilation:
                    job : Job = backend.run(
                        [circ],
                        shots = num_shots,
                        job_name = f'Circuit_{i}',
                        use_rz_phase_tracking = False
                    )
                else:
                    job : Job = backend.run([circ], shots = num_shots, job_name = f'Circuit_{i}')
            jobs += [job]

    for i, job in enumerate(jobs):
        if use_tomo:
            resulting_density_matrix = job.analysis_results("state").value
            fid = state_fidelity(resulting_density_matrix, zero)
            fids += [fid]

            if do_continous_evaluation:
                evaluation_fid = state_fidelity(resulting_density_matrix, ideal_states[i])
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
            pseudo_state = np.dot([np.array([1,0]), np.array([0,1])], np.sqrt(frequencies))
            # The target state is always the zero state.
            fid = state_fidelity(pseudo_state, zero)
            fids += [fid]

            if do_continous_evaluation:
                evaluation_fid = state_fidelity(pseudo_state, ideal_states[i])

        if do_continous_evaluation:
            logger.info(f"The fidelity between the simulated state and the executed statistics is: {evaluation_fid}.")

    return fids

def create_av_fidelity(backend: Backend, 
                       mini_batch: int | None = None, 
                       noise_model: NoiseModel | None = None,
                       use_tomo: bool = False,
                       do_continous_evaluation: bool = False,
                       skip_compilation: bool = False):
    """
    Create a function to calculate the average fidelity of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    mini_batch : int
        The size of mini-batch to use for evaluating circuits.
    noise_model : NoiseModel
        Optional NoiseModel to be used in the simulation results.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    do_continous_evaluation : bool
        Whether every time the fidelity of the resulting measurement statistics are close to the ideal simulation.
    skip_compilation : bool
        Whether to skip compilation.
    
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
        parameters: list[Num] | dict[Parameter, Num | Parameter],
        num_shots: int, 
        init_states: list[QuantumCircuit] | None, 
        final_rotations: list[QuantumCircuit] | None, 
        **kwargs
        ) -> Num:
        
        if isinstance(parameters, dict):
            param_dict = parameters
        else:
            param_dict = {ansatz_parameter : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}

        if init_states is None:
            init_states = init_states_complete.copy()

        uci = kwargs.get('used_circs_indices')

        fids = evaluate_final_state_fidelity(param_dict, backend, ansatz, init_states, final_rotations,  
                                                num_shots, mini_batch, used_circs_indices = uci, 
                                                noise_model = noise_model,  use_tomo = use_tomo,
                                                do_continous_evaluation = do_continous_evaluation,
                                                skip_compilation = skip_compilation)

        return -np.average(fids)
    
    return averaged_fidelity

def create_stochastic_av_fidelity(backend: Backend, 
                                  noise_model: NoiseModel | None = None,
                                  use_tomo: bool = False,
                                  do_continous_evaluation: bool = False,
                                  skip_compilation: bool = False):
    """
    Create a function to calculate the average fidelity of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    noise_model : NoiseModel
        Optional NoiseModel to be used in the simulation results.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    do_continous_evaluation : bool
        Whether every time the fidelity of the resulting measurement statistics are close to the ideal simulation.
    skip_compilation : bool
        Whether to skip compilation.

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
    def averaged_fidelity(ansatz: QuantumCircuit, parameters: list[Num], num_shots: int, init_states: list[QuantumCircuit] | None, final_rotations: list[QuantumCircuit] | None, **kwargs):
        param_dict = {ansatz_parameter : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}

        if init_states is None:
            init_states = init_states_complete.copy()

        used_circs_indices = kwargs['used_circs_indices'] 
        fids = evaluate_final_state_fidelity(param_dict, backend, ansatz, init_states, final_rotations, num_shots, 
                                             used_circs_indices = used_circs_indices, noise_model = noise_model,
                                             use_tomo = use_tomo, do_continous_evaluation = do_continous_evaluation,
                                             skip_compilation = skip_compilation)
            
        return -np.average(fids)
    
    return averaged_fidelity

