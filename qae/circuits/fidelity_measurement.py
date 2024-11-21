import random
import logging
import time
import numpy as np
from urllib3.exceptions import MaxRetryError, HTTPError
from requests.exceptions import ConnectionError

from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimator
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_experiments.library import StateTomography
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers import JobV1 as Job
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import state_fidelity, SparsePauliOp

from qae.circuits.circuit_building import build_aux_circs
from qae.circuits.circuit_constants import zero, init_states_complete

from umz_sequence_generator.sequence_generator import OperationSequence
from umz_backend_connector.umz_connector import UmzConnector

Num = float | int
logger = logging.getLogger(__name__)

def evaluate_final_state_expectation(
        param_dict: dict[Parameter, float], 
        backend: Backend, 
        ansatz: QuantumCircuit | None = None, 
        init_states: list[QuantumCircuit] | None = None,
        num_shots: int = 200,
        mini_batch: int | None = None,
        noise_model: NoiseModel | None = None,
        **kwargs) -> list[float]:
    """
    Calculate the expectation value of Z for each circuit with a given ansatz. Should be -1 for input state
    |0>, +1 for input state |1>, and +1 for input state |+>, since there is a final rotation.

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

    Returns
    ----------
    list[float] 
        A list of expectation values for each evaluated circuit.

    Description
    -----------
    This function calculates the expectation value of the Z operator for the final state of each circuit in relation to a given ansatz. 
    With |psi> = U_{ansatz}*U_{error}*U_{encoding}*|input>, the values calculated are <psi|Z|psi> (single qubit state).
    It first builds auxiliary circuits using the `build_aux_circs` function. If a mini-batch size is specified and non-zero, 
    a subset of circuits is selected from the auxiliary circuits. Each selected circuit is then executed on the specified backend with 
    the given number of shots.
    """

    exps = []
    ideal_values = [1,1,1,1,-1,-1,-1,-1,1,1,1,1]

    estimator = BackendEstimator(backend = backend, options = {'shots' : num_shots})

    final_rotations = [QuantumCircuit(3)]*8
    plus_rot = QuantumCircuit(3)
    plus_rot.h(0)
    final_rotations += [plus_rot]*4

    aux_circs = build_aux_circs(ansatz=ansatz, 
                                init_states=init_states, 
                                final_rotations=final_rotations, 
                                param_dict=param_dict,
                                use_measurement=False)
    
    if 'used_circs_indices' in kwargs:
        indices = kwargs['used_circs_indices']
        used_circs: list[QuantumCircuit] = [aux_circs[i] for i in indices]
        used_ideal: list[int] = [ideal_values[i] for i in indices]
    else:
        if mini_batch:
            num_circs_per_group = min(mini_batch, 4)
            inds = random.sample([0,1,2,3], num_circs_per_group)
            used_circs: list[QuantumCircuit] = []
            used_ideal: list[int] = []
            for i in inds:
                used_circs += [aux_circs[i],aux_circs[i+4],aux_circs[i+8]]
                used_ideal += [ideal_values[i],ideal_values[i+4],ideal_values[i+8]]
        else:
            used_circs: list[QuantumCircuit] = aux_circs.copy()
            used_ideal: list[int] = ideal_values.copy()

    logger.info(f"{len(used_circs)} circuits were used to calculate the averaged fidelity.")
    if noise_model:
        logger.info("A noise model was used.")
    for i, circ in enumerate(used_circs):
        if noise_model:
            job : PrimitiveJob = estimator.run(circ, 'IIZ', noise_model=noise_model, job_name = f'Circuit_{i}')
        else:
            job : PrimitiveJob = estimator.run(circ, 'IIZ', job_name = f'Circuit_{i}')
        res = job.result()
        logger.info(f"Got result from job {job}.")
        expectation_value =  res.values[0]
        exps += [1 - np.abs(expectation_value - used_ideal[i]) / 2]

    return exps

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
    
    if 'used_circs_indices' in kwargs:
        indices = kwargs['used_circs_indices']
        used_circs: list[QuantumCircuit] = [aux_circs[i] for i in indices]
        if do_continous_evaluation:
            used_circs_eval: list[QuantumCircuit] = [aux_circs_eval[i] for i in indices]
    else:
        if mini_batch:
            num_circs_per_group = min(mini_batch, 4)
            inds = random.sample([0,1,2,3], num_circs_per_group)
            used_circs: list[QuantumCircuit] = []
            for i in inds:
                used_circs += [aux_circs[i],aux_circs[i+4],aux_circs[i+8]]
                if do_continous_evaluation:
                    used_circs_eval += [aux_circs_eval[i],aux_circs_eval[i+4],aux_circs_eval[i+8]]
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
            # res = job.result()
            # try:
            #     res = job.result()
            #     logger.info(f"Got result from job {job}.")
            # except (MaxRetryError, HTTPError, ConnectionError) as e:
            #     logger.warning(f"Max retries {e} was raised, probably due to losing connection to internet. Sleeping for 10s and trying again.")
            #     time.sleep(10)
            #     res = job.result()
            # except Exception as e:
            #     logger.error(f"Getting the results for a cost function evaluation gave the exception: {type(e)}")
            #     return None
            # counts =  res.get_counts()
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
            logger.info(f"The fidelity between the simulated state and the executed statistics is: {evaluation_fid}")

    return fids

def evaluate_final_state_fidelity_no_compilation(
        param_dict: dict[Parameter, float], 
        connector: UmzConnector, 
        gate_seqs: list[OperationSequence],
        backend: str = 'simulator',
        num_shots: int = 200,
        mini_batch: int | None = None,
        **kwargs):
    """
    Calculate the fidelity of each gate sequence without compilation.

    Parameters
    ----------
    param_dict : dict[Parameter, float] 
        A dictionary mapping Parameters to numerical values for binding parameters in the gate sequences.
    connector : UmzConnector 
        An instance of UmzConnector used to submit circuits for execution.
    gate_seqs : list[OperationSequence] 
        A list of OperationSequence objects representing gate sequences.
    backend : str, optional
        The backend to execute the gate sequences on. Defaults to 'simulator'.
    num_shots : int, optional 
        The number of shots for each circuit execution. Defaults to 200.
    mini_batch : int, optional 
        The size of mini-batch to use for evaluating gate sequences. Defaults to None.
    **kwargs : dict 
        Additional keyword arguments.

    Returns
    ----------
    list[float] 
        A list of fidelity values for each evaluated gate sequence.

    Description
    -----------
    This function calculates the fidelity of each gate sequence without compilation. It submits each gate sequence 
    for execution to the specified backend using the UmzConnector. If a mini-batch size is specified and non-zero, 
    a subset of gate sequences is selected from the input gate sequences. Each selected gate sequence is then executed 
    on the specified backend with the given number of shots. The fidelity of each executed gate sequence is calculated 
    based on the resulting measurement statistics, compared to the target state |0⟩.

    The fidelity is calculated using state fidelity, where the resulting state is compared to the target state |0⟩.
    """

    fids = []
    job_ids = []
    if 'used_circs_indices' in kwargs:
        indices = kwargs['used_circs_indices']
        used_seqs: list[OperationSequence] = [gate_seqs[i] for i in indices]
    else:
        if mini_batch:
            num_circs_per_group = min(mini_batch, 4)
            inds = random.sample([0,1,2,3], num_circs_per_group)
            used_seqs: list[OperationSequence] = []
            for i in inds:
                used_seqs += [gate_seqs[i],gate_seqs[i+4],gate_seqs[i+8]]
        else:
            used_seqs: list[OperationSequence] = gate_seqs.copy()
    
    logger.info(f"{len(used_seqs)} sequences were used to calculate the averaged fidelity.")

    for i, gate_seq in enumerate(used_seqs):
        circ_code = gate_seq.serialize()
        circ_id = connector.submit_circuit(code=circ_code, 
                                         kind="op_seq",
                                         backend=backend,
                                         name=f"Circuit_{i}")
        job_id = connector.submit_job(name=f'Circuit_{i}',
                                      backend=backend,
                                      source='Qiskit',
                                      circuit=circ_id,
                                      execution_settings = {'parameters': param_dict}, 
                                      shots = num_shots)
        job_ids += [job_id]

    for job_id in job_ids:
        completed_job = connector.wait_for_completed_job(handle=job_id)
        result = completed_job['result']['result']
        counts = result[0]['measurements']
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
        state = np.dot([np.array([1,0]), np.array([0,1])], np.sqrt(frequencies))
        # The target state is always the zero state.
        fid = state_fidelity(state, zero)
        fids.append(fid)

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
    def averaged_fidelity(ansatz: QuantumCircuit, parameters: list[Num], num_shots: int, init_states: list[QuantumCircuit] | None, final_rotations: list[QuantumCircuit] | None, **kwargs):
        param_dict = {ansatz_parameter : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}

        if init_states is None:
            init_states = init_states_complete.copy()

        if 'used_circs_indices' in kwargs:
            fids = evaluate_final_state_fidelity(param_dict, backend, ansatz, init_states, final_rotations,  
                                                 num_shots, mini_batch, used_circs_indices = kwargs['used_circs_indices'], 
                                                 noise_model = noise_model,  use_tomo = use_tomo,
                                                 do_continous_evaluation = do_continous_evaluation,
                                                 skip_compilation = skip_compilation)
        else:
            fids = evaluate_final_state_fidelity(param_dict, backend, ansatz, init_states, final_rotations, 
                                                 num_shots, mini_batch, noise_model = noise_model, use_tomo = use_tomo,
                                                 do_continous_evaluation = do_continous_evaluation,
                                                 skip_compilation = skip_compilation)

        return -np.average(fids)
    
    return averaged_fidelity

def create_av_fidelity_no_compilation(backend: str, connector: UmzConnector, mini_batch: int | None = None):
    """
    Create a function to calculate the average fidelity of gate sequences without compilation.

    Parameters
    ----------
    backend : str 
        The backend to execute the gate sequences on.
    connector : UmzConnector 
        An instance of UmzConnector used to submit circuits for execution.
    mini_batch : int
        The size of mini-batch to use for evaluating gate sequences.

    Returns
    ----------
    function
        A function that calculates the average fidelity of gate sequences without compilation.

    Description
    -----------
    This function creates a closure that generates another function `averaged_fidelity`. The `averaged_fidelity` function 
    calculates the average fidelity of gate sequences without compilation. It utilizes the `evaluate_final_state_fidelity_no_compilation` 
    function to evaluate fidelity for each gate sequence, using the provided parameters, connector, and backend. The average fidelity 
    is computed as the negative average of the fidelity values obtained from `evaluate_final_state_fidelity_no_compilation`.
    """
    def averaged_fidelity(ansatz: QuantumCircuit, parameters: list[Num], gate_seqs: list[OperationSequence], num_shots: int, **kwargs):
        param_dict = {ansatz_parameter.name : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}

        if 'used_circs_indices' in kwargs:
            fids = evaluate_final_state_fidelity_no_compilation(param_dict, connector, gate_seqs, backend, num_shots, mini_batch, used_circs_indices = kwargs['used_circs_indices'])
        else:
            fids = evaluate_final_state_fidelity_no_compilation(param_dict, connector, gate_seqs, backend, num_shots, mini_batch)

        return -np.average(fids)
    
    return averaged_fidelity

def create_stochastic_av_fidelity(backend: Backend, 
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

def create_av_expectation(backend: Backend, 
                       mini_batch: int | None = None, 
                       noise_model: NoiseModel | None = None):
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
    
    Returns
    ----------
    function
        A function that calculates the average expectation of circuits with a given ansatz.

    Description
    -----------
    This function creates a closure that generates another function `averaged_fidelity`. The `averaged_fidelity` function 
    calculates the average fidelity of circuits with a given ansatz. It utilizes the `evaluate_final_state_expectation` function to 
    evaluate expectation for each circuit, using the provided ansatz, parameters, initial states, and backend. The average expectation 
    is computed as 1 minus the values obtained from `evaluate_final_state_expectation`.
    """
    def averaged_expectation(ansatz: QuantumCircuit, parameters: list[Num], num_shots: int, init_states: list[QuantumCircuit] | None, **kwargs):
        param_dict = {ansatz_parameter : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}

        if init_states is None:
            init_states = init_states_complete.copy()

        if 'used_circs_indices' in kwargs: 
            exps = evaluate_final_state_expectation(param_dict, backend, ansatz, init_states, num_shots, mini_batch,
                                             used_circs_indices = kwargs['used_circs_indices'], noise_model = noise_model)
        else:
            exps = evaluate_final_state_expectation(param_dict, backend, ansatz, init_states, num_shots, mini_batch,
                                                    noise_model=noise_model)
            
        return -np.average(exps)
    
    return averaged_expectation

def create_av_expectation_no_compilation(backend: str, connector: UmzConnector, mini_batch: int | None = None):
    """
    Create a function to calculate the average expectation of gate sequences without compilation.

    Parameters
    ----------
    backend : str 
        The backend to execute the gate sequences on.
    connector : UmzConnector 
        An instance of UmzConnector used to submit circuits for execution.
    mini_batch : int
        The size of mini-batch to use for evaluating gate sequences.

    Returns
    ----------
    function
        A function that calculates the average expectation of gate sequences without compilation.

    Description
    -----------
    This function creates a closure that generates another function `averaged_expectation`. The `averaged_expectation` function 
    calculates the average expectation of gate sequences without compilation. It utilizes the `evaluate_final_state_expectation_no_compilation` 
    function to evaluate expectation for each gate sequence, using the provided parameters, connector, and backend. The average expectation 
    is computed as the negative average of the expectation values obtained from `evaluate_final_state_expectation_no_compilation`.
    """

    raise NotImplementedError("Not implemented yet.")

    def averaged_expectation(ansatz: QuantumCircuit, parameters: list[Num], gate_seqs: list[OperationSequence], num_shots: int, **kwargs):
        param_dict = {ansatz_parameter.name : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}

        if 'used_circs_indices' in kwargs:
            fids = evaluate_final_state_expectation_no_compilation(param_dict, connector, gate_seqs, backend, num_shots, mini_batch, used_circs_indices = kwargs['used_circs_indices'])
        else:
            fids = evaluate_final_state_expectation_no_compilation(param_dict, connector, gate_seqs, backend, num_shots, mini_batch)

        return -np.average(fids)
    
    return averaged_expectation

def create_stochastic_av_expectation(backend: Backend, 
                                  mini_batch: int | None = None,  
                                  noise_model: NoiseModel | None = None):
    """
    Create a function to calculate the average expectation of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    mini_batch : int
        The size of mini-batch to use for evaluating circuits.
    noise_model : NoiseModel
        Optional NoiseModel to be used in the simulation results.

    Returns
    ----------
    function
        A function that calculates the average expectation of circuits with a given ansatz.

    Description
    -----------
    This function creates a closure that generates another function `averaged_expectation`. The `averaged_expectation` function 
    calculates the average expectation of circuits with a given ansatz. It utilizes the `evaluate_final_state_expectation` function to 
    evaluate expectation for each circuit, using the provided ansatz, parameters, initial states, and backend. The average expectation 
    is computed as the negative average of the expectation values obtained from `evaluate_final_state_expectation`.
    """
    def averaged_expectation(ansatz: QuantumCircuit, parameters: list[Num], num_shots: int, init_states: list[QuantumCircuit] | None, **kwargs):
        param_dict = {ansatz_parameter : parameters[i] for i, ansatz_parameter in enumerate(ansatz.parameters)}

        if init_states is None:
            init_states = init_states_complete.copy()

        used_circs_indices = kwargs['used_circs_indices'] 
        exps = evaluate_final_state_expectation(param_dict, backend, ansatz, init_states, num_shots, 
                                             used_circs_indices = used_circs_indices, noise_model = noise_model)
            
        return -np.average(exps)
    
    return averaged_expectation
