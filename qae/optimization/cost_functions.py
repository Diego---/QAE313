import random

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

Num = float | int

from umz_sequence_generator.sequence_generator import OperationSequence
from umz_backend_connector.umz_connector import UmzConnector

from qae.circuits.circuit_constants import init_states_complete, ansatz3
from qae.circuits.fidelity_measurement import (
create_av_fidelity, create_av_fidelity_no_compilation, create_stochastic_av_fidelity,
create_av_expectation, create_stochastic_av_expectation
)


def create_cost_func(backend: Backend, 
                     ansatz: QuantumCircuit | None = None, 
                     num_shots: int = 200, 
                     init_states: list[QuantumCircuit] | None = None, 
                     final_rotations: list[QuantumCircuit] | None = None,
                     mini_batch: int | None = None,
                     use_expected_values: bool = False,
                     use_tomo: bool = False,
                     do_continous_evaluation: bool = False,
                     skip_compilation: bool = False):
    """
    Create a cost function to evaluate the fidelity of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    ansatz : QuantumCircuit, optional 
        The QuantumCircuit object representing the ansatz circuit. Defaults to ansatz3.
    num_shots : int 
        The number of shots for each circuit execution.
    init_states : list[QuantumCircuit], optional 
        A list of QuantumCircuit objects representing initial states. Defaults to init_states_complete.
    final_rotations : list[QuantumCircuit] 
        A list of QuantumCircuit objects representing the final rotations required by the initial states.
    mini_batch : int, optional
        The size of mini-batch to use for evaluating circuits. Defaults to None.
    use_expected_values : bool
        Whether to use the expectation value of Z for the final quantity instead of fidelity.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    do_continous_evaluation : bool
        Whether every time the fidelity of the resulting measurement statistics are close to the ideal simulation.
    skip_compilation : bool
        Whether you want to skip the compilation stage. Useful for the red trap.

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
    
    if ansatz is None:
        ansatz = ansatz3.copy()
    
    if init_states is None:
        init_states = init_states_complete.copy()
    
    if use_expected_values:
        av_exp = create_av_expectation(backend, mini_batch)

        def cost(params: list[Num], **kwargs):
            if 'used_circs_indices' in kwargs:
                return av_exp(ansatz, params, num_shots, init_states, used_circs_indices = kwargs['used_circs_indices'])
            else:
                return av_exp(ansatz, params, num_shots, init_states)
            
        return cost
    
    av_fid = create_av_fidelity(backend, mini_batch, use_tomo=use_tomo, do_continous_evaluation=do_continous_evaluation, skip_compilation=skip_compilation)

    def cost(params: list[Num], **kwargs):

        if 'used_circs_indices' in kwargs:
            return av_fid(ansatz, params, num_shots, init_states, final_rotations, used_circs_indices = kwargs['used_circs_indices'])
        else:
            return av_fid(ansatz, params, num_shots, init_states, final_rotations)

    return cost

def create_stochastic_cost_func(backend: Backend, 
                     ansatz: QuantumCircuit | None = None, 
                     num_shots: int = 200, 
                     init_states: list[QuantumCircuit] | None = None, 
                     final_rotations: list[QuantumCircuit] | None = None,
                     mini_batch: int | None = None,
                     use_expected_values: bool = False,
                     use_tomo: bool = False,
                     do_continous_evaluation: bool = False):
    """
    Create a cost function to evaluate the fidelity of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    ansatz : QuantumCircuit, optional 
        The QuantumCircuit object representing the ansatz circuit. Defaults to ansatz3.
    num_shots : int 
        The number of shots for each circuit execution.
    init_states : list[QuantumCircuit], optional 
        A list of QuantumCircuit objects representing initial states. Defaults to init_states_complete.
    final_rotations : list[QuantumCircuit] 
        A list of QuantumCircuit objects representing the final rotations required by the initial states.
    mini_batch : int, optional
        The size of mini-batch to use for evaluating circuits. Defaults to None.
    use_expected_values : bool
        Whether to use the expectation value of Z for the final quantity instead of fidelity.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    do_continous_evaluation : bool
        Whether every time the fidelity of the resulting measurement statistics are close to the ideal simulation.

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
    
    if ansatz is None:
        ansatz = ansatz3.copy()
    
    if init_states is None:
        init_states = init_states_complete.copy()
    
    if use_expected_values:
        av_exp = create_stochastic_av_expectation(backend)

        def cost(params: list[Num]):
            if mini_batch is None or min(mini_batch, 4) == 4:
                used_circs_indices = list(range(len(init_states)))
            else:
                num_circs_per_group = min(mini_batch, 4)
                inds = random.sample([0,1,2,3], num_circs_per_group)
                used_circs_indices = list(inds)
                for i in inds:
                    used_circs_indices += [i+4, i+8]
            return av_exp(ansatz, params, num_shots, init_states, used_circs_indices = used_circs_indices)

        return cost

    av_fid = create_stochastic_av_fidelity(backend, use_tomo=use_tomo, do_continous_evaluation=do_continous_evaluation)

    def cost(params: list[Num]):
        if mini_batch is None or min(mini_batch, 4) == 4:
            used_circs_indices = list(range(len(init_states)))
        else:
            num_circs_per_group = min(mini_batch, 4)
            inds = random.sample([0,1,2,3], num_circs_per_group)
            used_circs_indices = list(inds)
            for i in inds:
                used_circs_indices += [i+4, i+8]
        return av_fid(ansatz, params, num_shots, init_states, final_rotations, used_circs_indices = used_circs_indices)

    return cost

def create_cost_func_no_compilation(backend: str, 
                                    connector: UmzConnector,
                                    gate_seqs: list[OperationSequence], 
                                    ansatz: QuantumCircuit | None = None, 
                                    num_shots = 200, 
                                    mini_batch: int | None = None):
    """
    Create a cost function to evaluate the fidelity of gate sequences without compilation.

    Parameters
    ----------
    backend : str 
        The backend to execute the gate sequences on.
    connector : UmzConnector 
        An instance of UmzConnector used to submit circuits for execution.
    gate_seqs : list[OperationSequence] 
        A list of OperationSequence objects representing gate sequences.
    ansatz : QuantumCircuit, optional 
        The QuantumCircuit object representing the ansatz circuit. Defaults to ansatz3.
    num_shots : int 
        The number of shots for each circuit execution.
    mini_batch : int, optional
        The size of mini-batch to use for evaluating gate sequences. Defaults to None.

    Returns
    ----------
    function
        A cost function that evaluates the fidelity of gate sequences without compilation.

    Description
    -----------
    This function creates a closure that generates another function `cost`. The `cost` function calculates the cost 
    associated with the average fidelity of gate sequences without compilation. It utilizes the `create_av_fidelity_no_compilation` 
    function to create a function `av_fid`, which calculates the average fidelity. The `cost` function then evaluates the average 
    fidelity using the provided parameters, connector, and backend. The cost represents the negative 
    average fidelity.
    """
    
    if ansatz is None:
        ansatz = ansatz3.copy()
    
    av_fid = create_av_fidelity_no_compilation(backend, connector, mini_batch)

    def cost(params: list[Num], **kwargs):

        if 'used_circs_indices' in kwargs:
            return av_fid(ansatz, params, gate_seqs, num_shots, used_circs_indices = kwargs['used_circs_indices'])
        else:
            return av_fid(ansatz, params, gate_seqs, num_shots)
        
    return cost

def create_noisy_cost_func(backend: Backend,
                     noise_model: NoiseModel, 
                     ansatz: QuantumCircuit | None = None, 
                     num_shots: int = 200, 
                     init_states: list[QuantumCircuit] | None = None, 
                     final_rotations: list[QuantumCircuit] | None = None,
                     mini_batch: int | None = None,
                     use_tomo: bool = False,
                     do_continous_evaluation: bool = False):
    """
    Create a cost function to evaluate the fidelity of circuits with a given ansatz.

    Parameters
    ----------
    backend : Backend 
        The backend to execute the circuits on.
    noise_model : NoiseModel
        NoiseModel to be used in the simulation results.
    ansatz : QuantumCircuit, optional 
        The QuantumCircuit object representing the ansatz circuit. Defaults to ansatz3.
    num_shots : int 
        The number of shots for each circuit execution.
    init_states : list[QuantumCircuit], optional 
        A list of QuantumCircuit objects representing initial states. Defaults to init_states_complete.
    final_rotations : list[QuantumCircuit] 
        A list of QuantumCircuit objects representing the final rotations required by the initial states.
    mini_batch : int, optional
        The size of mini-batch to use for evaluating circuits. Defaults to None.
    use_tomo : bool
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    do_continous_evaluation : bool
        Whether every time the fidelity of the resulting measurement statistics are close to the ideal simulation.

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

    if ansatz is None:
        ansatz = ansatz3.copy()
    
    if init_states is None:
        init_states = init_states_complete.copy()
    
    if mini_batch and mini_batch < 4:
        av_fid = create_stochastic_av_fidelity(backend, mini_batch, noise_model, use_tomo, do_continous_evaluation)
    else:
        av_fid = create_av_fidelity(backend, mini_batch, noise_model, use_tomo, do_continous_evaluation)

    def cost(params: list[Num]):
        if mini_batch is None or min(mini_batch, 4) == 4:
            used_circs_indices = list(range(len(init_states)))
        else:
            num_circs_per_group = min(mini_batch, 4)
            inds = random.sample([0,1,2,3], num_circs_per_group)
            used_circs_indices = list(inds)
            for i in inds:
                used_circs_indices += [i+4, i+8]
        return av_fid(ansatz, params, num_shots, init_states, final_rotations, used_circs_indices = used_circs_indices)

    return cost
