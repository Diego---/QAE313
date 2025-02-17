import random
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

from qae.circuits.circuit_constants import init_states_complete, ansatz3
from qae.circuits.fidelity_measurement import (
create_av_fidelity, create_stochastic_av_fidelity,
)

Num = float | int

def create_cost_func(backend: Backend, 
                     ansatz: QuantumCircuit | None = None, 
                     num_shots: int = 200, 
                     init_states: list[QuantumCircuit] | None = None, 
                     final_rotations: list[QuantumCircuit] | None = None,
                     mini_batch: int | None = None,
                     use_tomo: bool = False,
                     do_continous_evaluation: bool = False,
                     skip_compilation: bool = False) -> Callable:
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
        The number of shots for each circuit executio, optionaln.
    init_states : list[QuantumCircuit], optional 
        A list of QuantumCircuit objects representing initial states. Defaults to init_states_complete.
    final_rotations : list[QuantumCircuit] , optional
        A list of QuantumCircuit objects representing the final rotations required by the initial states.
    mini_batch : int, optional
        The size of mini-batch to use for evaluating circuits. Defaults to None.
    use_tomo : bool, optional
        Whether to do state tomography on the resulting one qubit state or just use the measurement counts.
    do_continous_evaluation : bool, optional
        Whether every time the fidelity of the resulting measurement statistics are close to the ideal simulation.
    skip_compilation : bool, optional
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
    
    if init_states is None:
        init_states = init_states_complete.copy()
    
    av_fid = create_av_fidelity(backend, mini_batch, use_tomo=use_tomo, do_continous_evaluation=do_continous_evaluation, skip_compilation=skip_compilation)
    if ansatz is None:   
        def cost(params: list[Num], ansatz: QuantumCircuit, **kwargs):
            used_circs_indices = kwargs.get('used_circs_indices')
            return av_fid(ansatz, params, num_shots, init_states, final_rotations, used_circs_indices = used_circs_indices)
        
        return cost
        
    def cost(params: list[Num], **kwargs):
        used_circs_indices = kwargs.get('used_circs_indices')
        return av_fid(ansatz, params, num_shots, init_states, final_rotations, used_circs_indices = used_circs_indices)

    return cost

def create_stochastic_cost_func(backend: Backend, 
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
        av_fid = create_stochastic_av_fidelity(backend, noise_model, use_tomo, do_continous_evaluation)
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
