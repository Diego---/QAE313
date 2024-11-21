# -*- coding: utf-8 -*-
"""
Created on Thursday Apr 15 2024

@author: Diego Alberto Olvera Millán
"""

import random
import numpy as np

Num = float | int

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate, RYGate, HGate

from qae.circuits.circuit_constants import ansatz3, init_states_complete, circ_labels

def build_aux_circs(
        ansatz: QuantumCircuit | None = None, 
        init_states: list[QuantumCircuit] | None = None, 
        final_rotations: list[QuantumCircuit] | None = None, 
        encoding_circ: QuantumCircuit | None = None,
        error_list: list[int] | None = None,
        param_dict: dict[Parameter, float] | None = None,
        use_resets: bool = False,
        use_measurement: bool = True
        ) -> list[QuantumCircuit]:
    """
    Build auxiliary circuits used for fidelity calculations.

    Parameters
    ----------
    ansatz : QuantumCircuit 
        The QuantumCircuit object representing the ansatz circuit.
    init_states : list[QuantumCircuit], optional 
        A list of QuantumCircuit objects representing initial states.
    final_rotations : list[QuantumCircuit], optional
        A list of QuantumCirctuit objects that represents the final rotations.
    encoding_circ: QuantumCircuit, optional
        The type of circuit used for encoding/decoding the initial state into the codespace.
    error_list : list[int], optional
        List of errors to be used.
    param_dict : dict[Parameter, Num] 
        A dictionary mapping Parameters to numerical values for binding parameters in the ansatz circuit.
    use_resets : bool
        Whether to perform a final reset on qubits 1 and 2. Defaults to false.
    use_measurement : bool
        Whether to have a measurement. It's not neccessary to have them when using Estimators. Defaults to True.

    Returns:
    ----------
    list[QuantumCircuit] 
        A list of QuantumCircuit objects representing the auxiliary circuits.

    Description:
    This function constructs auxiliary circuits for fidelity calculations. It combines the initial state circuits
    with the ansatz circuit, incorporating resets and measurements on the first qubit. Additionally, it applies
    a final rotation to bring the resulting state to the |0⟩ state, facilitating comparison with the target state.

    The `measurement_circ` circuit contains a barrier, can reset qubits 1 and 2 and measure qubit 0.
    """
    measurement_circ = QuantumCircuit(3,1)
    measurement_circ.barrier()
    if use_resets:
        measurement_circ.reset([1,2])
    if use_measurement:
        measurement_circ.measure(0,0)

    if ansatz is None:
        ansatz_copy = ansatz3.copy()
    else:  
        ansatz_copy = ansatz.copy()
    
    if param_dict:
        ansatz_instantiated = ansatz_copy.assign_parameters(param_dict)
    else:
        ansatz_instantiated = ansatz_copy

    if encoding_circ:
        init0 = QuantumCircuit(3)
        init0.compose(encoding_circ, inplace = True)
        # init0.barrier()
        init1 = QuantumCircuit(3)
        init1.x(0)
        init1.barrier()
        init1.compose(encoding_circ, inplace = True)
        # init1.barrier()
        init_p = QuantumCircuit(3)
        init_p.h(0)
        init_p.barrier()
        init_p.compose(encoding_circ, inplace = True)
        # init_p.barrier()
        
        error0 = QuantumCircuit(3)
        error1 = QuantumCircuit(3)
        error2 = QuantumCircuit(3)
        error0.x(0)
        error1.x(1)
        error2.x(2)
        error0.barrier()
        error1.barrier()
        error2.barrier()
        
        init_states = []
        
        for initial_code_state in [init0, init1, init_p]:
            for error in [QuantumCircuit(3), error0, error1, error2]:
                temp = initial_code_state.compose(error, inplace = False)
                init_states.append(temp)
                
    else:
        if init_states is None:
            init_states = init_states_complete.copy()
    
    aux_circs = []
    for i in range(len(init_states)):
        # Compose the initial state circuit with the ansatz circuit, which itself is composed with the resets and the measurement on the first qubit.
        # We also add a final rotation which should take the target state to the 0 state. For example, if the input state is the logical + state, 
        # before the measurement we add an H gate, which would take the single qubit |+> state to |0> (the |-> state would be taken to |1>, and we'd see
        # different statistics). 
        # This is done so that in the end we can use only the measurement statistics to compare the resulting state with a target state, which should 
        # always be the 0 state.
        if final_rotations:
            assert len(final_rotations) == len(init_states), "There must be a final rotation provided for each initial state."
            final_rotations_copy = final_rotations[i].copy()
            final_rotations_copy.barrier()
            aux_circ = init_states[i].compose(ansatz_instantiated.compose(final_rotations_copy.compose(measurement_circ)))
            aux_circs += [aux_circ]
        else:
            final_rotation = QuantumCircuit(3)
            final_rotation.barrier()
            if circ_labels[i] in ['1', '1err0', '1err1', '1err2']:
                final_rotation.x(0)
            elif circ_labels[i] in ['+', '+err0', '+err1', '+err2']:
                final_rotation.h(0)
            aux_circ = init_states[i].compose(ansatz_instantiated.compose(final_rotation.compose(measurement_circ)))
            aux_circs += [aux_circ]
    return aux_circs

def build_tomo_circs(
    circ: QuantumCircuit,
    induce_error: bool = True,
    encoding: bool | QuantumCircuit = True,
    re_encoding: bool | QuantumCircuit = False,
    measurement: QuantumCircuit | None = None
    ):
    """
    Build circuits for which tomography will be done from a given ansatz.

    Parameters
    ----------
    circ : QuantumCircuit
        The input quantum circuit that represents the ansatz for which circuits will be constructed.
    induce_error : bool, optional
        Flag indicating whether qubit flip error should be introduced artificially. Defaults to True.
    encoding : bool | QuantumCircuit, optional
        Flag indicating whether the initial states should be encoded into the three-qubit repetition code.
        Defaults to true.
    re_encoding : bool | QuantumCircuit, optional
        Flag indicating whether to include re-encoding circuits. Defaults to False.
    measurement : QuantumCircuit, optional
        Whether to measure circuits. Must be passed as a circuit which has the measurements that you're interested in.

    Returns
    -------
    Tuple[List[QuantumCircuit], QuantumCircuit]
        A tuple containing a list of tomography circuits and a random state circuit.

    Description
    -----------
    This function constructs circuits on which tomography will be performed. It optionally includes re-encoding
    circuits if the `re_encoding` flag is set to True.  Additionally, a random state circuit is generated for reference.

    If `re_encoding` is set to True, the re-encoding circuit will be constructed by applying controlled-X gates and
    barriers to the reset qubits. Otherwise, only reset operations are applied to the reset qubits.

    The random state circuit is generated by applying a sequence of random single-qubit gates followed by controlled-X
    gates to create an encoded state.

    Returns a tuple containing the list of tomography circuits and the random state circuit.
    """

    if isinstance(re_encoding, QuantumCircuit):
        reset_circ = QuantumCircuit(3)
        reset_circ.barrier()
        reset_circ.reset([1,2])
        reset_circ.barrier()
        re_encoding_circ = reset_circ.compose(re_encoding.copy())
    elif re_encoding:
        re_encoding_circ = QuantumCircuit(3)
        re_encoding_circ.barrier()
        re_encoding_circ.reset([1,2])
        re_encoding_circ.cx(0,1)
        re_encoding_circ.cx(0,2)
        re_encoding_circ.barrier()
    else:
        re_encoding_circ = QuantumCircuit(3)
        re_encoding_circ.barrier()

    num_qubits = circ.num_qubits

    aux_circ = QuantumCircuit(num_qubits)
    for instruction in circ.data:
        if instruction.operation.name not in ['measure', 'Measure']:
            aux_circ.append(instruction)

    target_random = QuantumCircuit(1)
    random_state = QuantumCircuit(num_qubits)
    random_state_err0 = QuantumCircuit(num_qubits)
    random_state_err1 = QuantumCircuit(num_qubits)
    random_state_err2 = QuantumCircuit(num_qubits)
    random_state_err = [random_state_err0, random_state_err1, random_state_err2]
    gates_to_apply = [HGate, RYGate, RXGate]

    for i in range(3):
        gate = random.choice(gates_to_apply)
        if gate in [RYGate, RXGate]:
            rand_angle = random.uniform(-2*np.pi, 2*np.pi,)
            random_state.append(gate(rand_angle),[0])
            target_random.append(gate(rand_angle),[0])
            for i in range(3):
                random_state_err[i].append(gate(rand_angle), [0])
        else:
            random_state.append(HGate(), [0])
            target_random.append(HGate(), [0])
            for i in range(3):
                random_state_err[i].append(HGate(), [0])
                
    random_state.barrier()
    for state in random_state_err:
        state.barrier()
    
    if encoding:
        if isinstance(encoding, QuantumCircuit):
            random_state.compose(encoding, inplace = True)
            for i in range(3):
                random_state_err[i].compose(encoding, inplace = True)
                if induce_error:
                    # random_state_err[i].barrier()
                    random_state_err[i].x(i)
                    random_state_err[i].barrier()
        else:
            random_state.cx(0,1)
            random_state.cx(0,2)
            for i in range(3):
                random_state_err[i].cx(0,1)
                random_state_err[i].cx(0,2)
                if induce_error:
                    random_state_err[i].barrier()
                    random_state_err[i].x(i)
                    random_state_err[i].barrier()
    
    tomo_circs = []
    
    if isinstance(encoding, QuantumCircuit):
        init0 = QuantumCircuit(3)
        init0.compose(encoding, inplace = True)
        # init0.barrier()
        init1 = QuantumCircuit(3)
        init1.x(0)
        init1.barrier()
        init1.compose(encoding, inplace = True)
        # init1.barrier()
        init_p = QuantumCircuit(3)
        init_p.h(0)
        init_p.barrier()
        init_p.compose(encoding, inplace = True)
        # init_p.barrier()
        
        error0 = QuantumCircuit(3)
        error1 = QuantumCircuit(3)
        error2 = QuantumCircuit(3)
        error0.x(0)
        error1.x(1)
        error2.x(2)
        error0.barrier()
        error1.barrier()
        error2.barrier()
        
        init_states = []
        
        for initial_code_state in [init0, init1, init_p]:
            for error in [QuantumCircuit(3), error0, error1, error2]:
                temp = initial_code_state.compose(error, inplace = False)
                init_states.append(temp)
    else:
        init_states = init_states_complete.copy()
    
    for i, init_circ in enumerate(init_states + [random_state] + random_state_err):
        if isinstance(measurement, QuantumCircuit):
            composed_circ = init_circ.compose(aux_circ.compose(re_encoding_circ.compose(measurement)))
        else:
            composed_circ = init_circ.compose(aux_circ.compose(re_encoding_circ))
        tomo_circs += [composed_circ]
    
    return tomo_circs, random_state, target_random

def create_count_taking_circs(
        circ: QuantumCircuit, 
        num_rand_states: int = 5, 
        random_angles: list[list[Num]] = None, 
        re_encoding: bool = True,
        do_resets: bool = False):
    """
    Build circuits to get the measurement results from circuits with initial states |0>, |1>, |+>, and some random states
    plus all possible 1 qubit flips.

    Parameters
    ----------
    circ : QuantumCircuit
        The input quantum circuit that represents the ansatz for which circuits will be constructed.
    num_rand_states : int, optional
        Number of random states tu be used. Defaults to 5.
    random_angles: list[list[Num]]
        List of random angles that produce the random circuits. Used for reproducibility.
    re_encoding : bool
        Flag indicating whether to include re-encoding circuits. Defaults to True.
    do_resets : bool
        Flag indicating whether to use resets. Defaults to True.

    Returns
    -------
    Tuple[list[QuantumCircuit], list[QuantumCircuit], list[Num]]
        A tuple containing a list of the circuits to be run, the random circuits and the random angles.
    """

    if re_encoding:
        re_encoding_circ = QuantumCircuit(3)
        re_encoding_circ.barrier()
        re_encoding_circ.reset([1,2])
        re_encoding_circ.cx(0,1)
        re_encoding_circ.cx(0,2)
        re_encoding_circ.barrier()

        meas_circ = QuantumCircuit(3,3)
        for i in range(3):
            meas_circ.measure(i,i)
    else:

        meas_circ = QuantumCircuit(3,1)
        if do_resets:
            meas_circ.barrier()
            meas_circ.reset([1,2])
        meas_circ.measure(0,0)

    num_qubits = circ.num_qubits

    aux_circ = QuantumCircuit(num_qubits)
    for instruction in circ.data:
        if instruction.operation.name not in ['measure', 'Measure']:
            aux_circ.append(instruction)

    validation_circs = []
    for i, init_circ in enumerate(init_states_complete):
        if re_encoding:
            composed_circ = init_circ.compose(aux_circ.compose(re_encoding_circ))
        else:
            composed_circ = init_circ.compose(aux_circ)
        validation_circs += [composed_circ]

    random_states = []
    if not random_angles:
        random_states_angles = []
    else:
        random_states_angles = random_angles.copy()
        num_rand_states = len(random_angles)    
    for i in range(num_rand_states):

        random_state = QuantumCircuit(num_qubits)
        random_states.append(random_state)

        random_state_err0 = QuantumCircuit(num_qubits)
        random_state_err1 = QuantumCircuit(num_qubits)
        random_state_err2 = QuantumCircuit(num_qubits)
        random_state_err = [random_state_err0, random_state_err1, random_state_err2]

        if not random_angles:
            rand_angle1 = random.uniform(-2*np.pi, 2*np.pi)
            rand_angle2 = random.uniform(-2*np.pi, 2*np.pi)
            random_states_angles += [[rand_angle1, rand_angle2]]
        else:
            rand_angle1, rand_angle2 = random_states_angles[i]
        random_state.rx(rand_angle1, 0)
        random_state.ry(rand_angle2, 0)

        for i in range(3):
            random_state_err[i].rx(rand_angle1, 0)
            random_state_err[i].ry(rand_angle2, 0)

        random_state.cx(0,1)
        random_state.cx(0,2)
        for i in range(3):
            random_state_err[i].cx(0,1)
            random_state_err[i].cx(0,2)
            random_state_err[i].x(i)
            random_state_err[i].barrier()

        for i, init_circ in enumerate([random_state] + random_state_err):
            if re_encoding:
                composed_circ = init_circ.compose(aux_circ.compose(re_encoding_circ))
            else:
                composed_circ = init_circ.compose(aux_circ)
            validation_circs += [composed_circ]

    for circ in validation_circs:
        circ.compose(meas_circ, inplace=True)    
    return validation_circs, random_states, random_states_angles
