import random
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit.providers import Backend
from qiskit.circuit.library import RXGate, RYGate, HGate
from qiskit_aer import Aer
from qiskit_experiments.library import StateTomography

from qae.circuits.circuit_building import build_tomo_circs
from qae.circuits.circuit_constants import (
    one, zero, plus, ooo, ooz, ozo, zoo, zzz, zzo, zoz, ozz, plus_state,
    plus_err0, plus_err1, plus_err2
)

def evaluate_circuit(circ: QuantumCircuit) -> tuple[dict[str, float], list[QuantumCircuit]]:
    """
    Evaluate the fidelity of various states prepared by the given quantum circuit.

    Parameters
    ----------
    circ : QuantumCircuit
        The QuantumCircuit object to evaluate.
    backend : Backend, optional
        The backend to execute the circuits on. If None, defaults to the QASM simulator.

    Returns
    ----------
    dict
        A dictionary containing fidelity values for different prepared states.
    list
        A list containing the randomly generated quantum states.

    Description
    -----------
    This function evaluates the fidelity of various states prepared by the given quantum circuit. It prepares several initial 
    states, applies the given circuit to each of them, and measures the resulting states. The fidelity between the resulting 
    state and the expected final states is calculated for each case. The function returns a dictionary containing fidelity values 
    for different prepared states, along with a list containing the randomly generated quantum states used in the evaluation.

    The function supports preparing and evaluating the fidelity of the following states:
    - |1>
    - |1err2>
    - |1err1>
    - |1err0>
    - |0>
    - |0err2>
    - |0err1>
    - |0err0>
    - |+>
    - |+err0>
    - |+err1>
    - |+err2>
    - Randomly generated quantum states (labeled as 'rand', 'randerr0', 'randerr1', and 'randerr2')
    
    Additionally, the function prints the mean fidelity and fidelity values for each prepared state.
    """

    reset_circ = QuantumCircuit(3,1)
    reset_circ.barrier()
    reset_circ.reset([1,2])

    circ_copy: QuantumCircuit = circ.copy()
    circ_with_measurement: QuantumCircuit = circ_copy.compose(reset_circ)

    num_qubits = circ.num_qubits

    encoded_state = QuantumCircuit(1,1)

    gates_to_apply = [HGate, RYGate, RXGate]

    random_state = QuantumCircuit(num_qubits)
    random_state_err0 = QuantumCircuit(num_qubits)
    random_state_err1 = QuantumCircuit(num_qubits)
    random_state_err2 = QuantumCircuit(num_qubits)
    random_state_err = [random_state_err0, random_state_err1, random_state_err2]

    for i in range(4):
        gate = random.choice(gates_to_apply)
        if gate in [RYGate, RXGate]:
            rand_angle = random.uniform(-2*np.pi, 2*np.pi,)
            encoded_state.append(gate(rand_angle),[0])
            random_state.append(gate(rand_angle),[0])
            for i in range(3):
                random_state_err[i].append(gate(rand_angle), [0])
        else:
            encoded_state.append(HGate(),[0])
            random_state.append(HGate(), [0])
            for i in range(3):
                random_state_err[i].append(HGate(), [0])
    random_state.cx(0,1)
    random_state.cx(0,2)
    for i in range(3):
        random_state_err[i].cx(0,1)
        random_state_err[i].cx(0,2)
        random_state_err[i].x(i)

    expected_random = Statevector(encoded_state)

    circ_labels = ['1','1err2','1err1','1err0',
                   '0','0err2','0err1','0err0',
                   '+','+err0','+err1','+err2']

    plus_err = [plus_err0, plus_err1, plus_err2]

    init = [ooo, ooz, ozo, zoo, zzz, zzo, zoz, ozz, plus_state]
    init = init + plus_err + [random_state] + random_state_err
    final = [one]*4 + [zero]*4 + [plus]*4 + [expected_random]*4

    fids = []
    for i in range(len(init)):
        aux_circ = init[i].compose(circ_with_measurement)
        aux_circ = aux_circ.decompose()
        state = Statevector(aux_circ)
        state = partial_trace(state, [1,2])
        fid = state_fidelity(state, final[i])
        fids.append(fid)

    fids_dict = {circ_labels[i] : fids[i] for i in range(len(circ_labels))}
    fids_dict['rand'] = fids[len(circ_labels)]
    fids_dict['randerr0'] = fids[len(circ_labels)+1]
    fids_dict['randerr1'] = fids[len(circ_labels)+2]
    fids_dict['randerr2'] = fids[len(circ_labels)+3]

    print(f"The mean fidelity is: {np.mean(fids)}")
    for keys, values in fids_dict.items():
        print(f"The fidelity for state {keys} is: {values}")

    return fids_dict, [random_state] + random_state_err

def evaluate_circuit_tomography(circ: QuantumCircuit, backend: Backend | None = None, re_encoding: bool = False) -> list[float]:
    """
    Evaluate the fidelities of an ansatz with initial states |0>, |1>, |+>, and a random state using quantum state tomography.

    Parameters
    ----------
    circ : QuantumCircuit
        The ansatz to be evaluated.
    backend : Backend, optional
        The backend to execute the circuits on. Defaults to the qasm_simulator backend.
    re_encoding : bool, optional
        Flag indicating whether to include re-encoding circuits. Defaults to False.

    Returns
    -------
    List[float]
        A list of fidelity values corresponding to each evaluated quantum state.

    Description
    -----------
    This function evaluates the fidelity of a quantum circuit by performing quantum state tomography. It constructs
    tomography circuits for the input quantum circuit, executes them on the specified backend, and calculates the fidelity
    between the resulting state and the target state for each tomography circuit.

    If `re_encoding` is set to True, re-encoding is done and 3 qubit state tomography is performed. Otherwise, the
    tomography is done only for qubit q0.

    Returns a list of fidelity values corresponding to each evaluated quantum state.
    """

    def custom_function(x):
        if 0 <= x <= 3:
            return 0
        elif 4 <= x <= 7:
            return 1
        elif 8 <= x <= 11:
            return 2
        elif 12 <= x:
            return 3
        
    circ_labels = ['1','1err2','1err1','1err0',
                   '0','0err2','0err1','0err0',
                   '+','+err0','+err1','+err2',
                   'rand','rand_err0','rand_err1','rand_err2']

    if backend is None:
        backend = Aer.get_backend('qasm_simulator')

    circs_for_tomo, random_state, target_random = build_tomo_circs(circ, re_encoding)
    if re_encoding:
        meas_indices = None
        init_states_no_noise = [zzz, ooo, plus_state, random_state]
    else:
        meas_indices = [0]
        init_states_no_noise = [zero, one, plus, Statevector(target_random)]
        
    fids = []
    for i, circ_tomo in enumerate(circs_for_tomo):
        tomography = StateTomography(
            circuit = circ_tomo,
            backend = backend,
            measurement_indices = meas_indices,
        )
        tomography_experiment = tomography.run(backend=backend)
        resulting_density_matrix = tomography_experiment.analysis_results("state").value
        target_state = Statevector(init_states_no_noise[custom_function(i)])

        fids.append(state_fidelity(resulting_density_matrix, target_state))
        print(f"State fidelity for state {circ_labels[i]} is: {fids[-1]}.")

    print(f"The average fidelity is {np.average(fids)}.")
    return fids

def evaluate_phase_correcting_circuit(
    circ: QuantumCircuit, 
    backend: Backend | None = None,
    do_inverse_operation: bool = False,
    num_rand: int = 3,
    ) -> tuple[dict[str, float], list[QuantumCircuit]]:
    
    """
    Evaluate the fidelity of final state with respect to the expected final state using tomography.

    Parameters
    ----------
    circ : QuantumCircuit
        The QuantumCircuit object to evaluate.
    backend : Backend, optional
        The backend to execute the circuits on. If None, defaults to the QASM simulator.
    do_inverse_operation : bool, optional
        Flag to indicate whether the reverse operation should be performed or whether to use
        quantum state tomography. Defaults to False.
    num_rand : int, optional
        Number of random states to be used. Defaults to 3.

    Returns
    ----------
    dict
        A dictionary containing fidelity values for different prepared states.
    rands
        A list of the circuits representing initial random states.
    """

    if backend is None:
        backend = Aer.get_backend('qasm_simulator')

    circ_copy: QuantumCircuit = circ.copy()

    circ_labels = ['0','1', '+'] 
    for i in range(num_rand):
        circ_labels += ['rand' + str(i+1)]

    rands = []
    rand_rotations = []
    if do_inverse_operation:
        inverse_0 = QuantumCircuit(3,1)
        inverse_0.measure(0, 0)
        inverse_1 = QuantumCircuit(3,1)
        inverse_1.x(0)
        inverse_1.measure(0, 0)
        inverse_plus = QuantumCircuit(3,1)
        inverse_plus.h(0)
        inverse_plus.measure(0, 0)
        inverse_rotations = [inverse_0, inverse_1, inverse_plus]
    final_rands = []
    
    for i in range(num_rand):
        rand_state_circ = QuantumCircuit(1)
        rand_circ = QuantumCircuit(3)            
        
        # Create the random initial states circuits
        rand_rotation_x = random.uniform(-2*np.pi, 2*np.pi)
        rand_circ.rx(rand_rotation_x, 0)
        rand_state_circ.rx(rand_rotation_x, 0)
        rand_rotation_y = random.uniform(-2*np.pi, 2*np.pi)
        rand_circ.ry(rand_rotation_y, 0)
        rand_state_circ.ry(rand_rotation_y, 0)
        
        rands += [rand_circ]
        rand_rotations.append([rand_rotation_x, rand_rotation_y])
        
        # Create inverse rotations
        if do_inverse_operation:
            inverse_rand_circ = QuantumCircuit(3,1)
            inverse_rand_circ.ry(-rand_rotation_y, 0)
            inverse_rand_circ.rx(-rand_rotation_x, 0)
            inverse_rand_circ.measure(0,0)
            
            inverse_rotations += [inverse_rand_circ]
        
        # Store the expected final state
        final_rands += [Statevector(rand_state_circ)]

    final = [zero, one, plus] + final_rands

    init1 = QuantumCircuit(3)
    init1.rx(0,0)
    init1.barrier()
    init2 = QuantumCircuit(3)
    init2.x(0)
    init2.barrier()
    init3 = QuantumCircuit(3)
    init3.h(0)
    init3.barrier()
    init = [init1, init2, init3] + rands

    fids = []
    
    if do_inverse_operation:
        for i in range(len(init)):
            aux_circ = init[i].compose(circ_copy.compose(inverse_rotations[i]))
            job = backend.run(
                [aux_circ],
                shots = 200,
                job_name = f'Circuit_{circ_labels[i]}'
            )
            
            res = job.result()
            counts = res.get_counts()
            count_labels = list(counts.keys())
            count_labels.sort()
            if len(counts) == 2:
                frequencies = [counts[c] / 200 for c in count_labels]
            else:
                # Counts is keyed as '0x0' or '0x1', or as '0' or '1', so the last 
                # letter of the string gives which was the only result one gets.
                bit = count_labels[0][-1]
                freq_to_append = [1, 0] if bit == '0' else [0, 1]
                frequencies = freq_to_append
                
            pseudo_state = np.dot([np.array([1,0]), np.array([0,1])], np.sqrt(frequencies))
            # The target state is always the zero state.
            fid = state_fidelity(pseudo_state, zero)
            fids += [fid]
    else:
        for i in range(len(init)):
            aux_circ = init[i].compose(circ_copy)
            aux_circ = aux_circ.decompose()
            tomography = StateTomography(
                circuit = aux_circ,
                backend = backend,
                measurement_indices = [0],
            )
            
            tomography_experiment = tomography.run(backend = backend)
            resulting_density_matrix = tomography_experiment.analysis_results("state").value
            
            fids.append(state_fidelity(resulting_density_matrix, final[i]))
        
    fids_dict = {circ_labels[i] : fids[i] for i in range(len(circ_labels))}

    print(f"The mean fidelity is: {np.mean(fids)}")

    for keys, values in fids_dict.items():
        print(f"The fidelity for state {keys} is: {values}")

    return fids_dict, rands, rand_rotations
