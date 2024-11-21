# -*- coding: utf-8 -*-
"""
Created on Thursday Apr 15 2024

@author: Diego Alberto Olvera Millán
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

############################################################################
# The circuit that corrects single qubit flips
# With 3 qubits
small_circuit = QuantumCircuit(3)
small_circuit.cx(0,1)
small_circuit.cx(0,2)
small_circuit.ccx(2,1,0)

# Parameters that were trained before.
trained_params1 = [-1.56718515,  1.62250712,  3.17719811, -3.11345426,  1.53389556,
       -3.11400701, -1.59009191,  2.35221162,  2.32323258,  0.00911531,
       -1.52724965, -1.55241754, -1.55595735,  1.52423975, -0.78484903,
       -2.36992862]

trained_params2 = [ 1.57377556,  1.54695642,  3.10896604,  3.07596975, -4.73872175,
        -3.08489143,  1.55138315,  3.9387495 , -2.3432902 , -0.00735987,
         4.62438917,  1.58893027,  1.55435847,  4.65317625,  0.77735699,
         3.96016981]

trained_params3 = [ 1.61849293,  1.58540679,  0.00485851, -3.15897692, -1.54575676,
        0.057356  ,  1.56740799,  0.7841273 ,  0.79900657,  0.05556881,
        1.6316916 ,  1.61280027,  1.56284039,  1.55141433,  0.83549275,
        0.79710971]

trained_params4 = [-4.773637454760679,
  4.698725648281665,
  -6.283185307179586,
  -3.131486037525913,
  4.6929643197019795,
  -3.1616798778786075,
  -1.5303783503020696,
  0.7280117469163131,
  -5.474620023383424,
  -3.1272393148041115,
  -1.5177859590821916,
  -1.5646316636259159,
  1.5686899953685687,
  -4.6735815417604,
  -2.383563759962414,
  0.7408278812964363]

trained_params5 = [ 1.54113995e+00,  1.54640150e+00, -1.09415928e-03, -5.24394001e-02,
        1.60182960e+00,  7.45636283e-02,  1.53850418e+00,  2.34266313e+00,
       -7.38606368e-01, -1.57482906e-02,  1.50053373e+00,  1.53606492e+00,
        1.55443170e+00,  1.55220321e+00, -7.88757496e-01,  2.32632860e+00]

trained_params_minibatch_1pergroup = [ 7.87531141, -1.58511866,  6.33051754,  0.08546485, -7.8928737 ,
       -3.46749536,  4.84563512,  5.54580016,  8.69209397,  0.02333205,
        4.92512543,  1.50720307,  4.7672333 , -4.72821972, -5.50632928,
        3.88743008]

# We define the ansatz.
ansatz0 = TwoLocal(num_qubits=3, rotation_blocks=['ry','rx'], entanglement_blocks='cx',
                  entanglement = 'circular', reps=3)
ansatz0 = ansatz0.decompose()

ansatz_start = QuantumCircuit(3)
for i,instruction in enumerate(ansatz0.data):
    operation = instruction.operation
    if not operation.is_parameterized():
        if i < 25:
            ansatz_start.append(instruction)
    else:
        if operation.params[0].index < 18 and operation.params[0].index not in [13,16]:
            ansatz_start.append(instruction)

ansatz3 = ansatz_start

# Initial states
num = 3

ooo = QuantumCircuit(num)
ooo.x(0)
ooo.x(1)
ooo.x(2)
ooo.barrier()

ooz = QuantumCircuit(num)
ooz.x(0)
ooz.x(1)
ooz.barrier()

ozo = QuantumCircuit(num)
ozo.x(0)
ozo.x(2)
ozo.barrier()

zoo = QuantumCircuit(num)
zoo.x(1)
zoo.x(2)
zoo.barrier()

zzz = QuantumCircuit(num)
zzz.barrier()

zzo = QuantumCircuit(num)
zzo.x(2)
zzo.barrier()

zoz = QuantumCircuit(num)
zoz.x(1)
zoz.barrier()

ozz = QuantumCircuit(num)
ozz.x(0)
ozz.barrier()

plus_state = QuantumCircuit(num)
plus_state.h(0)
plus_state.cx(0,1)
plus_state.cx(0,2)
plus_state.barrier()

plus_err0 = QuantumCircuit(num)
plus_err0.h(0)
plus_err0.cx(0,1)
plus_err0.cx(0,2)
plus_err0.x(0)
plus_err0.barrier()

plus_err1 = QuantumCircuit(num)
plus_err1.h(0)
plus_err1.cx(0,1)
plus_err1.cx(0,2)
plus_err1.x(1)
plus_err1.barrier()

plus_err2 = QuantumCircuit(num)
plus_err2.h(0)
plus_err2.cx(0,1)
plus_err2.cx(0,2)
plus_err2.x(2)
plus_err2.barrier()
############################################################################


############################################################################
# Target states
zero = np.array([1,0])
one = np.array([0,1])
plus = 1/np.sqrt(2) * (one + zero)
############################################################################


############################################################################
# List of initial states without errors.
init_states_no_noise = [zzz, ooo, plus_state]

# Complete list of initial states (along with errors) and corresponding target state.
# The initial states are |000>, |111>, and |+_L> = |000> + |111> plus all 
# possible one qubit flips for those states.
init_states_complete = [zzz, ozz, zoz, zzo, ooo, zoo, ozo, ooz, 
                    plus_state, plus_err0, plus_err1, plus_err2]
# plus_state, plus_err2, plus_err1, plus_err0]

circ_labels = ['0', '0err0', '0err1', '0err2', '1', '1err0', '1err1', '1err2',
               '+', '+err0', '+err1', '+err2']
