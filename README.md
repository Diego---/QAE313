# Quantum Autoencoder

A Python package for training the 313 Quantum Autoencoders for autonomous correction of 1 qubit flips.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

### Introduction
Quantum Autoencoders are an innovative approach to compress quantum data, allowing for more efficient storage and processing. This package provides an easy-to-use interface for training and evaluating the 313 Quantum Autoencoders, compatible with qiskit. It's inspired by the work in [this article](https://quantum-journal.org/papers/q-2023-03-09-942/).

### Installation
To install the package, contact me at dolveram@uni-mainz.de.

### Usage
This package can be used to build the circuits used for training, as well as the cost function, termination checkers, call-back for the optimizer, etc.

### Examples
```python
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector

from qae.optimization.cost_functions import create_stochastic_cost_func
from qae.evaluation.ansatz_evaluation import evaluate_circuit
from qae.plotting.plotting import plot_cost_evolution, plot_parameter_evolution
from qae.optimization.my_spsa import SPSA, powerseries
from qae.utils.termination_and_callback import TerminationChecker, create_live_plot_callback

loc_sim = Aer.get_backend('qasm_simulator')
th = ParameterVector('t', 22)
ansatz = QuantumCircuit(3)

# CNOT_01
ansatz.ry(0.5*pi, 1) # ansatz.ry(th[0], 1)
ansatz.rx(pi, 1) # ansatz.rx(th[1], 1)
ansatz.cz(0,1)
ansatz.ry(0.5*pi, 1) # ansatz.ry(th[2], 1)
ansatz.rx(pi, 1) # ansatz.rx(th[3], 1) ...

mini_batch=1
num_shots=200
cost_func = create_stochastic_cost_func(loc_sim, ansatz, num_shots, mini_batch=mini_batch)

# Learning rate sequence (a is the initial scaling factor, alpha controls decay rate)
a = 0.6
alpha = 0.402
A = 0
learning_rate_gen = powerseries(a, alpha, offset=A)

# Perturbation sequence (c is the initial scaling factor, gamma controls decay rate)
c = 0.1
gamma = 0.101
perturbation_gen = powerseries(c, gamma)

# Optimizer settings
iterations = 300
random_initi_val = [1.]*8

# We keep track of counts(of function evaluations), values and parameters through the optimization.
counts = []
values = []
params = []
stepsize = []
# Create the directory if it doesn't exist
directory = "./int_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# File path to store intermediate data
json_file_path = './int_data/experiment.json'
store_data = True

# Set up the callback function
store_intermediate_result_no_est3 = create_live_plot_callback(
    counts,
    values,
    params,
    stepsize,
    json_file_path,
    store_data=store_data
)

# Set up the termination checker
target_value = 1
tolerance = 0.01
stagnation_tol = 0.0001
number_past_iterations = 40
term_check = TerminationChecker(
    target_value,
    tolerance,
    stagnation_tol,
    number_past_iterations
)

# Set up the optimizer
spsa = SPSA(
    maxiter = iterations, 
    callback=store_intermediate_result_no_est3,
    termination_checker=term_check,
    perturbation = perturbation, 
    learning_rate = learning_rate,
    resamplings=2,
    )

result = spsa.minimize(
    fun=cost, 
    x0=random_start_point, 
    fun_next=cost_simulator
    )

average_fidelity = result.fun
spsa_params = [[] for i in range(len(params[0]))]
for i, params_at_i in enumerate(params):
    for j in range(len(params_at_i)):
        spsa_params[j].append(params_at_i[j])

fidelity_values = [-values[i] for i in range(len(values))] 
ind_max = np.argmax(fidelity_values)
print('Max av. fidelity achieved:', fidelity_values[ind_max], '|', max(fidelity_values))
best_params = params[ind_max]
best_params_map = {sym_param : best_params[i] for i,sym_param in enumerate(used_ansatz.parameters)}
best_circ = used_ansatz.assign_parameters(best_params_map)

eval_dict, random_states_evaluated = evaluate_circuit(best_circ)
fig, ax = plot_cost_evolution(fidelity_values, fidelity_values[ind_max], 1)
fig, ax = plot_parameter_evolution(
    spsa_params,
    index_of_params=None, 
    answer_values=list(answer_values)
    )

```


