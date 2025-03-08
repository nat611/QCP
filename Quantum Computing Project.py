#Quantum Computing Project 
import numpy as np
import random

class Qubit:
    def __init__(self, alpha=1, beta=0):  # default setting |0>
        # Convert to complex numbers
        alpha = complex(alpha)
        beta = complex(beta)
        
        # Ensure normalization
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
            
        self.state = np.array([alpha, beta], dtype=complex) / norm
        
    def __repr__(self):
        return f"Qubit state: {self.state[0]:.4f}|0⟩ + {self.state[1]:.4f}|1⟩"
        
    def measure(self):
        """Measure the qubit, collapsing it to |0⟩ or |1⟩"""
        prob_0 = abs(self.state[0])**2
        result = 0 if random.random() < prob_0 else 1
        
        # Collapse the state
        if result == 0: 
            self.state = np.array([1+0j, 0+0j]) 
        else: 
            self.state = np.array([0+0j, 1+0j])
        
        return result

#---------------------------------------
#defining qubit function for any alpha, beta 
def qubit(alpha, beta):
    norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    return np.array([alpha/norm, beta/norm])

# defining gates
H_gate = (1/np.sqrt(2))*np.array([[1,1],[1,-1]]) # Hadamard gate
X_gate = np.array([[0, 1], [1, 0]])              # Pauli-X (NOT gate)
Y_gate = np.array([[0, -1j], [1j, 0]])           # Pauli-Y
Z_gate = np.array([[1, 0], [0, -1]])             # Pauli-Z
S_gate = np.array([[1, 0], [0, 1j]])             # Phase gate
T_gate = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])  # T gate
C_not = np.array([[1, 0, 0, 0], 
                 [0, 1, 0, 0], 
                 [0, 0, 0, 1], 
                 [0, 0, 1, 0]])                   # Controlled-NOT

# defining basic qubits
q0 = np.array([1, 0])  # |0>
q1 = np.array([0, 1])  # |1> 
q_plus = np.matmul(H_gate, q0) # |+>
q_minus = np.matmul(H_gate, q1) # |->


# Tensor product function
def tensor_product(A, B):
    # Ensure A and B are reshaped to 2D arrays
    A = A.reshape(-1, 1) if A.ndim == 1 else A
    B = B.reshape(-1, 1) if B.ndim == 1 else B
    
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    result = np.zeros((rows_A * rows_B, cols_A * cols_B))
    
    for i in range(rows_A):
        for j in range(cols_A):
            for k in range(rows_B):
                for l in range(cols_B):
                    # Calculate the position in the result matrix
                    row_idx = i * rows_B + k
                    col_idx = j * cols_B + l
                    # Set the value at that position
                    result[row_idx, col_idx] = A[i, j] * B[k, l]
    return result

# quantum register for variable number of qubits
def quantum_register(*qubits):
    n = len(qubits)
    result = qubits[0]
    for i in range(1, n):
        result = tensor_product(result, qubits[i])
    return result

#tensor product for gates as well 
def tensor_gates(*gates):
    result = gates[0]
    for i in range(1, len(gates)):
        result = tensor_product(result, gates[i])
    return result

def oracle(n_qubits, target_state):
    N = 2**n_qubits
    oracle_matrix = np.eye(N)
    oracle_matrix[target_state, target_state] = -1
    return oracle_matrix

def diffusion_operator(n_qubits):
    #dimension of state space
    N = 2**n_qubits
    
    #hadamard on each qubit
    h_all = H_gate
    for i in range(1, n_qubits):
        h_all = tensor_product(h_all, H_gate)
    
    # creating matrix of - identity 
    phase_operator = -np.eye(N)  #flip signs of all elements
    phase_operator[0, 0] = 1     # correct 0 element so it doesnt flip 
    
    #apply hadamard on each qubit again 
    result = h_all
    result = np.matmul(phase_operator, result)
    result = np.matmul(h_all, result)
    
    return result

def grovers_algorithm(n_qubits, target_state, num_iterations = None):

    N = 2**n_qubits
    
    # calc optimal number of iterations if not specified
    if num_iterations is None:
        num_iterations = int(np.floor(np.pi/4 * np.sqrt(N)))

    # Initialize all qubits to 0
    state = np.zeros(N)
    state[0] = 1 #putting alpha to 1; 100% chance of register being in 0 state: |000> (ex)
    
    # apply Hadamard to all qubits 
    H_all = tensor_gates(*([H_gate] * n_qubits))
    state = np.matmul(H_all, state)

    # create oracle and diffusion operators
    oracle_op = oracle(n_qubits, target_state)
    diffusion_op = diffusion_operator(n_qubits)
    
    # apply Grover iteration
    for i in range(num_iterations):
        state = np.matmul(oracle_op, state)  # apply oracle
        state = np.matmul(diffusion_op, state)  # apply diffusion operator
    
    return state

# function to measure the final state
def measure_state(state):
    probabilities = np.abs(state)**2
    probabilities = probabilities / np.sum(probabilities) # normalize probabilities to ensure they sum to 1
    cumulative_probs = np.cumsum(probabilities) # calc cumulative probabilities
    random_val = np.random.random() #generate a random number for measurement
    
    # finding measured state 
    for i, cum_prob in enumerate(cumulative_probs):
        if random_val < cum_prob: #comapring probability to random number for measurement 
            return i

    return len(state) - 1 # return last element if no state is found

def run_grovers_search(n_qubits, target):

    # calculate optimal number of iterations
    N = 2**n_qubits
    optimal_iterations = int(np.floor(np.pi/4 * np.sqrt(N)))
    
    print(f"Searching for state |{target}⟩ in a space of {N} elements")
    print(f"Using {optimal_iterations} Grover iterations")
    
    # running grovers algorithm
    final_state = grovers_algorithm(n_qubits, target)
    
    # calculate probabilities
    probabilities = np.abs(final_state)**2
    probabilities = probabilities / np.sum(probabilities)  # normalizing
    
    # print top probabilities
    top_states = np.argsort(-probabilities)[:5]
    print("\nTop measured states and their probabilities:")
    for state in top_states:
        print(f"|{state}⟩: {probabilities[state]:.4f}")
    
    # measurement
    measured_state = measure_state(final_state)
    print(f"\nMeasured state: |{measured_state}⟩")
    
    return measured_state, probabilities[measured_state]

# testing algorithm 
# for 3 qubits and searching for state 6:
result, probability = run_grovers_search(3, 6)
print(f"Success rate: {probability:.2%}")

# example 2 qubits 
result, probability = run_grovers_search(2, 3)
print(f"Success rate: {probability:.2%}")