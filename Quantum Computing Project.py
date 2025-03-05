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
# defining basic qubits
q0 = np.array([1, 0])  # |0>
q1 = np.array([0, 1])  # |1> 
q_plus = (1 / np.sqrt(2)) * np.array([1, 1])   # |+>
q_minus = (1 / np.sqrt(2)) * np.array([1, -1]) # |->

#defining qubit function for any alpha, beta 
def qubit(alpha, beta):
    norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    return np.array([alpha/norm, beta/norm])

# Hadamard gate
H_matrix = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
print(H_matrix)

# tensor product function
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
                    # Calculate the position in the result matrix directly
                    row_idx = i * rows_B + k
                    col_idx = j * cols_B + l
                    # Set the value at that position
                    result[row_idx, col_idx] = A[i, j] * B[k, l]
    
    return result

# Example usage
q_reg = tensor_product(q0, q0) # quntum register: 00
result = tensor_product(H_matrix, q_reg) 
print(f'Tensor product: {result}')

print(f'Tensor product: { tensor_product(H_matrix, result)}') 

print(f'Tensor product hadamard: { tensor_product(H_matrix, q0)}') 
#---------------------------------------
#code to be checked/worked through below 

# Create an n-qubit equal superposition state
def equal_superposition(n):
    state = np.array([1])
    for _ in range(n):
        state = tensor_product(state.reshape(-1, 1), q_plus.reshape(-1, 1)).flatten()
    return state

# Oracle function (flips the phase of the marked state)
def oracle(n, target_index):
    N = 2**n
    oracle_matrix = np.identity(N)
    oracle_matrix[target_index, target_index] = -1  # Phase flip
    return oracle_matrix

# Diffusion operator (inversion about the mean)
def diffusion_operator(n):
    N = 2**n
    H_n = H_matrix
    for _ in range(n-1):
        H_n = tensor_product(H_n, H_matrix)
    
    mean_matrix = (2/N) * np.ones((N, N))
    return H_n @ (2 * np.outer(np.ones(N), np.ones(N)) / N - np.identity(N)) @ H_n

# Grover's algorithm implementation
def grover(n, target_index, iterations):
    N = 2**n
    state = equal_superposition(n)  # Step 1: Initialize superposition
    
    O = oracle(n, target_index)
    D = diffusion_operator(n)
    
    for _ in range(iterations):
        state = O @ state  # Apply Oracle
        state = D @ state  # Apply Diffusion
    
    return state

# Parameters
n = 10 # Number of qubits (searching a space of size 2^3 = 8)
target_index = 4  # Marked state
iterations = int(np.pi/4 * np.sqrt(float(2**n))) # Optimal number of iterations

# Run Grover's algorithm
final_state = grover(n, target_index, iterations)

# Print final state probabilities
probabilities = np.abs(final_state)**2
print("Final state probabilities:", probabilities)

# Measurement (finding the most probable state)
found_index = np.argmax(probabilities)
print("Measured state (most probable):", found_index)








