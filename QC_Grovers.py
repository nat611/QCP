#imports and libraries
import numpy as np
import math
import time
import matplotlib.pyplot as plt

class QuantumGates:
    """Class initialising all different quantum gates"""
    I_gate = np.array([[1, 0], [0, 1]], dtype=complex) #Identity
    X_gate = np.array([[0, 1], [1, 0]], dtype=complex) #Pauli X gate
    Y_gate = np.array([[0, -1j], [1j, 0]], dtype=complex) #Pauli Y gate
    Z_gate = np.array([[1, 0], [0, -1]], dtype=complex) #Pauli Z gate
    H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex) #Hadamard
    S_gate = np.array([[1, 0], [0, 1j]], dtype=complex) #S gate (phase gate)
    T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex) #T gate

    """Controlled NOT gate"""
    def CNOT(self):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)

    """Controlled Z gate"""
    def CZ(self):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, -1]], dtype=complex)


    @staticmethod
    def tensor_product(A, B):
        """Tensor product function: Begins by creating empty matrix of correct dimensions,
            then loops through rows and columns of matrices A and B, multiplying their corresponding
            values and appending them to the empty matrix at the correct index"""

        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape

        result = np.zeros((rows_A * rows_B, cols_A * cols_B), dtype=complex)

        for i in range(rows_A):
            for j in range(cols_A):
                for k in range(rows_B):
                    for l in range(cols_B):
                        result[i * rows_B + k, j * cols_B + l] = A[i, j] * B[k, l]
        return result

class QuantumRegister:
    """Quantum register class: initialises an n qubit quantum register in state |000...>
       and defines functions for applying different gates, phase flips, and measurments"""
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2 ** num_qubits, dtype=complex) #Initialises register with n qubits (in 0 state)
        self.state[0] = 1.0 #100% probability that qubit is in zero state (alpha = 1)

    def apply_gate(self, gate, target):
        """Apply a single-qubit gate to the target qubit
        Arguments:
            gate: The quantum gate to apply
            target: Index of the target qubit        """

        quantum_gate = np.array([[1]], dtype=complex)
        for i in range(self.num_qubits): #Looping through number of qubits in register
            if i == target:
                quantum_gate = QuantumGates.tensor_product(quantum_gate, gate) #If the qubit is the target qubit, apply the quantum gate

            else: #Otherwise, if qubit is not the target, apply the identity so it remains unchanged
                quantum_gate = QuantumGates.tensor_product(quantum_gate, QuantumGates.I_gate)
        #Apply gate to state vector
        self.state = quantum_gate @ self.state

    def apply_controlled_NOT_gate(self, control, target):
        """Apply a controlled-NOT gate between control and target qubits

            CNOT flips the target bit ONLY if the control bit is |1>

           The Quantum Gates CNOT function is not used here as that method only works for 2 qubits. This method is for
           any amount of qubits and is executed differently"""

        n = self.num_qubits
        dim = 2 ** n #Dimension of vector space

        # Create a new state vector consisting of all zeros
        new_state = np.zeros(dim, dtype=complex)

        for i in range(dim): #Looping through each state in quantum register
            # Labelling basis states with binary integers (e.g: |7> = |111> for 3 qubit register)
            binary = format(i, f'0{n}b')[::-1] #Reversed for proper qubit ordering (little endian)
            # Check if control qubit is |1>
            if binary[control] == '1':
                # Flip target qubit
                flipped = list(binary) #Convert binary string to list

                # If target state is 0, flip to 1
                if flipped[target] == '0':
                    flipped[target] = '1'
                else: # Else, if target state is 1, flip to 0
                    flipped[target] = '0'

                flipped = ''.join(flipped)[::-1]  # Reverse back to standard ordering (and convert list to string)
                j = int(flipped, 2) #Converting binary string to decimal integer
                new_state[j] += self.state[i] #Using decimal integer (e.g j=1-7 for 3 qubit register) to update state

            else:
                # Leave state unchanged (if control qubit is |0>
                new_state[i] += self.state[i]

        self.state = new_state

    def apply_controlled_z(self):

        """This function is for when there are more than 2 qubits
        Apply a phase flip to the state where all qubits are 1"""

        n = self.num_qubits
        dim = 2 ** n #Dimension of vector space
        #Create a copy of new state vector- to avoid editing original state which can lead to carry-over errors
        new_state = np.copy(self.state)

        # If both control and target qubits are |1>, apply phase flip
        new_state[dim - 1] *= -1 #Apply phase flip (to last state in register because this will always correspond to all 1's
        self.state = new_state #Updating state

    def apply_phase_flip(self, target_state):
        """Apply a phase flip directly to target state"""
        #Convert target state to an index
        target_index = int(target_state, 2)
        #Flip the phase of the state
        self.state[target_index] *= -1

    def measure(self):
        """Measure and collapse the quantum state- return result and probabilities"""
        #Calculate probabilities (aquaring the amplitudes)
        probabilities = np.abs(self.state) ** 2
        #Sum probabilities to 1
        probabilities = probabilities / np.sum(probabilities)
        #Randomly picking one of the states within 2^n based on their probabilities
        outcome = np.random.choice(2 ** self.num_qubits, p=probabilities)
        #Convert to binary representation of the state
        binary_result = format(outcome, f'0{self.num_qubits}b')
        return binary_result, probabilities #return state (in binary representaiton) and its corresponding probability

class QuantumCircuit:
    """This class build a quantum circuit that can apply gates to a quantum register"""
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.register = QuantumRegister(num_qubits) #initialising the quantum register

    def h(self, qubit):
        """Apply Hadamard gate to qubit"""
        self.register.apply_gate(QuantumGates.H_gate, qubit)
        return self

    def x(self, qubit):
        """Apply X gate to qubit"""
        self.register.apply_gate(QuantumGates.X_gate, qubit)
        return self

    def z(self, qubit):
        """Apply Z gate to qubit"""
        self.register.apply_gate(QuantumGates.Z_gate, qubit)
        return self

    def cnot(self, control, target):
        """Apply CNOT gate with control and target qubits"""
        self.register.apply_controlled_NOT_gate(control, target)
        return self

    def cz(self):
        """Apply CZ gate with control and target qubits"""
        self.register.apply_controlled_z()
        return self

    def phase_flip(self, target_state):
        """Apply phase flip to target state"""
        self.register.apply_phase_flip(target_state)
        return self

    def measure(self):
        """Collapse all qubits in the circuit"""
        return self.register.measure()

def grovers_algorithm(num_qubits=2, target_state='01'):
    print(f"\nRunning Grover's Algorithm on {num_qubits} qubits, searching for |{target_state}⟩")
    circuit = QuantumCircuit(num_qubits) #initalising instance of QuantumCircuit class

    start_time = time.time()  # Start overall timer

    #Calculate optimal number of iterations
    iterations = math.floor((math.pi / 4) * math.sqrt(2 ** num_qubits))
    print(f"Optimal number of iterations: {iterations}")

    #Putting the quantum register in an equal superposition
    # This is done by applying hadamard gate to all qubits in the register
    print("Applying Hadamard gates to all qubits.")
    for i in range(num_qubits):
        circuit.h(i) #applying hadamard

    #Calculating and printing the probabilities after the initialisation step
    _, initial_probs = circuit.register.measure()
    print("\nState after initialisation:")
    for i, prob in enumerate(initial_probs):
        if prob > 0.001:  #Not showing probabilities once they become too small
            print(f"|{format(i, f'0{num_qubits}b')}⟩: {prob:.4f}") #Printing all probabilities

    total_iteration_time = 0 #Initialising iteration time
    #Alternating between oracle and diffusion steps for optimal number of iterations
    for iteration in range(iterations):
        iteration_start_time = time.time()  # Start iteration timer
        print(f"\nIteration {iteration + 1}:")

        #Applying oracle to mark the target state
        print("Applying Oracle")
        #First need to apply x gate to target: if the state is 0, it is flipped to 1
        for i, bit in enumerate(target_state):
            if bit == '0':
                circuit.x(i)

        #Apply phase flip to state where all qubits are 1 (which will be true because of x gate)
        print("Applying controlled-Z to mark the target state")
        circuit.register.apply_controlled_z()

        #Now we need to restore the target back to its original state, so we apply an x gate again to reverse the operation
        for i, bit in enumerate(target_state):
            if bit == '0':
                circuit.x(i)

        #Calculating probabilites after oracle has been applied and printing them
        _, oracle_probs = circuit.register.measure()
        print("State after oracle:")
        for i, prob in enumerate(oracle_probs):
            if prob > 0.001: #Not showing probabilities once they become too small
                print(f"|{format(i, f'0{num_qubits}b')}⟩: {prob:.4f}") #printing probabilities

        #Applying the diffusion operator
        print("Applying Diffusion Operator:")
        #Apply hadamard to all qubits in register
        for i in range(num_qubits):
            circuit.h(i)

        #Apply X gates to all qubits
        for i in range(num_qubits):
            circuit.x(i)

        #Apply controlled-Z for diffusion
        print("Applying controlled-Z for diffusion operator:")
        circuit.register.apply_controlled_z()

        #Apply x gates and hadamard gates to all qubits again
        for i in range(num_qubits):
            circuit.x(i)
            circuit.h(i)

        #Calculating probabilites after diffusion
        _, diffusion_probs = circuit.register.measure()
        print("State after diffusion:")
        for i, prob in enumerate(diffusion_probs):
            if prob > 0.001:  #Not showing probabilities once they become too small
                print(f"|{format(i, f'0{num_qubits}b')}⟩: {prob:.4f}") #Printing probabilities

        iteration_time = time.time() - iteration_start_time  # End iteration timer
        total_iteration_time += iteration_time

        avg_iteration_time = total_iteration_time / iterations if iterations > 0 else 0

    #Measure the final result
    result, probabilities = circuit.measure()
    print("State Probabilities:")
    for i, prob in enumerate(probabilities):
        if prob > 0.001:  #Not showing probabilities once they become too small
            print(f"|{format(i, f'0{num_qubits}b')}⟩: {prob:.4f}") #Printing final probability

    total_time = time.time() - start_time  # End overall timer
    print(f"Total execution time: {total_time:.6f} seconds")
    print(f"Average iteration execution time: {avg_iteration_time:.6f} seconds")
    print(f"\nFinal Measurement Result: |{result}⟩") #Printing final result

    # Plotting bar chart of probabilites
    states = []  # Initialize list to store states

    # Loop through all the possible values
    for i in range(2 ** num_qubits):
        # Converting to binary string
        binary_string = format(i, f'0{num_qubits}b')

        # Creating state labels
        state = f"|{binary_string}⟩"
        states.append(state)

    plt.bar(states, probabilities, color='pink', alpha=0.7)  # Plotting bar chart
    plt.xlabel('States')
    plt.ylabel('Probability')
    plt.ylim(0,1)
    plt.title(f'Top States and Their Probabilities for {num_qubits}-Qubit Grover Search')
    plt.show()

    return result, probabilities

#Adding input prompts for desired number of qubits and target state
qubits = int(input('Enter number of qubits: '))
target = str(input('Enter target state: '))

#Running grovers algorithm
grovers_algorithm(num_qubits=qubits, target_state=target)




