import numpy as np

class QuantumGates:
    """Class initialising different quantum gates"""
    I_gate = np.array([[1, 0], [0, 1]], dtype=complex)  # Identity
    X_gate = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X gate
    Y_gate = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli Y gate
    Z_gate = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli Z gate
    H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)  # Hadamard gate
    S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)  # S gate (phase gate)
    T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # T gate
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
        self.state = np.zeros(2 ** num_qubits, dtype=complex)  # Initialises register with n qubits (in 0 state)
        self.state[0] = 1.0  # 100% probability that qubit is in zero state (alpha = 1)
    def apply_gate(self, gate, target):
        """Apply a single-qubit gate to the target qubit
        Arguments:
            gate: The quantum gate to apply
            target: Index of the target qubit        """
        quantum_gate = np.array([[1]], dtype=complex)
        for i in range(self.num_qubits):  # Looping through number of qubits in register
            if i == target:
                quantum_gate = QuantumGates.tensor_product(quantum_gate, gate)  # If the qubit is the target qubit, apply the quantum gate

            else:  # Otherwise, if qubit is not the target, apply the identity so it remains unchanged
                quantum_gate = QuantumGates.tensor_product(quantum_gate, QuantumGates.I_gate)
        # Apply gate to state vector
        self.state = quantum_gate @ self.state

    def apply_controlled_NOT_gate(self, control, target):
        """Apply a controlled-NOT gate between control and target qubits
            CNOT flips the target bit ONLY if the control bit is |1>

           The Quantum Gates CNOT function is not used here as that method only works for 2 qubits. This method is for
           any amount of qubits and is executed differently"""

        n = self.num_qubits
        dim = 2 ** n  # Dimension of vector space
        # Create a new state vector consisting of all zeros
        new_state = np.zeros(dim, dtype=complex)
        for i in range(dim):  # Looping through each state in quantum register
            # Labelling basis states with binary integers (e.g: |7> = |111> for 3 qubit register)
            binary = format(i, f'0{n}b')[::-1]  # Reversed for proper qubit ordering (little endian)
            # Check if control qubit is |1>
            if binary[control] == '1':
                # Flip target qubit
                flipped = list(binary)  # Convert binary string to list
                # If target state is 0, flip to 1
                if flipped[target] == '0':
                    flipped[target] = '1'
                else:  # Else, if target state is 1, flip to 0
                    flipped[target] = '0'
                flipped = ''.join(flipped)[::-1]  # Reverse back to standard ordering (and convert list to string)
                j = int(flipped, 2)  # Converting binary string to decimal integer
                new_state[j] += self.state[i]  # Using decimal integer (e.g j=1-7 for 3 qubit register) to update state
            else:
                # Leave state unchanged (if control qubit is |0>
                new_state[i] += self.state[i]
        self.state = new_state

    def apply_controlled_z(self):
        """This function is for when there are more than 2 qubits
        Apply a phase flip to the state where all qubits are 1"""
        n = self.num_qubits
        dim = 2 ** n  # Dimension of vector space
        # Create a copy of new state vector- to avoid editing original state which can lead to carry-over errors
        new_state = np.copy(self.state)
        # If both control and target qubits are |1>, apply phase flip
        new_state[dim - 1] *= -1  # Apply phase flip (to last state in register because this will always correspond to all 1's
        self.state = new_state  # Updating state

    def apply_phase_flip(self, target_state):
        """Apply a phase flip directly to target state"""
        # Convert target state to an index
        target_index = int(target_state, 2)
        # Flip the phase of the state
        self.state[target_index] *= -1

    def measure(self):
        """Measure and collapse the quantum state- return result and probabilities"""
        # Calculate probabilities (aquaring the amplitudes)
        probabilities = np.abs(self.state) ** 2
        # Sum probabilities to 1
        probabilities = probabilities / np.sum(probabilities)
        # Randomly picking one of the states within 2^n based on their probabilities
        outcome = np.random.choice(2 ** self.num_qubits, p=probabilities)
        # Convert to binary representation of the state
        binary_result = format(outcome, f'0{self.num_qubits}b')
        return binary_result, probabilities  # return state (in binary representaiton) and its corresponding probability

class QuantumCircuit:
    """This class build a quantum circuit that can apply gates to a quantum register"""

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.register = QuantumRegister(num_qubits)  # initialising the quantum register

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


class QuantumErrorCorrection:
    """Implementation of 3-qubit bit flip quantum error correction with 5 qubits
    (3 data qubits and 2 auxiliary qubits)"""

    def __init__(self):
        self.qubits = QuantumCircuit(5)  # 3 data qubits + 2 auxiliary qubits

    def print_state_description(self, description):
        """Prints the quantum state with a description"""
        # Get the state vector
        state = self.qubits.register.state
        # Find non-zero amplitudes
        non_zero_indices = [i for i in range(len(state)) if abs(state[i]) > 1e-10]
        # Create a string representation of the state using binary representation
        state_str = " + ".join(f"|{format(i, f'0{self.qubits.num_qubits}b')}⟩" for i in non_zero_indices)
        # Print the description and state
        print(f"{description}: {state_str}")

    def print_final_state(self, description):
        """Prints the quantum state with a description, showing only the first digit of binary representation"""
        # Get the state vector
        state = self.qubits.register.state
        # Find non-zero amplitudes
        non_zero_indices = [i for i in range(len(state)) if abs(state[i]) > 1e-10]
        # Create a string representation of the state with only the first digit and real part of amplitude
        state_str = " + ".join(
            f"{np.round(np.real(state[i]), 3)}|{format(i, f'0{self.qubits.num_qubits}b')[0]}⟩" for i in
            non_zero_indices)
        # Print the description and state
        print(f"{description}: |ψ⟩ = {state_str}")

    def encode(self, alpha, beta):
        """Encode a single qubit state |ψ⟩ = α|0⟩ + β|1⟩ into a 3-qubit error correction code

        We use the following encoding:
        |0⟩ → |000⟩
        |1⟩ → |111⟩
        So ( α|0⟩ + β|1⟩ )|0000⟩ → ( α|000⟩ + β|111⟩ )|00⟩ """

        # Create initial state |ψ⟩ = α|0⟩ + β|1⟩
        self.qubits.register.state[0] = alpha  # |00000⟩
        self.qubits.register.state[16] = beta  # |10000⟩
        # Encode state into 3-qubit code
        # (CNOT gates use little endian ordering, whereas this QEC section uses big endian for intuitive displaying of results)
        # This spreads the state across 3 qubits, creating the encoded state
        self.qubits.cnot(4, 3)  # CNOT gate with qubit 0 as control and qubit 1 as target, gives |00000⟩ + |11000⟩
        self.qubits.cnot(4, 2)  # CNOT gate with qubit 0 as control and qubit 2 as target, gives |00000⟩ + |11100⟩
        self.print_state_description("State after encoding")

    def introduce_error(self, qubit_index):
        """Introduce a bit flip error on the specified qubit.
        This simulates noise that can occur during transmission of qubits."""
        # Apply X gate (bit flip) to the specified qubit
        self.qubits.x(qubit_index)
        self.print_state_description(f"State after introducing error on qubit {qubit_index}")

    def syndrome_measurement(self):
        """Perform syndrome measurement to detect bit flip errors
        This uses the two auxiliary qubits (3 and 4) to extract information about
        errors in the data qubits without measuring them directly.

        The syndrome values correspond to:
         00: No error
         11: Error on qubit 0
         10: Error on qubit 1
         01: Error on qubit 2 """

        # Apply CNOTs to extract error information into the auxiliary qubits (qubits 3 and 4)
        # Measure parity between data qubits 0 and 1 into auxilliary qubit 3
        self.qubits.cnot(4, 1).cnot(3,1)  # qubit 1 detecting error in qubits 3 and 4 in little endian notation for CNOT gate
        # Measure parity between data qubits 0 and 2 into auxilliary qubit 4
        self.qubits.cnot(4, 0).cnot(2, 0)  # qubit 0 detecting error in qubits 2 and 4 in little endian notation for CNOT gate
        self.print_state_description("State after syndrome measurement")
        result = self.qubits.measure()[0]
        syndrome = result[3:5]  # syndrome is the last two bits (qubits 3 and 4)
        print(f"Syndrome: {syndrome}")
        return syndrome

    def correct_error(self, syndrome):
        """ Apply error correction using X-gate based on the syndrome measurement"""
        if syndrome == "00":
            print("No error detected, no correction needed")
        if syndrome == "01":
            self.qubits.x(2)  # Apply X gate to qubit 2 to correct the bit flip
        elif syndrome == "10":
            self.qubits.x(1)  # Apply X gate to qubit 1 to correct the bit flip
        elif syndrome == "11":
            self.qubits.x(0)  # Apply X gate to qubit 0 to correct the bit flip
        self.print_state_description("State after error correction")

    def decode(self):
        """Decode the 3-qubit error correction code back to a single logical qubit """
        # Apply CNOTs to decode the state (reverse of encoding)
        self.qubits.cnot(2, 4).cnot(3, 4)  # Decode the state back to qubit 0 (qubit 4 in little endian)
        self.print_state_description("State after decoding")
        self.print_final_state("Final state")

# Initialize with a superposition state
alpha, beta = 1 / np.sqrt(2), 1 / np.sqrt(2)  # Equal superposition |0⟩ + |1⟩ (normalized)
print(f"Original state (including amplitudes): |ψ⟩ = {np.round(alpha, 3)}|0⟩ + {np.round(beta, 3)}|1⟩")
# Run quantum error correction
qec = QuantumErrorCorrection()
qec.encode(alpha, beta)  # Encode the qubit into a 3-qubit code
# Introduce bit-flip error on first qubit
q = int(input("Enter the qubit to introduce an error on (0-2) : "))
qec.introduce_error(q)  # Introduce error on the specified qubit
# Perform syndrome measurement and correction
syndrome = qec.syndrome_measurement()
qec.correct_error(syndrome)
# Decode and get final state
result = qec.decode()