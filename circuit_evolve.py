import random
import itertools
from copy import deepcopy

# Gate types with their cost/quality values
GATE_TYPES = {
    'and': 1,
    'or': 1,
    'xor': 2,
    'xnor': 2,
    'nand': 1,
    'nor': 1
}

# Mapping for complement mutations
COMPLEMENT_MAP = {
    'and': 'nand',
    'or': 'nor',
    'xor': 'xnor',
    'xnor': 'xor',
    'nand': 'and',
    'nor': 'or'
}

class LogicCircuit:
    def __init__(self, num_inputs, num_layers, gates_per_layer):
        """
        Initialize a random logic circuit
        Args:
            num_inputs: Number of input signals
            num_layers: Number of layers in the circuit
            gates_per_layer: Number of gates in each layer (except last)
        """
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.gates_per_layer = gates_per_layer
        self.layers = []
        
        # Build each layer of the circuit
        for layer_idx in range(num_layers):
            # Last layer has exactly one gate (output)
            num_gates = 1 if layer_idx == num_layers-1 else gates_per_layer
            layer = []
            
            for _ in range(num_gates):
                # Create gate with random type and valid inputs
                gate = {
                    'type': random.choice(list(GATE_TYPES.keys())),
                    'input1': self._generate_input(layer_idx),
                    'input2': self._generate_input(layer_idx),
                    'cost': 0  # Will be set based on gate type
                }
                gate['cost'] = GATE_TYPES[gate['type']]
                layer.append(gate)
            
            self.layers.append(layer)
    
    def _generate_input(self, current_layer):
        """
        Generate a valid input connection for a gate
        Returns tuple: ('input', index) or ('layer', layer_idx, gate_idx)
        """
        possible_sources = []
        
        # Add primary inputs
        possible_sources.extend([('input', i) for i in range(self.num_inputs)])
        
        # Add outputs from previous layers
        for prev_layer in range(current_layer):
            for gate_idx in range(len(self.layers[prev_layer])):
                possible_sources.append(('layer', prev_layer, gate_idx))
        
        # Ensure inputs are not identical (will be checked when connecting)
        return random.choice(possible_sources)
    
    def evaluate(self, inputs):
        """
        Evaluate the circuit for given input values
        Returns the final output (0 or 1)
        """
        layer_outputs = []  # Stores outputs of each layer
        
        for layer in self.layers:
            current_outputs = []
            for gate in layer:
                # Get input values
                val1 = self._get_input_value(gate['input1'], inputs, layer_outputs)
                val2 = self._get_input_value(gate['input2'], inputs, layer_outputs)
                
                # Calculate gate output
                output = self._apply_gate(gate['type'], val1, val2)
                current_outputs.append(output)
            
            layer_outputs.append(current_outputs)
        
        # Final output is from the last gate in last layer
        return layer_outputs[-1][0]
    
    def _get_input_value(self, source, inputs, layer_outputs):
        """Get value from an input source"""
        if source[0] == 'input':
            return inputs[source[1]]
        elif source[0] == 'layer':
            return layer_outputs[source[1]][source[2]]
    
    def _apply_gate(self, gate_type, a, b):
        """Perform the logic gate operation"""
        a, b = bool(a), bool(b)
        if gate_type == 'and': return int(a and b)
        if gate_type == 'or': return int(a or b)
        if gate_type == 'xor': return int(a ^ b)
        if gate_type == 'xnor': return int(not (a ^ b))
        if gate_type == 'nand': return int(not (a and b))
        if gate_type == 'nor': return int(not (a or b))
        return 0
    
    def calculate_cost(self):
        """Calculate total cost of all gates in the circuit"""
        return sum(gate['cost'] for layer in self.layers for gate in layer)
    
    def visualize(self):
        """Generate a visual representation of the circuit"""
        diagram = []
        for i, layer in enumerate(self.layers):
            layer_diag = f"Layer {i}:\n"
            for j, gate in enumerate(layer):
                input1 = f"Input{gate['input1'][1]}" if gate['input1'][0] == 'input' else f"L{gate['input1'][1]}G{gate['input1'][2]}"
                input2 = f"Input{gate['input2'][1]}" if gate['input2'][0] == 'input' else f"L{gate['input2'][1]}G{gate['input2'][2]}"
                layer_diag += f"  G{j}: {gate['type'].upper()} ({input1}, {input2})\n"
            diagram.append(layer_diag)
        return "\n".join(diagram)

def calculate_fitness(circuit, truth_table):
    """
    Calculate fitness score for a circuit
    Higher is better (matches + (max_possible_cost - actual_cost))
    """
    matches = 0
    total_possible_cost = sum(GATE_TYPES.values()) * sum(len(layer) for layer in circuit.layers)
    
    for inputs, expected_output in truth_table.items():
        if circuit.evaluate(inputs) == expected_output:
            matches += 1
    
    # Fitness combines correctness and cost efficiency
    return matches + (total_possible_cost - circuit.calculate_cost())

def crossover(parent1, parent2):
    """Perform crossover between two parent circuits"""
    child = deepcopy(parent1)
    crossover_point = random.randint(1, parent1.num_layers-1)
    
    # Combine layers from both parents
    child.layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
    
    return child

def mutate(circuit):
    """Apply random mutation to the circuit"""
    mutated = deepcopy(circuit)
    layer_idx = random.randint(0, mutated.num_layers-1)
    gate_idx = random.randint(0, len(mutated.layers[layer_idx])-1)
    
    mutation_type = random.choice(['gate_type', 'input1', 'input2'])
    
    if mutation_type == 'gate_type':
        # Change gate type (possibly to complement)
        current_type = mutated.layers[layer_idx][gate_idx]['type']
        if random.random() < 0.5:  # 50% chance for complement
            new_type = COMPLEMENT_MAP.get(current_type, current_type)
        else:
            new_type = random.choice(list(GATE_TYPES.keys()))
        
        mutated.layers[layer_idx][gate_idx]['type'] = new_type
        mutated.layers[layer_idx][gate_idx]['cost'] = GATE_TYPES[new_type]
    
    elif mutation_type in ['input1', 'input2']:
        # Change one of the inputs
        new_input = mutated._generate_input(layer_idx)
        mutated.layers[layer_idx][gate_idx][mutation_type] = new_input
    
    return mutated

def read_truth_table(file_path):
    """
    Read truth table from file
    File format:
    Each line: input_bits output (e.g., "000 1")
    Returns: dictionary {(input_tuple): output}
    """
    truth_table = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                inputs = tuple(int(bit) for bit in parts[0])
                output = int(parts[1])
                truth_table[inputs] = output
    return truth_table

def genetic_algorithm(truth_table, num_layers=4, gates_per_layer=4, 
                     population_size=200, generations=200, 
                     mutation_rate=0.1, elitism=0.1):
    """
    Evolve a logic circuit to match the given truth table
    Args:
        truth_table: Dictionary of {(input_tuple): expected_output}
        num_layers: Number of circuit layers
        gates_per_layer: Gates per layer (except last)
        population_size: Number of circuits in population
        generations: Number of evolution generations
        mutation_rate: Probability of mutation
        elitism: Percentage of top performers to keep
    """
    # Determine number of inputs from truth table
    n = len(next(iter(truth_table.keys())))
    
    # Initialize population
    population = [LogicCircuit(n, num_layers, gates_per_layer) 
                  for _ in range(population_size)]
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [calculate_fitness(circuit, truth_table) 
                         for circuit in population]
        
        # Selection (elitism + tournament)
        elite_size = int(elitism * population_size)
        elite = sorted(zip(population, fitness_scores), 
                      key=lambda x: x[1], reverse=True)[:elite_size]
        elite = [circuit for circuit, _ in elite]
        
        new_population = elite.copy()
        
        # Fill remaining population with offspring
        while len(new_population) < population_size:
            # Tournament selection
            parents = random.sample(population, 2)
            parent1, parent2 = max(parents, key=lambda x: calculate_fitness(x, truth_table)), \
                              random.choice(parents)
            
            # Crossover
            child = crossover(parent1, parent2) if random.random() < 0.7 else deepcopy(parent1)
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Print progress
        best_circuit = max(population, key=lambda x: calculate_fitness(x, truth_table))
        best_fitness = calculate_fitness(best_circuit, truth_table)
        matches = sum(best_circuit.evaluate(inputs) == output 
                  for inputs, output in truth_table.items())
        
        print(f"Generation {generation+1}: {matches}/{len(truth_table)} correct, Cost: {best_circuit.calculate_cost()}")
        
        # Early termination if perfect match
        if matches == len(truth_table):
            print("\nPerfect solution found!")
            break
    
    # Return best circuit
    best_circuit = max(population, key=lambda x: calculate_fitness(x, truth_table))
    return best_circuit

def verify_circuit(circuit, truth_table):
    """Verify circuit against truth table and print results"""
    print("\nTruth Table Verification:")
    print("Inputs | Expected | Actual | Match")
    print("---------------------------------")
    
    matches = 0
    for inputs, expected in truth_table.items():
        actual = circuit.evaluate(inputs)
        match = "✓" if actual == expected else "✗"
        if match == "✓":
            matches += 1
        print(f"{''.join(map(str, inputs))} | {expected} | {actual} | {match}")
    
    print(f"\nSummary: {matches}/{len(truth_table)} correct ({matches/len(truth_table)*100:.2f}%)")
    print(f"Total gate cost: {circuit.calculate_cost()}")

if __name__ == "__main__":
    print("Logic Circuit Evolution using Genetic Algorithm")
    print("---------------------------------------------")
    
    # Get truth table from file
    file_path = input("Enter path to truth table file: ")
    truth_table = read_truth_table(file_path)
    
    # Get parameters from user
    num_layers = int(input("Number of layers (default 3): ") or 3)
    gates_per_layer = int(input("Gates per layer (default 4): ") or 4)
    population_size = int(input("Population size (default 100): ") or 100)
    generations = int(input("Number of generations (default 200): ") or 200)
    
    # Run genetic algorithm
    print("\nStarting evolution...")
    best_circuit = genetic_algorithm(
        truth_table,
        num_layers=num_layers,
        gates_per_layer=gates_per_layer,
        population_size=population_size,
        generations=generations
    )
    
    # Display results
    print("\nOptimized Circuit Structure:")
    print(best_circuit.visualize())
    
    # Verify against truth table
    verify_circuit(best_circuit, truth_table)
