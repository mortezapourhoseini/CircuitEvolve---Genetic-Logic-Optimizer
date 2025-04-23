# CircuitEvolve

A genetic algorithm-based tool for evolving optimal logic circuits that match a given truth table. 
This project automatically designs efficient digital circuits using evolutionary computation principles.

## Features 
- Evolves logic circuits using genetic algorithms
- Supports user-defined truth tables (via file input)
- Configurable parameters for circuit structure and evolution
- Optimizes for both correctness and hardware cost
- Detailed verification against truth table
- Circuit visualization with layer/gate details



## Usage 
1. Prepare a truth table file (see example format below)
2. Run the program:
   ```bash
   python circuit_evolve.py
   ```
3. Follow interactive prompts to:
   - Specify truth table file path
   - Set algorithm parameters (or use defaults)
   - View optimized circuit structure
   - Verify performance against truth table

## Truth Table Format 📝
Create a text file with one entry per line:  
`[input_bits] [output]`  
Example (`example.txt`):
```
000 0
001 1
010 1
011 0
100 1
101 0
110 0
111 1
```

## Key Parameters ⚙️
| Parameter         | Default | Description                          |
|-------------------|---------|--------------------------------------|
| Number of Layers  | 3       | Circuit depth                        |
| Gates per Layer   | 4       | Gates in each layer (except output)  |
| Population Size   | 100     | Number of circuits per generation    |
| Generations       | 200     | Maximum evolution iterations         |
| Mutation Rate     | 10%     | Probability of mutations             |

## Sample Output 🖥️
```
Optimized Circuit Structure:
Layer 0:
  G0: XOR (Input0, Input2)
  G1: OR (Input1, Input2)
Layer 1:
  G0: AND (L0G0, L0G1)

Truth Table Verification:
Inputs | Expected | Actual | Match
000 | 0 | 0 | ✓
001 | 1 | 1 | ✓
...
111 | 1 | 1 | ✓

Summary: 8/8 correct (100.00%)
Total gate cost: 4
```
