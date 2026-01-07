# Tartarus

GPU-accelerated evolution of Finite State Machines (FSMs) to solve the Tartarus problem.

## The Tartarus Problem

Tartarus is a benchmark problem for evaluating artificial agents. An agent operates on a 6x6 grid containing 6 boxes. The goal is to push boxes to the walls (edges) of the grid within 80 moves.

**Scoring:**
- Box on edge (not corner): +1 point
- Box in corner: +2 points
- Maximum possible score: 10 points (4 boxes in corners + 2 on edges)

**Agent capabilities:**
- Sees 8 neighboring cells (each can be: empty, box, or wall)
- 3 actions: move forward, turn left, turn right
- Can push a box if the cell behind it is empty

**Problem space:**
- 383 valid sensory combinations (out of 3^8 = 6561 possible)
- 1,869 unique board layouts (6 boxes in 4x4 inner grid)
- 74,760 total configurations (1,869 layouts × 10 starting positions × 4 directions)

## Files

### Training Kernels

 - kernel.cu - Kernel developed in 2019-2020 using CUDA on RTX 1050... [^2]
 - kernel_2026.cu - Kernel updated to take advantage of CUDA on RTX 5070
 - kernel_allboards.cu - 2026 kernel updated to use all 74760 boards during training 
 - kernel_D2_allboards.cu - This version uses the D2 design from 2020 paper. 
 - kernel_test_all.cu - Runs a solution on all boards, and prints statistics, such as state and score distribution

Compile using `nvcc -arch=sm_120 -o Tartarus74K.exe kernel_allboards.cu -lcurand` - update the input and output filenames as needed. The `-arch=sm_120` flag optimizes for RTX 50 series (Blackwell). Use `-arch=sm_89` for RTX 40 series, `-arch=sm_86` for RTX 30 series. 

### Visualization (Python)

| File | Description | Requirements |
|------|-------------|--------------|
| `visualize_analysis.py` | Generate heatmaps, network graphs, state profiles | `pip install pandas numpy matplotlib seaborn networkx` |
| `combo_lookup.py` | Look up what each of the 383 combinations looks like | None (standard library) |

### Data Files

The realboard.txt contains all 1,869 unique board layouts. Start the agent in the remaining 10 positions with a random orientation. This results in 1869 * 4 * 10 = 74760 unique boards [^2]. The boards.txt contains the boards before we showed that the number of boards is 74760 in the 2020 paper [^1][^2]. 
## Usage

### Training with Sampled Boards (kernel_2026.cu)

```bash
nvcc -o Tartarus2026.exe kernel_2026.cu -lcurand
Tartarus2026.exe <N1> <S> <K> <P> <T1> <L>
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| N1 | Population size multiplier (×256) | 5 → 1280 individuals |
| S | Number of FSM states | 16, 20, 24 |
| K | Number of generations | 1000 |
| P | Boards per test multiplier (×128) | 4 → 512 boards |
| T1 | Tournament size | 1 |
| L | Run ID (for file naming) | 1 |

Example:
```bash
Tartarus2026.exe 5 16 1000 4 1 1
```

### Training on All Boards (kernel_allboards.cu)

```bash
nvcc -o Tartarus74K.exe kernel_allboards.cu -lcurand
Tartarus74K.exe <N> <S> <K> <L>
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| N | Population size | 512, 1024 |
| S | Number of FSM states | 20, 24, 32 |
| K | Number of generations | 4000, 6000 |
| L | Run ID | 1 |

Example:
```bash
Tartarus74K.exe 1024 24 6000 1
```

Output files:
- `txt/r-all-N-S-L.txt` - Results per generation
- `txt/b-all-N-S-L.txt` - Best solution

### Testing a Solution

```bash
nvcc -o TartarusTestAll.exe kernel_test_all.cu -lcurand
TartarusTestAll.exe <solution_file> <S>
```

Example:
```bash
TartarusTestAll.exe txt/b-all-1024-24-1.txt 24
```

Output includes:
- Average score across all 74,760 boards
- Score histogram (how many boards get 10, 9, 8, ... 0)
- State usage statistics

### Analyzing a Solution

```bash
nvcc -o TartarusAnalyze.exe kernel_analyze.cu
TartarusAnalyze.exe <solution_file> <S>
```

Example:
```bash
TartarusAnalyze.exe txt/b-all-1024-24-1.txt 24
```

Generates CSV files for:
- State-combination heatmap
- State transition matrix
- Push statistics per state
- Memory usage analysis

### Visualizing Analysis Results

```bash
python visualize_analysis.py <S> [output_prefix]
```

Example:
```bash
python visualize_analysis.py 24 brain_24state
```

Generates PNG visualizations:
- `brain_24state_heatmap.png` - State × Combination usage
- `brain_24state_transitions.png` - State transition network
- `brain_24state_state_profiles.png` - Mover vs Turner states
- `brain_24state_push_analysis.png` - Which states push boxes
- `brain_24state_communities.png` - State clusters
- `brain_24state_memory_usage.png` - State-dependent behavior

## Results

| Configuration | States | Score | Notes |
|---------------|--------|-------|-------|
| 2020 Paper (baseline) | 8 | 8.54 | Sampled training |
| kernel_2026 (P=512) | 16 | 8.85 | Training fitness (overfit) |
| kernel_allboards | 16 | 8.77 | True fitness |
| kernel_allboards | 20 | 8.96 | True fitness |
| kernel_allboards | 24 | 9.32 | **Current best** |

## FSM Structure

Each FSM solution consists of:
- `S` states (e.g., 24)
- `383` valid sensor combinations
- `G = S × 383 + 1` genes total

Each gene contains:
- `action`: 0 (forward), 1 (turn left), 2 (turn right)
- `next_state`: which state to transition to (0 to S-1)

The last gene (`G-1`) stores the initial state.

## References

Here are all the related references, both by Kaya Oğuz and others. 

[^1]: Kaya Oğuz, Adaptive Evolution of Finite State Machines for the Tartarus Problem, 2019 Innovations in Intelligent Systems and Applications Conference (ASYU), 2019, pp. 1-5, doi: 10.1109/ASYU48272.2019.8946413
[^2]: Kaya Oğuz, True scores for tartarus with adaptive GAs that evolve FSMs on GPU, Information Sciences, 2020, pp. 1-15, doi: 10.1016/j.ins.2020.03.072.
[^3]: Kaya Oğuz, Estimating the difficulty of Tartarus instances, Pamukkale Univ Muh Bilim Derg, 2021, pp. 114-121, doi: 10.5505/pajes.2020.00515.
