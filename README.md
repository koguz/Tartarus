# Tartarus

GPU-accelerated evolution of Finite State Machines (FSMs) to solve the Tartarus problem, and a behavior discovery methodology to explain the resulting agent's policy. 

A 128-state FSM is evolved using a genetic algorithm on CUDA, trained on all 74,760 valid board configurations, achieving a true score of **9.84** where 90% of boards solved with a perfect 10. The agent's policy is analyzed through spectral clustering of a tactic transition graph and Infomap community detection on the state transition graph. 

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
- 74,760 total configurations (1,869 layouts x 10 starting positions x 4 directions)

## Repository Structure

### CUDA Training Kernels

| File | Description |
|------|-------------|
| `kernel.cu` | Original kernel from the 2020 paper, developed for RTX 1050 |
| `kernel_2026.cu` | Updated kernel for modern GPUs (RTX 50 series) with sampled board training |
| `kernel_allboards.cu` | Trains on all 74,760 boards using the baseline D0 design |
| `kernel_D2_allboards.cu` | **Main training kernel.** Trains on all 74,760 boards using the D2 design (per-gene fitness crossover). Produced the 9.84 agent |
| `kernel_test_all.cu` | Evaluates a trained solution on all 74,760 boards. Outputs score distribution and per-state usage statistics |
| `kernel_analyze.cu` | Generates CSV files for state-combination heatmaps, transition matrices, and push statistics |

### Python Analysis (used in the paper)

| File | Description |
|------|-------------|
| `analyze_agent.py` | Runs the FSM agent on all 74,760 boards in Python. Builds the state transition graph, combination transition graph, and tactic transition graph. Saves sequences and statistics as `.json` and `.pkl` files |
| `behavior_segmentation.py` | Spectral clustering on the tactic transition graph. Combines transition probabilities with Hamming similarity (controlled by lambda) to discover behavioral clusters |
| `lambda_sensitivity.py` | Sweeps lambda from 0.0 to 1.0 to validate that the number of clusters is stable |
| `state_clustering.py` | InfoMap state clustering |
| `print_cluster_tactics.py` | Extracts and prints the dominant tactic sequences within each behavioral cluster |
| `analyze_communities.py` | Infomap community detection on the state transition graph. Identifies state modules based on transition flow |
| `null_model.py` | Null model test for behavior-state alignment. Generates 10,000 random partitions and computes ARI/NMI distributions to test significance |
| `ablation_study.py` | Lesion study: removes state clusters and evaluates the agent on all boards. Transitions into ablated states are redirected to random non-ablated states |

### Python Visualization

| File | Description |
|------|-------------|
| `viewer_2026.py` | Step-by-step visual replay of the agent on a Tartarus board. Saves each step as a PNG image with state/action annotations |
| `visualize_graph.py` | Visualizes the state transition graph with node sizes proportional to visit frequency and edge thickness proportional to transition weight. Supports interactive HTML output via PyVis |
| `visualize_tactic_pattern.py` | Renders tactic sequences as 3x3 grids showing the agent's perception and action at each step |
| `visualize_analysis.py` | Generates heatmaps, network graphs, and behavioral profiles from `kernel_analyze.cu` CSV output |

### Exploratory Scripts (not in the paper)

| File | Description |
|------|-------------|
| `analyze_behaviors.py` | Earlier behavior analysis approach |
| `analyze_combo_behaviors.py` | Combination-level behavior analysis |
| `behavior_cluster_decoder.py` | Attempts to decode actions within behavior clusters |
| `segment_behaviors.py` | Alternative segmentation approach |
| `segment_by_independence.py` | Independence-based segmentation |
| `sequitur_analysis.py` | Sequitur grammar induction on action sequences |
| `find_patterns.py` | Pattern finding in state sequences |
| `combo_find_patterns.py` | Pattern finding in combination sequences |
| `tactic_find_patterns.py` | Pattern finding in tactic sequences |
| `query_state.py` | Interactive query tool for inspecting individual states |
| `visualize_brain_map.py` | Brain-map-style visualization of state modules |
| `visualize_combo_graph.py` | Visualizes the combination transition graph |
| `analyze_iidx.py` | Compares inverted index arrays between kernels |
| `viewer.py` | Original step-by-step viewer (2020, hardcoded Windows paths) |
| `viewer_comparison.py` | Side-by-side comparison of two agents on the same board |
| `testing.py` | Python-based fitness evaluation on random boards (2020) |

### Data Files

| File | Description |
|------|-------------|
| `realboard.txt` | All 1,869 unique board layouts |
| `boards.txt` | Board file from before the unique board count was established |
| `best/b-D2-4096-128-3000-1.txt` | Best agent chromosome (the 9.84 agent) |
| `best/r-D2-4096-128-3000-1.txt` | Training log (generation, best, average) |
| `best/sc_txt_b-D2-4096-128-3000-1.txt` | Score distribution from `kernel_test_all` |
| `best/st_txt_b-D2-4096-128-3000-1.txt` | Per-state usage statistics |
| `best/results_txt_b-D2-4096-128-3000-1.txt` | Per-board results |
| `images/` | Agent and box sprites for the viewer |

## Requirements

**CUDA (for training and testing):**
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)

**Python (for analysis and visualization):**

```
pip install -r requirements.txt
```

Key libraries: `numpy`, `matplotlib`, `networkx`, `infomap`, `scikit-learn`, `scipy`, `seaborn`, `pandas`, `pillow`, `pyvis`

## Usage

### Step 1: Train an Agent

Compile and run the D2 all-boards kernel:

```bash
nvcc -arch=sm_120 -o TartarusD2.exe kernel_D2_allboards.cu -lcurand
./TartarusD2.exe <N> <S> <K> <L>
```

| Parameter | Description | Paper value |
|-----------|-------------|-------------|
| N | Population size | 4096 |
| S | Number of FSM states | 128 |
| K | Number of generations | 3000 |
| L | Run ID (for file naming) | 1 |

```bash
./TartarusD2.exe 4096 128 3000 1
```

Use `-arch=sm_120` for RTX 50 series (Blackwell), `-arch=sm_89` for RTX 40 series, `-arch=sm_86` for RTX 30 series.

Output files are saved to the `txt/` directory:
- `txt/r-D2-N-S-K-L.txt` -- Training log (generation, best score, average score)
- `txt/b-D2-N-S-K-L.txt` -- Best agent chromosome

### Step 2: Evaluate the Agent

```bash
nvcc -arch=sm_120 -o TartarusTestAll.exe kernel_test_all.cu -lcurand
./TartarusTestAll.exe <solution_file> <S>
```

```bash
./TartarusTestAll.exe best/b-D2-4096-128-3000-1.txt 128
```

Outputs:
- Average true score across all 74,760 boards
- `sc_<solution_file>` -- Score distribution (how many boards scored 10, 9, 8, ...)
- `st_<solution_file>` -- Per-state usage counts
- `results_<solution_file>` -- Per-board scores

### Step 3: Build Analysis Data

Run the agent in Python on all 74,760 boards to collect state, combination, and tactic transition graphs:

```bash
python analyze_agent.py best/b-D2-4096-128-3000-1.txt 128
```

This produces `analysis_*.json` and `analysis_*.pkl` files used by subsequent scripts.

### Step 4: Behavior Discovery (Spectral Clustering)

Sweep lambda values and find k. 

```bash
python lambda_sensitivity.py
```

Cluster the 64 tactics into behavioral groups:

```bash
python behavior_segmentation.py --lambda 0.25
```

To see the dominant tactic sequences in each cluster:

```bash
python print_cluster_tactics.py
```

### Step 5: State Module Detection (Infomap) and Alignment

Detect communities in the state transition graph using k:

```bash
python state_clustering.py --k 7
```

Test whether state modules and behavior clusters are aligned:

```bash
python null_model.py
```

### Step 6: Ablation

Measure the impact of removing state modules:

```bash
python ablation_study.py best/b-D2-4096-128-3000-1.txt 128
```

### Visualization

Replay the agent step by step on a board:

```bash
python viewer_2026.py best/b-D2-4096-128-3000-1.txt 128
```

Visualize the state transition graph:

```bash
python visualize_graph.py [--interactive]
```

Visualize a tactic sequence:

```bash
python visualize_tactic_pattern.py --numeric 34-34-35
```

## FSM Structure (D2 Design)

Each individual in the population consists of:
- `S` states (e.g., 128)
- `383` valid sensor combinations
- `G = S x 383 + 1` genes total

Each gene is a tuple of `(action, next_state, used, fitness)`:
- `action`: 0 (forward), 1 (turn left), 2 (turn right)
- `next_state`: which state to transition to (0 to S-1)
- `used`: how many times this gene is activated during evaluation
- `fitness`: cumulative score contribution of this gene

The last gene (`G-1`) stores the initial state. During crossover, offspring inherit the gene with the higher fitness-to-usage ratio, ensuring beneficial genes are preserved.

## Results

| Configuration | States | Boards | Design | True Score |
|---------------|--------|--------|--------|------------|
| 2020 Paper | 12 | 256 (sampled) | D2 | 8.54 |
| This paper | 128 | 74,760 (all) | D2 | **9.84** |

Score distribution for the 9.84 agent:

| Score | Boards | Percentage |
|-------|--------|------------|
| 10 | 67,499 | 90.2867% |
| 9 | 5,903 | 7.8959% |
| 8 | 520 | 0.6956% |
| 7 | 50 | 0.0669% |
| 6 | 1 | 0.0013% |
| 5 | 1 | 0.0013% |
| 4 | 666 | 0.8909% |
| 3 | 97 | 0.1297% |
| 2 | 23 | 0.0308% |
| 1 | 0 | 0.00% |
| 0 | 0 | 0.00% |

## References

[1] K. Oguz, "Adaptive Evolution of Finite State Machines for the Tartarus Problem," *2019 Innovations in Intelligent Systems and Applications Conference (ASYU)*, 2019, pp. 1-5, doi: 10.1109/ASYU48272.2019.8946413

[2] K. Oguz, "True scores for tartarus with adaptive GAs that evolve FSMs on GPU," *Information Sciences*, 2020, pp. 1-15, doi: 10.1016/j.ins.2020.03.072

[3] K. Oguz, "Estimating the difficulty of Tartarus instances," *Pamukkale Univ Muh Bilim Derg*, 2021, pp. 114-121, doi: 10.5505/pajes.2020.00515
