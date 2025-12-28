#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

// --- OPTIMIZATION: Constant Memory for Lookups ---
// Powers of 3 for base-3 encoding (up to 3^8)
__constant__ int POW3[8] = { 1, 3, 9, 27, 81, 243, 729, 2187 };

// Direction encoding matches kernel.cu's switch statement exactly:
// dir=0: (0,-1), dir=1: (0,1), dir=2: (1,0), dir=3: (-1,0)
__constant__ int DIR_X[4] = { 0, 0, 1, -1 };
__constant__ int DIR_Y[4] = { -1, 1, 0, 0 };

// Precomputed 8-step rotation sequences for each starting direction
// SCAN_X[dir][step], SCAN_Y[dir][step] - computed from rotate_ccw(&cd, 4)
// dir=0 starts (0,-1): (0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)
// dir=1 starts (0,1):  (0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)
// dir=2 starts (1,0):  (1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)
// dir=3 starts (-1,0): (-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)
__constant__ int SCAN_X[4][8] = {
    { 0,  1,  1,  1,  0, -1, -1, -1},  // dir=0
    { 0, -1, -1, -1,  0,  1,  1,  1},  // dir=1
    { 1,  1,  0, -1, -1, -1,  0,  1},  // dir=2
    {-1, -1,  0,  1,  1,  1,  0, -1}   // dir=3
};
__constant__ int SCAN_Y[4][8] = {
    {-1, -1,  0,  1,  1,  1,  0, -1},  // dir=0
    { 1,  1,  0, -1, -1, -1,  0,  1},  // dir=1
    { 0,  1,  1,  1,  0, -1, -1, -1},  // dir=2
    { 0, -1, -1, -1,  0,  1,  1,  1}   // dir=3
};

// Turn lookup tables (precomputed from rotate_ccw)
// Left turn: rotate_ccw(&cd, 0.66) ≈ 273° CCW
// Right turn: rotate_ccw(&cd, 2) = 90° CCW
__constant__ int TURN_LEFT[4]  = { 3, 2, 0, 1 };  // 0→3, 1→2, 2→0, 3→1
__constant__ int TURN_RIGHT[4] = { 2, 3, 1, 0 };  // 0→2, 1→3, 2→1, 3→0

// Inverted Index for fast sensor lookup (fits in 64KB constant cache)
__constant__ int CONST_IX[6561];

typedef struct gene_struct {
    int next_state;
    int action;
    int m_fitness;
    int fitness;
    int used;
} gene;

typedef struct position {
    int x;
    int y;
} pos;

// --- HOST HELPERS ---
void initConstMemory(int* h_ix) {
    cudaMemcpyToSymbol(CONST_IX, h_ix, 6561 * sizeof(int));
}

void shuffle(int* arr, int S) {
    int i;
    for (i = 0; i < S; i++) arr[i] = i;
    for (i = S - 1; i > 0; i--) {
        int r = rand() % i;
        int t = arr[i];
        arr[i] = arr[r];
        arr[r] = t;
    }
}

// --- KERNELS ---

__global__ void __launch_bounds__(512) setup_states(curandState* state, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void __launch_bounds__(512) init_population(gene* pop, int G, int S, curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[id];
    int s = id * G;
    for (int i = s; i < s + G; i++) {
        int myrand = curand(&localState) % 100;
        if (myrand < 40) pop[i].action = 0;      // Forward
        else if (myrand < 70) pop[i].action = 1; // Left
        else pop[i].action = 2;                  // Right
        
        pop[i].next_state = curand(&localState) % S;
        pop[i].fitness = 0;
        pop[i].m_fitness = 0;
        pop[i].used = 0;
    }
    state[id] = localState;
}

__global__ void __launch_bounds__(512) generate_boards(int* boards, curandState* state, int R) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[id];
    int s = id * R * R;
    while (true) {
        // Clear board
        for (int i = 0; i < R * R; i++) boards[s + i] = 0;
        
        // Place 6 boxes
        int i = 0;
        while (i < 6) {
            int x = (curand(&localState) % (R - 2)) + 1;
            int y = (curand(&localState) % (R - 2)) + 1;
            if (boards[s + (x * R + y)] == 0) {
                boards[s + (x * R + y)] = 1;
                i++;
            }
        }
        
        // Check for 2x2 blocks (impossible condition)
        bool repeat = false;
        for (int x = 1; x < R - 2; x++) {
            for (int y = 1; y < R - 2; y++) {
                int idx = s + x * R + y;
                if (boards[idx] && boards[idx + 1] && boards[idx + R] && boards[idx + R + 1]) {
                    repeat = true;
                    break;
                }
            }
            if (repeat) break;
        }
        if (!repeat) break;
    }
    state[id] = localState;
}

// --- OPTIMIZED RUN KERNEL ---
// Launch bounds: 512 threads max, aim for 2 blocks per SM for RTX 5070
__global__ void __launch_bounds__(512, 2) run_boards_optimized(
    gene* pop,
    int* boards,
    curandState* state,
    int R,
    int G,
    int M, // Moves (80)
    int C, // Combinations (383)
    int* F,
    int* sum_occ,
    int* statistics) 
{
    // Shared memory: [Actions (G)] [NextStates (G)]
    extern __shared__ int shared_mem[];
    int* s_actions = &shared_mem[0];
    int* s_nexts   = &shared_mem[G];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid * blockDim.x + tid;
    
    // 1. Cooperative Load: Load Individual's FSM into Shared Memory
    int ind_offset = bid * G;
    for (int i = tid; i < G; i += blockDim.x) {
        s_actions[i] = pop[ind_offset + i].action;
        s_nexts[i]   = pop[ind_offset + i].next_state;
    }
    __syncthreads(); 

    // 2. Setup Board
    curandState localState = state[id];
    int brd_offset = id * R * R;
    
    // Find start position
    int px, py;
    while (true) {
        px = (curand(&localState) % (R - 2)) + 1;
        py = (curand(&localState) % (R - 2)) + 1;
        if (boards[brd_offset + (px * R + py)] == 0) break;
    }

    // Random Direction (0=N, 1=E, 2=S, 3=W)
    int dir = curand(&localState) % 4;

    int current_state = s_nexts[G - 1]; // Initial state gene
    
    // Path tracking for fitness credit (Max 80 moves)
    // We store the 'gene_index' used for each move to update stats later
    int path[80]; 

    // 3. Move Loop
    for (int move = 0; move < M; move++) {
        // Calculate Sensor Input using precomputed rotation table
        // SCAN_X[dir][n], SCAN_Y[dir][n] gives exact same result as rotate_ccw(&cd, 4)
        int sensor_val = 0;

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int nx = px + SCAN_X[dir][n];
            int ny = py + SCAN_Y[dir][n];

            int cell_val = 2; // Wall (out of bounds)
            if (nx >= 0 && ny >= 0 && nx < R && ny < R) {
                cell_val = boards[brd_offset + (nx * R + ny)];
            }

            sensor_val += cell_val * POW3[n];
        }

        int cc = CONST_IX[sensor_val];
        int gene_idx = current_state * C + cc;

        // Store path for later stats
        path[move] = gene_idx;

        int action = s_actions[gene_idx];
        int next_s = s_nexts[gene_idx];

        if (action == 0) { // Forward
            int fx = px + DIR_X[dir];
            int fy = py + DIR_Y[dir];
            if (fx >= 0 && fy >= 0 && fx < R && fy < R) {
                int content = boards[brd_offset + (fx * R + fy)];
                if (content == 0) {
                    px = fx; py = fy;
                } else if (content == 1) { // Box
                    int bx = fx + DIR_X[dir];
                    int by = fy + DIR_Y[dir];
                    if (bx >= 0 && by >= 0 && bx < R && by < R) {
                        if (boards[brd_offset + (bx * R + by)] == 0) {
                            boards[brd_offset + (fx * R + fy)] = 0;
                            boards[brd_offset + (bx * R + by)] = 1;
                            px = fx; py = fy;
                        }
                    }
                }
            }
        }
        else if (action == 1) { // Turn Left - use precomputed lookup
             dir = TURN_LEFT[dir];
        }
        else if (action == 2) { // Turn Right - use precomputed lookup
             dir = TURN_RIGHT[dir];
        }
        current_state = next_s;
    }

    // 4. Fitness Calculation - MUST match kernel_v3.cu exactly
    // A box at edge row gets +1, a box at edge column gets +1
    // Corner boxes get +2 (touching two walls)
    int f = 0;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            int ii = brd_offset + (i * R + j);
            if (boards[ii] == 1) {
                if (i % R == 0 || i % (R - 1) == 0) f++;
                if (j % R == 0 || j % (R - 1) == 0) f++;
            }
        }
    }
    F[id] = f;

    // 5. Update Gene Fitness (Design 2 requirement)
    // We use the 'path' array to credit genes that contributed to this score.
    // Use a small local bitmask to prevent double-counting per board (as per original logic)
    // G is max ~4600. We need ~144 ints for bitmask.
    int seen_mask[150]; // Covers up to G=4800
    for(int k=0; k<150; k++) seen_mask[k] = 0;
    
    for (int k = 0; k < M; k++) {
        int g_idx = path[k];
        int word = g_idx / 32;
        int bit  = g_idx % 32;
        
        if (!(seen_mask[word] & (1 << bit))) {
            // First time seeing this gene on this board
            seen_mask[word] |= (1 << bit);
            
            // Atomic updates to global memory
            // Note: This is the bottleneck, but required for Design 2
            atomicAdd(&(pop[ind_offset + g_idx]).fitness, f);
            atomicAdd(&(pop[ind_offset + g_idx]).used, 1);
            
            // Max fitness tracking
            int old = pop[ind_offset + g_idx].m_fitness;
            if (f > old) atomicMax(&(pop[ind_offset + g_idx]).m_fitness, f);
        }
    }
    state[id] = localState;
}

__global__ void __launch_bounds__(512) average_fitness(int P, int* F, float* avg_fit, int G) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int s = id * P;
    int sum = 0;
    for (int i = s; i < s + P; i++) sum += F[i];
    avg_fit[id] = (float)sum / (float)P;
}

__global__ void __launch_bounds__(512) crossover(
    int* idx, float* f, gene* pop, int G, int S, curandState* state,
    float ftmax, float ftbar, int* sum_occ, int option
) {
    int id = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    curandState localState = state[id];

    // Tournament Selection (Best 2 of 4)
    float max1 = -1.0f, max2 = -1.0f;
    int idx1 = -1, idx2 = -1;
    for (int i = id; i < id + 4; i++) {
        int indiv = idx[i];
        if (f[indiv] > max1) {
            max2 = max1; idx2 = idx1;
            max1 = f[indiv]; idx1 = indiv;
        } else if (f[indiv] > max2) {
            max2 = f[indiv]; idx2 = indiv;
        }
    }
    
    // Identify Losers (to replace)
    int min1 = -1, min2 = -1;
    for (int i = id; i < id + 4; i++) {
        int indiv = idx[i];
        if (indiv != idx1 && indiv != idx2) {
            if (min1 == -1) min1 = indiv;
            else min2 = indiv;
        }
    }

    // Adaptive Mutation Rate
    int k = 10;
    float pm1 = k * 0.5f / G;
    float pm2 = k * 0.5f / G;
    if (f[idx1] >= ftbar) pm1 = k * (ftmax - f[idx1]) / (ftmax - ftbar) / G;
    if (f[idx2] >= ftbar) pm2 = k * (ftmax - f[idx2]) / (ftmax - ftbar) / G;

    // Crossover Loop
    for (int i = 0; i < G; i++) {
        float v1 = 0, v2 = 0;
        
        // Design 2: Fitness-based
        if (option == 2) {
            if (pop[idx1 * G + i].used > 0)
                v1 = (float)pop[idx1 * G + i].fitness / pop[idx1 * G + i].used; // or m_fitness
            if (pop[idx2 * G + i].used > 0)
                v2 = (float)pop[idx2 * G + i].fitness / pop[idx2 * G + i].used;
        }
        
        // Parent Selection Logic
        int p1_action, p1_next, p2_action, p2_next;
        
        // Elitist / Weighted Selection between Parents
        if (v1 > v2 || (v1 == v2 && (curand(&localState) % 2 == 0))) {
            p1_action = pop[idx1 * G + i].action; p1_next = pop[idx1 * G + i].next_state;
            p2_action = pop[idx2 * G + i].action; p2_next = pop[idx2 * G + i].next_state;
        } else {
            p1_action = pop[idx2 * G + i].action; p1_next = pop[idx2 * G + i].next_state;
            p2_action = pop[idx1 * G + i].action; p2_next = pop[idx1 * G + i].next_state;
        }

        // Apply to Offspring 1
        if (curand_uniform(&localState) <= pm1) {
            pop[min1 * G + i].action = curand(&localState) % 3;
            pop[min1 * G + i].next_state = curand(&localState) % S;
        } else {
            pop[min1 * G + i].action = p1_action;
            pop[min1 * G + i].next_state = p1_next;
        }
        
        // Apply to Offspring 2
        if (curand_uniform(&localState) <= pm2) {
            pop[min2 * G + i].action = curand(&localState) % 3;
            pop[min2 * G + i].next_state = curand(&localState) % S;
        } else {
            pop[min2 * G + i].action = p2_action;
            pop[min2 * G + i].next_state = p2_next;
        }

        // Reset stats for next gen
        pop[idx1 * G + i].fitness = 0; pop[idx1 * G + i].used = 0; pop[idx1 * G + i].m_fitness = 0;
        pop[idx2 * G + i].fitness = 0; pop[idx2 * G + i].used = 0; pop[idx2 * G + i].m_fitness = 0;
        pop[min1 * G + i].fitness = 0; pop[min1 * G + i].used = 0; pop[min1 * G + i].m_fitness = 0;
        pop[min2 * G + i].fitness = 0; pop[min2 * G + i].used = 0; pop[min2 * G + i].m_fitness = 0;
    }
    state[id] = localState;
}

// --- MAIN ---
int main(int argc, char** argv) {
    // Default Arguments or CLI
    // RTX 5070 optimized defaults: larger population and more boards
    int N_mul = (argc > 1) ? atoi(argv[1]) : 8;   // Was 5, now 8 -> N=2048
    int S = (argc > 2) ? atoi(argv[2]) : 12;
    int P_mul = (argc > 3) ? atoi(argv[3]) : 2;   // P=512 boards per individual
    int T = (argc > 4) ? atoi(argv[4]) : 2;       // Design 2 (best performer)
    int L = (argc > 5) ? atoi(argv[5]) : 1;

    // RTX 5070 optimization: larger block size for better occupancy
    int block_size = 512;  // Was 256, RTX 5070 handles larger blocks well
    int N = 256 * N_mul;   // Population size
    int P = 128 * P_mul;   // Boards per individual (same as paper)
    int C = 383;
    int G = S * C + 1;
    int R = 6;
    int K = 2000; // Generations

    printf("kernel_2026 (LUT): N=%d, S=%d, P=%d, Design=%d\n", N, S, P, T);
    srand(time(0));

    // File Setup
    char fname[64], bname[64];
    sprintf(fname, "txt/r-%d-%d-%d-%d-%d.txt", N, P, S, T, L);
    sprintf(bname, "txt/b-%d-%d-%d-%d-%d.txt", N, P, S, T, L);
    FILE *results = fopen(fname, "w");
    FILE *bestf = fopen(bname, "w");
    if (!results || !bestf) {
        printf("Error opening files! Create 'txt' folder.\n");
        return 1;
    }

    // Memory Allocation
    gene *pop, *best;
    cudaMallocManaged(&pop, N * G * sizeof(gene));
    cudaMallocManaged(&best, G * sizeof(gene));

    int *F, *boards, *statistics, *sum_occ, *idx;
    float *arr_avgfit;
    curandState *devStates;
    
    cudaMallocManaged(&F, N * P * sizeof(int));
    cudaMallocManaged(&boards, N * P * R * R * sizeof(int));
    cudaMallocManaged(&statistics, N * C * sizeof(int));
    cudaMallocManaged(&sum_occ, N * G * sizeof(int)); // Not used in opt kernel, kept for compat
    cudaMallocManaged(&arr_avgfit, N * sizeof(float));
    cudaMallocManaged(&idx, N * sizeof(int));
    cudaMalloc((void**)&devStates, N * P * sizeof(curandState));

    // Initialize Constant Memory Index
    int iidx[383] = { 0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546 };
    int *h_ix = (int*)malloc(6561 * sizeof(int));
    memset(h_ix, 0, 6561 * sizeof(int));
    for(int k=0; k<383; k++) h_ix[iidx[k]] = k;
    initConstMemory(h_ix);
    free(h_ix);

    // Initial Setup Kernel
    int setup_blocks = (N * P + block_size - 1) / block_size;
    setup_states<<<setup_blocks, block_size>>>(devStates, time(0));
    
    int pop_blocks = (N + block_size - 1) / block_size;
    init_population<<<pop_blocks, block_size>>>(pop, G, S, devStates);
    cudaDeviceSynchronize();

    float best_ind_fitness = 0.0f;
    float best_gen_fitness = 0.0f;
    float convergence = 0.0f;
    int gen = 0;
    time_t start = time(0);

    // --- EVOLUTION LOOP ---
    // Stop when max generations reached OR population converged (avg/best > 0.97)
    while (gen < K && convergence < 0.99) {
        // 1. Generate Boards
        generate_boards<<<N, P>>>(boards, devStates, R);
        
        // 2. Run Boards (Optimized)
        // Shared mem size = (2 * G) integers
        int shmem_size = (2 * G) * sizeof(int);
        run_boards_optimized<<<N, P, shmem_size>>>(pop, boards, devStates, R, G, 80, C, F, sum_occ, statistics);
        
        // 3. Average Fitness
        int avg_blocks = (N + block_size - 1) / block_size;
        average_fitness<<<avg_blocks, block_size>>>(P, F, arr_avgfit, G);
        cudaDeviceSynchronize();

        // 4. CPU Stats (Find Best)
        float gen_fitness = 0.0f;
        float ind_fitness = 0.0f;
        
        for (int j = 0; j < N; j++) {
            gen_fitness += arr_avgfit[j];
            if (arr_avgfit[j] > ind_fitness) {
                ind_fitness = arr_avgfit[j];
            }
            // Global Best Tracking
            if (arr_avgfit[j] > best_ind_fitness) {
                best_ind_fitness = arr_avgfit[j];
                // Copy best to host
                memcpy(best, &pop[j * G], G * sizeof(gene));
            }
        }
        gen_fitness /= (float)N;
        if (gen_fitness > best_gen_fitness) best_gen_fitness = gen_fitness;

        // Progress logging (no dots - they slow things down)
        if (gen % 100 == 0) printf("Gen %d: Best=%.2f Avg=%.2f\n", gen, ind_fitness, gen_fitness);
        fprintf(results, "%0.2f ", gen_fitness);
        
        convergence = gen_fitness / (ind_fitness + 0.0001f);
        
        // 5. Crossover
        shuffle(idx, N);
        int cross_blocks = ((N / 4) + block_size - 1) / block_size;
        int cross_threads = (N / 4 < block_size) ? N / 4 : block_size;
        if (cross_blocks == 0) cross_blocks = 1;

        crossover<<<cross_blocks, cross_threads>>>(idx, arr_avgfit, pop, G, S, devStates, best_ind_fitness, best_gen_fitness, sum_occ, T);
        
        gen++;
    }

    // --- RESULTS ---
    printf("\n\nFinished!\nBest Individual: %.4f\nBest Gen Mean: %.4f\n", best_ind_fitness, best_gen_fitness);
    printf("Time: %lld seconds\n", time(0) - start);

    // Save Best Agent
    for (int i = 0; i < G; i++) {
        fprintf(bestf, "%d %d ", best[i].action, best[i].next_state);
    }
    
    fclose(results);
    fclose(bestf);
    
    // Cleanup
    cudaFree(pop); cudaFree(best); cudaFree(F); cudaFree(boards);
    cudaFree(statistics); cudaFree(sum_occ); cudaFree(arr_avgfit);
    cudaFree(idx); cudaFree(devStates);
    
    return 0;
}