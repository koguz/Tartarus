#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA Kernel Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// --- CONSTANTS ---
#define NUM_BOARDS 1869
#define CONFIGS_PER_BOARD 40
#define TOTAL_CONFIGS 74760  // 1869 * 40
#define R 6

__constant__ int POW3[8] = { 1, 3, 9, 27, 81, 243, 729, 2187 };
__constant__ int DIR_X[4] = { 0, 0, 1, -1 };
__constant__ int DIR_Y[4] = { -1, 1, 0, 0 };
__constant__ int SCAN_X[4][8] = {
    { 0,  1,  1,  1,  0, -1, -1, -1},
    { 0, -1, -1, -1,  0,  1,  1,  1},
    { 1,  1,  0, -1, -1, -1,  0,  1},
    {-1, -1,  0,  1,  1,  1,  0, -1}
};
__constant__ int SCAN_Y[4][8] = {
    {-1, -1,  0,  1,  1,  1,  0, -1},
    { 1,  1,  0, -1, -1, -1,  0,  1},
    { 0,  1,  1,  1,  0, -1, -1, -1},
    { 0, -1, -1, -1,  0,  1,  1,  1}
};
__constant__ int TURN_LEFT[4]  = { 3, 2, 0, 1 };
__constant__ int TURN_RIGHT[4] = { 2, 3, 1, 0 };
__constant__ int CONST_IX[6561];

// Coord mapping for inner 4x4 grid
__constant__ int COORD[16] = { 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28 };

typedef struct gene_struct {
    int next_state;
    int action;
    int m_fitness;
    int fitness;
    int used;
} gene;

void initConstMemory(int* h_ix) {
    cudaMemcpyToSymbol(CONST_IX, h_ix, 6561 * sizeof(int));
}

void shuffle(int* arr, int S) {
    for (int i = 0; i < S; i++) arr[i] = i;
    for (int i = S - 1; i > 0; i--) {
        int r = rand() % i;
        int t = arr[i]; arr[i] = arr[r]; arr[r] = t;
    }
}

// --- KERNELS ---

__global__ void setup_states(curandState* state, unsigned long long seed, int count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) curand_init(seed, id, 0, &state[id]);
}

__global__ void init_population(gene* pop, int G, int S, curandState* state, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;
    curandState localState = state[id];
    int s = id * G;
    for (int i = s; i < s + G; i++) {
        int myrand = curand(&localState) % 100;
        if (myrand < 40) pop[i].action = 0;
        else if (myrand < 70) pop[i].action = 1;
        else pop[i].action = 2;
        pop[i].next_state = curand(&localState) % S;
        pop[i].fitness = 0;
        pop[i].m_fitness = 0;
        pop[i].used = 0;
    }
    state[id] = localState;
}

// Run one individual on one board configuration
// Grid: (TOTAL_CONFIGS, N), Block: 1 thread
// Or Grid: ceil(TOTAL_CONFIGS * N / 256), Block: 256
__global__ void run_all_boards(
    gene* pop,
    char* base_boards,  // 1869 * 36 board layouts
    int* F,             // N * TOTAL_CONFIGS fitness scores
    int G,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * TOTAL_CONFIGS;
    if (idx >= total) return;

    int ind = idx / TOTAL_CONFIGS;  // which individual (0 to N-1)
    int cfg = idx % TOTAL_CONFIGS;  // which config (0 to 74759)

    int board_idx = cfg / 40;       // which board layout (0 to 1868)
    int dir = (cfg % 40) / 10;      // direction (0-3)
    int pos_idx = (cfg % 40) % 10;  // position index (0-9)

    // Copy board to local memory
    char board[36];
    for (int i = 0; i < 36; i++) {
        board[i] = base_boards[board_idx * 36 + i];
    }

    // Find starting position (pos_idx-th empty cell)
    int cpx = 0, cpy = 0;
    int empty_count = 0;
    for (int i = 0; i < 16; i++) {
        if (board[COORD[i]] == 0) {
            if (empty_count == pos_idx) {
                cpy = COORD[i] / 6;
                cpx = COORD[i] % 6;
                break;
            }
            empty_count++;
        }
    }

    // Get individual's genes
    gene* my_genes = &pop[ind * G];
    int cs = my_genes[G - 1].next_state;  // initial state

    // Run 80 moves
    for (int move = 0; move < 80; move++) {
        // Compute sensor input
        int cc = 0;
        for (int m = 0; m < 8; m++) {
            int sx = cpx + SCAN_X[dir][m];
            int sy = cpy + SCAN_Y[dir][m];
            int val = (sx < 0 || sy < 0 || sx >= 6 || sy >= 6) ? 2 : board[sx * 6 + sy];
            cc += POW3[m] * val;
        }
        cc = CONST_IX[cc];

        int action = my_genes[cs * 383 + cc].action;
        cs = my_genes[cs * 383 + cc].next_state;

        // Execute action
        if (action == 0) {  // Forward
            int cx = cpx + DIR_X[dir];
            int cy = cpy + DIR_Y[dir];
            if (cx >= 0 && cy >= 0 && cx < 6 && cy < 6) {
                if (board[cx * 6 + cy] == 0) {
                    cpx = cx; cpy = cy;
                } else {
                    int dx = cx + DIR_X[dir];
                    int dy = cy + DIR_Y[dir];
                    if (dx >= 0 && dy >= 0 && dx < 6 && dy < 6 && board[dx * 6 + dy] == 0) {
                        board[cx * 6 + cy] = 0;
                        board[dx * 6 + dy] = 1;
                        cpx = cx; cpy = cy;
                    }
                }
            }
        } else if (action == 1) {
            dir = TURN_LEFT[dir];
        } else if (action == 2) {
            dir = TURN_RIGHT[dir];
        }
    }

    // Calculate fitness
    int f = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (board[i * 6 + j] == 1) {
                if (i == 0 || i == 5) f++;
                if (j == 0 || j == 5) f++;
            }
        }
    }
    F[idx] = f;
}

// Average fitness for each individual
__global__ void average_fitness(int* F, float* avg_fit, int N) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= N) return;

    int sum = 0;
    int offset = ind * TOTAL_CONFIGS;
    for (int i = 0; i < TOTAL_CONFIGS; i++) {
        sum += F[offset + i];
    }
    avg_fit[ind] = (float)sum / (float)TOTAL_CONFIGS;
}

// Tournament crossover (simplified)
__global__ void crossover(
    int* idx_arr,
    float* avg_fit,
    gene* pop,
    int G, int S,
    curandState* state,
    float best_ind_fit,
    float best_gen_fit,
    int N
) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= N / 4) return;

    curandState localState = state[group];
    int base = group * 4;

    // Find best 2 and worst 2 in group of 4
    float fits[4];
    int ids[4];
    for (int i = 0; i < 4; i++) {
        ids[i] = idx_arr[base + i];
        fits[i] = avg_fit[ids[i]];
    }

    // Simple bubble sort
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (fits[j] > fits[i]) {
                float tf = fits[i]; fits[i] = fits[j]; fits[j] = tf;
                int ti = ids[i]; ids[i] = ids[j]; ids[j] = ti;
            }
        }
    }

    int p1 = ids[0], p2 = ids[1];  // parents (best 2)
    int c1 = ids[2], c2 = ids[3];  // children (worst 2)

    // Uniform crossover
    for (int i = 0; i < G; i++) {
        gene g1 = pop[p1 * G + i];
        gene g2 = pop[p2 * G + i];

        if (curand(&localState) % 2 == 0) {
            pop[c1 * G + i] = g1;
            pop[c2 * G + i] = g2;
        } else {
            pop[c1 * G + i] = g2;
            pop[c2 * G + i] = g1;
        }

        // Mutation
        float f_c1 = avg_fit[c1];
        float pm = (f_c1 >= best_gen_fit) ?
            10.0f * (best_ind_fit - f_c1) / (best_ind_fit - best_gen_fit + 0.001f) / G :
            5.0f / G;

        if ((curand(&localState) % 1000) / 1000.0f < pm) {
            int myrand = curand(&localState) % 100;
            if (myrand < 40) pop[c1 * G + i].action = 0;
            else if (myrand < 70) pop[c1 * G + i].action = 1;
            else pop[c1 * G + i].action = 2;
            pop[c1 * G + i].next_state = curand(&localState) % S;
        }
        if ((curand(&localState) % 1000) / 1000.0f < pm) {
            int myrand = curand(&localState) % 100;
            if (myrand < 40) pop[c2 * G + i].action = 0;
            else if (myrand < 70) pop[c2 * G + i].action = 1;
            else pop[c2 * G + i].action = 2;
            pop[c2 * G + i].next_state = curand(&localState) % S;
        }
    }
    state[group] = localState;
}

// Copy best individual
__global__ void copy_best(gene* pop, gene* best, int G, int best_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < G) {
        best[i] = pop[best_idx * G + i];
    }
}

// --- MAIN ---
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 64;    // Population size (small!)
    int S = (argc > 2) ? atoi(argv[2]) : 16;    // States
    int K = (argc > 3) ? atoi(argv[3]) : 500;   // Generations
    int L = (argc > 4) ? atoi(argv[4]) : 1;     // Run ID

    int C = 383;
    int G = S * C + 1;

    printf("All-Boards Training: N=%d, S=%d, K=%d generations\n", N, S, K);
    printf("Testing on ALL %d board configurations per individual\n", TOTAL_CONFIGS);
    srand(time(0));

    // Load boards from realboard.txt
    char* h_boards = (char*)malloc(NUM_BOARDS * 36 * sizeof(char));
    memset(h_boards, 0, NUM_BOARDS * 36);

    FILE* bf = fopen("realboard.txt", "r");
    if (!bf) {
        printf("Error: Cannot open realboard.txt\n");
        return -1;
    }

    char line[20];
    int board_count = 0;
    while (fgets(line, sizeof(line), bf) && board_count < NUM_BOARDS) {
        for (int j = 0; j < 16 && line[j] != '\0' && line[j] != '\n'; j++) {
            int coord_idx = 7 + (j / 4) * 6 + (j % 4);  // Map to 6x6 grid
            h_boards[board_count * 36 + coord_idx] = (line[j] == '1') ? 1 : 0;
        }
        board_count++;
    }
    fclose(bf);
    printf("Loaded %d board layouts\n", board_count);

    // Setup inverted index
    int iidx[383] = { 0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546 };
    int* h_ix = (int*)malloc(6561 * sizeof(int));
    memset(h_ix, 0, 6561 * sizeof(int));
    for (int k = 0; k < 383; k++) h_ix[iidx[k]] = k;
    initConstMemory(h_ix);
    free(h_ix);

    // Allocate GPU memory
    gene *pop, *best;
    int *F, *idx_arr;
    float *avg_fit;
    char *d_boards;
    curandState *devStates;

    CUDA_CHECK(cudaMallocManaged(&pop, N * G * sizeof(gene)));
    CUDA_CHECK(cudaMallocManaged(&best, G * sizeof(gene)));
    CUDA_CHECK(cudaMallocManaged(&F, N * TOTAL_CONFIGS * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&avg_fit, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&idx_arr, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_boards, NUM_BOARDS * 36 * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&devStates, N * sizeof(curandState)));

    printf("Memory allocated: pop=%lluMB, F=%lluMB\n",
        (unsigned long long)(N * G * sizeof(gene)) / (1024*1024),
        (unsigned long long)(N * TOTAL_CONFIGS * sizeof(int)) / (1024*1024));

    // Copy boards to GPU
    memcpy(d_boards, h_boards, NUM_BOARDS * 36 * sizeof(char));
    free(h_boards);

    // File setup
    char fname[64], bname[64];
    sprintf(fname, "txt/r-all-%d-%d-%d.txt", N, S, L);
    sprintf(bname, "txt/b-all-%d-%d-%d.txt", N, S, L);
    FILE* results = fopen(fname, "w");
    FILE* bestf = fopen(bname, "w");
    if (!results || !bestf) {
        printf("Error: Cannot create output files. Make sure 'txt' directory exists.\n");
        return -1;
    }

    // Initialize
    int block_size = 256;
    int setup_blocks = (N + block_size - 1) / block_size;
    setup_states<<<setup_blocks, block_size>>>(devStates, time(0), N);
    CUDA_CHECK_KERNEL();
    init_population<<<setup_blocks, block_size>>>(pop, G, S, devStates, N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Initialization complete\n");

    float best_ind_fitness = 0.0f;
    float best_gen_fitness = 0.0f;
    int best_idx = 0;
    time_t start = time(0);

    // --- EVOLUTION LOOP ---
    for (int gen = 0; gen < K; gen++) {
        // Run all boards for all individuals
        int total_runs = N * TOTAL_CONFIGS;
        int run_blocks = (total_runs + block_size - 1) / block_size;
        run_all_boards<<<run_blocks, block_size>>>(pop, d_boards, F, G, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Average fitness
        average_fitness<<<setup_blocks, block_size>>>(F, avg_fit, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Find best individual
        float gen_best = 0.0f;
        float gen_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            gen_sum += avg_fit[i];
            if (avg_fit[i] > gen_best) {
                gen_best = avg_fit[i];
                best_idx = i;
            }
        }
        float gen_avg = gen_sum / N;

        if (gen_best > best_ind_fitness) {
            best_ind_fitness = gen_best;
            int copy_blocks = (G + block_size - 1) / block_size;
            copy_best<<<copy_blocks, block_size>>>(pop, best, G, best_idx);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());  // Must sync before CPU accesses managed memory
        }
        if (gen_avg > best_gen_fitness) {
            best_gen_fitness = gen_avg;
        }

        if (gen % 10 == 0) {
            printf("Gen %d: Best=%.4f Avg=%.4f (all %d boards)\n", gen, gen_best, gen_avg, TOTAL_CONFIGS);
            fprintf(results, "%d,%.4f,%.4f\n", gen, gen_best, gen_avg);
            fflush(results);
        }

        // Shuffle and crossover
        shuffle(idx_arr, N);
        int cross_blocks = (N / 4 + block_size - 1) / block_size;
        crossover<<<cross_blocks, block_size>>>(idx_arr, avg_fit, pop, G, S, devStates, best_ind_fitness, best_gen_fitness, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- RESULTS ---
    cudaDeviceSynchronize();
    printf("\n\nFinished!\nBest Individual: %.4f\nBest Gen Mean: %.4f\n", best_ind_fitness, best_gen_fitness);
    printf("Time: %lld seconds\n", (long long)(time(0) - start));

    // Save best
    for (int i = 0; i < G; i++) {
        fprintf(bestf, "%d %d ", best[i].action, best[i].next_state);
    }

    fclose(results);
    fclose(bestf);

    cudaFree(pop); cudaFree(best); cudaFree(F);
    cudaFree(avg_fit); cudaFree(idx_arr); cudaFree(d_boards);
    cudaFree(devStates);

    return 0;
}
