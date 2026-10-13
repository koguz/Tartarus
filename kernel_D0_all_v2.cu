// kernel_D0_all_v2.cu
//
// D0 training on all 74,760 Tartarus board configurations (1869 layouts x 10
// start cells x 4 directions). Baseline for kernel_D2_all_v2.cu: same agent,
// games, selection and mutation, but crossover ignores gene scores.
//
// Algorithm
//   - FSM agent: G = S * 383 + 1 genes, gene = {action, next_state}.
//     Gene G-1 only supplies the initial state.
//   - Each generation every individual plays all 74,760 games (80 moves each).
//     Individual fitness = average board score.
//   - Population is shuffled (Fisher-Yates) into groups of 4. The best two are
//     parents and survive; the worst two are overwritten by children.
//     Per gene: a coin flip (p = 0.5) picks which parent's gene goes to child 1,
//     the other parent's gene goes to child 2.
//   - Adaptive mutation (k = 10), f = fitness of the child's parent
//     (parent 1 -> child 1, parent 2 -> child 2):
//         pm = k * (f_m - f) / (f_m - f_bar) / G   if f >= f_bar
//         pm = 0.5 * k / G                          if f <  f_bar
//     f_m = best individual fitness so far, f_bar = best generation mean so far.
//     Mutated genes get action 40/30/30 (forward/left/right) and a uniform state.
//
// Output (same format as kernel_D2_all_v2.cu)
//   txt/r-D0-N-S-K-L.txt   "generation,best,average" every 10 generations
//   txt/b-D0-N-S-K-L.txt   best chromosome as "action next_state " pairs
//
// Usage
//   TartarusD0v2.exe <N> <S> <K> <L> [seed]    train (N multiple of 4)
//   TartarusD0v2.exe --eval <chromosome> <S>   average score of a saved agent
//
// Build (run from a VS x64 developer prompt; VS 2026 needs -allow-unsupported-compiler with CUDA 12.9)
//   nvcc -O3 -std=c++17 -arch=sm_120 -o TartarusD0v2.exe kernel_D0_all_v2.cu

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

// --- CONSTANTS ---
constexpr int NUM_BOARDS        = 1869;
constexpr int CONFIGS_PER_BOARD = 40;                                // 4 directions x 10 start cells
constexpr int TOTAL_CONFIGS     = NUM_BOARDS * CONFIGS_PER_BOARD;    // 74,760
constexpr int C                 = 383;                               // valid sensor combinations
constexpr int MOVES             = 80;
constexpr float MUTATION_K      = 10.0f;

constexpr int PLAY_THREADS      = 512;    // threads per block in play_games (one block per individual)
constexpr int CROSS_THREADS     = 256;
constexpr int RNG_PER_GENE      = 8;      // max random numbers one crossover thread draws per generation

// Gene packing: action in the top 2 bits, next_state in the low 14 bits.
constexpr int STATE_BITS        = 14;
constexpr uint16_t STATE_MASK   = (1u << STATE_BITS) - 1;
constexpr int MAX_S             = 1 << STATE_BITS;

// Boards are 64-bit masks of an 8x8 grid: the 6x6 board padded with a wall ring.
// Cell (x, y) of the 6x6 board is bit (y + 1) * 8 + (x + 1).
constexpr uint64_t INTERIOR  = 0x007E7E7E7E7E7E00ULL;
constexpr uint64_t WALL      = ~INTERIOR;
constexpr uint64_t EDGE_ROWS = 0x007E000000007E00ULL;  // board rows 0 and 5
constexpr uint64_t EDGE_COLS = 0x0042424242424200ULL;  // board columns 0 and 5

// Directions: 0 = up (y-1), 1 = down (y+1), 2 = right (x+1), 3 = left (x-1)
__constant__ int c_DIR_OFF[4] = { -8, 8, 1, -1 };
// 8-neighbourhood scan order per direction as padded offsets (SCAN_Y * 8 + SCAN_X),
// filled from the SCAN_X / SCAN_Y tables in init_tables().
__constant__ int c_SCAN_OFF[32];
// TURN_LEFT = {3, 2, 0, 1}, TURN_RIGHT = {2, 3, 1, 0}, packed 2 bits per direction
constexpr unsigned TURN_LEFT_LUT  = 0x4B;
constexpr unsigned TURN_RIGHT_LUT = 0x1E;

__host__ __device__ inline uint16_t pack_gene(int action, int state) {
    return (uint16_t)((action << STATE_BITS) | state);
}
__host__ __device__ inline int gene_action(uint16_t g) { return g >> STATE_BITS; }
__host__ __device__ inline int gene_state(uint16_t g)  { return g & STATE_MASK; }

template <typename RNG>
__device__ inline uint16_t random_gene(int S, RNG* rng) {
    unsigned r = curand(rng) % 100;
    int action = (r < 40) ? 0 : (r < 70) ? 1 : 2;
    return pack_gene(action, curand(rng) % S);
}

// --- KERNELS ---

__global__ void init_population(uint16_t* genes, size_t total, int S, unsigned long long seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, i, 0, &rng);
    genes[i] = random_gene(S, &rng);
}

// One block per individual. Each thread plays games threadIdx.x, +PLAY_THREADS, ...
// Only the individual's average board score is recorded (D0 keeps no gene scores).
__global__ void __launch_bounds__(PLAY_THREADS)
play_games(
    const uint16_t* __restrict__ genes,
    float* __restrict__ avg_fit,
    const uint64_t* __restrict__ boards,   // NUM_BOARDS padded box masks
    const uint8_t* __restrict__ starts,    // NUM_BOARDS * 10 padded start cells
    const uint16_t* __restrict__ ix,       // 6561 -> 0..382
    int G
) {
    __shared__ int s_scan[32];
    __shared__ int s_dir_off[4];
    __shared__ int s_warp_sum[PLAY_THREADS / 32];

    const int ind = blockIdx.x;
    const uint16_t* my_genes = genes + (size_t)ind * G;

    if (threadIdx.x < 32) s_scan[threadIdx.x] = c_SCAN_OFF[threadIdx.x];
    if (threadIdx.x < 4) s_dir_off[threadIdx.x] = c_DIR_OFF[threadIdx.x];
    __syncthreads();

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int initial_state = gene_state(my_genes[G - 1]);
    int f_total = 0;

    for (int cfg = threadIdx.x; cfg < TOTAL_CONFIGS; cfg += PLAY_THREADS) {
        const int board_idx = cfg / CONFIGS_PER_BOARD;
        int dir = (cfg % CONFIGS_PER_BOARD) / 10;
        int pos = starts[board_idx * 10 + cfg % 10];
        uint64_t box = boards[board_idx];
        int cs = initial_state;

        for (int move = 0; move < MOVES; move++) {
            // Sensor: 0 = empty, 1 = box, 2 = wall, base-3 over the 8 neighbours
            int cc = 0;
            int p3 = 1;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                int c = pos + s_scan[dir * 8 + m];
                int val = (int)((box >> c) & 1) + 2 * (int)((WALL >> c) & 1);
                cc += p3 * val;
                p3 *= 3;
            }
            cc = ix[cc];

            const uint16_t g = my_genes[cs * C + cc];
            const int action = gene_action(g);
            cs = gene_state(g);

            if (action == 0) {  // forward
                const int d = s_dir_off[dir];
                const int ahead = pos + d;
                if (!((WALL >> ahead) & 1)) {
                    if (!((box >> ahead) & 1)) {
                        pos = ahead;
                    } else {
                        const int beyond = ahead + d;
                        if (!(((WALL | box) >> beyond) & 1)) {
                            box ^= (1ULL << ahead) | (1ULL << beyond);
                            pos = ahead;
                        }
                    }
                }
            } else if (action == 1) {
                dir = (TURN_LEFT_LUT >> (2 * dir)) & 3;
            } else {
                dir = (TURN_RIGHT_LUT >> (2 * dir)) & 3;
            }
        }

        // Edge rows and edge columns each score 1, so corners score 2
        f_total += __popcll(box & EDGE_ROWS) + __popcll(box & EDGE_COLS);
    }

    // Block sum of all board scores -> average fitness of this individual
    const int warp_sum = cg::reduce(warp, f_total, cg::plus<int>());
    if (warp.thread_rank() == 0) s_warp_sum[warp.meta_group_rank()] = warp_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int w = 0; w < PLAY_THREADS / 32; w++) sum += s_warp_sum[w];
        avg_fit[ind] = (float)sum / (float)TOTAL_CONFIGS;
    }
}

struct GroupInfo {
    int p1, p2;       // parents (best two of the group), survive unchanged
    int c1, c2;       // children (worst two), overwritten
    float pm1, pm2;   // mutation probability for child 1 and child 2
};

// One thread per (group, gene): blockIdx.y = group, x covers the genes.
// Uniform crossover: a coin flip sends one parent's gene to child 1 and the
// other parent's gene to child 2.
__global__ void crossover(
    uint16_t* __restrict__ genes,
    const GroupInfo* __restrict__ groups,
    int G, int S,
    unsigned long long seed, int gen
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= G) return;
    const int group = blockIdx.y;
    const GroupInfo gr = groups[group];

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)group * G + i, (unsigned long long)gen * RNG_PER_GENE, &rng);

    const uint16_t g1 = genes[(size_t)gr.p1 * G + i];
    const uint16_t g2 = genes[(size_t)gr.p2 * G + i];
    const bool p1_first = curand(&rng) & 1;

    const uint16_t to_c1 = p1_first ? g1 : g2;
    const uint16_t to_c2 = p1_first ? g2 : g1;

    genes[(size_t)gr.c1 * G + i] = (curand_uniform(&rng) < gr.pm1) ? random_gene(S, &rng) : to_c1;
    genes[(size_t)gr.c2 * G + i] = (curand_uniform(&rng) < gr.pm2) ? random_gene(S, &rng) : to_c2;
}

// --- HOST ---

static float mutation_rate(float f, float f_m, float f_bar, int G) {
    if (f < f_bar) return 0.5f * MUTATION_K / G;
    float denom = f_m - f_bar;
    if (denom <= 0.0f) return 0.0f;   // whole population identical: f == f_m
    return MUTATION_K * (f_m - f) / denom / G;
}

static void load_boards(std::vector<uint64_t>& boards, std::vector<uint8_t>& starts) {
    FILE* bf = fopen("realboard.txt", "r");
    if (!bf) { printf("Error: Cannot open realboard.txt\n"); exit(1); }

    boards.assign(NUM_BOARDS, 0);
    starts.assign(NUM_BOARDS * 10, 0);
    char line[32];
    int count = 0;
    while (count < NUM_BOARDS && fscanf(bf, "%31s", line) == 1) {
        if (strlen(line) != 16) { printf("Error: bad board line %d: %s\n", count + 1, line); exit(1); }
        uint64_t box = 0;
        int empty = 0;
        for (int j = 0; j < 16; j++) {
            // inner 4x4 cell (row j/4, col j%4) is board cell (x, y) = (j%4 + 1, j/4 + 1)
            int bit = (j / 4 + 2) * 8 + (j % 4 + 2);
            if (line[j] == '1') {
                box |= 1ULL << bit;
            } else {
                if (empty < 10) starts[count * 10 + empty] = (uint8_t)bit;
                empty++;
            }
        }
        if (empty != 10) { printf("Error: board %d has %d empty cells (expected 10)\n", count + 1, empty); exit(1); }
        boards[count++] = box;
    }
    fclose(bf);
    if (count != NUM_BOARDS) { printf("Error: loaded %d boards, expected %d\n", count, NUM_BOARDS); exit(1); }
}

static void init_tables(uint16_t** d_ix) {
    static const int iidx[C] = { 0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546 };
    std::vector<uint16_t> h_ix(6561, 0);
    for (int k = 0; k < C; k++) h_ix[iidx[k]] = (uint16_t)k;
    CUDA_CHECK(cudaMalloc(d_ix, 6561 * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(*d_ix, h_ix.data(), 6561 * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // Scan offsets from the SCAN_X / SCAN_Y tables of kernel_D2_allboards.cu
    const int SCAN_X[4][8] = {
        { 0,  1,  1,  1,  0, -1, -1, -1},
        { 0, -1, -1, -1,  0,  1,  1,  1},
        { 1,  1,  0, -1, -1, -1,  0,  1},
        {-1, -1,  0,  1,  1,  1,  0, -1}
    };
    const int SCAN_Y[4][8] = {
        {-1, -1,  0,  1,  1,  1,  0, -1},
        { 1,  1,  0, -1, -1, -1,  0,  1},
        { 0,  1,  1,  1,  0, -1, -1, -1},
        { 0, -1, -1, -1,  0,  1,  1,  1}
    };
    int scan_off[32];
    for (int d = 0; d < 4; d++)
        for (int m = 0; m < 8; m++)
            scan_off[d * 8 + m] = SCAN_Y[d][m] * 8 + SCAN_X[d][m];
    CUDA_CHECK(cudaMemcpyToSymbol(c_SCAN_OFF, scan_off, sizeof(scan_off)));
}

static void run_games(int N, const uint16_t* genes, float* avg_fit,
                      const uint64_t* boards, const uint8_t* starts, const uint16_t* ix, int G) {
    play_games<<<N, PLAY_THREADS>>>(genes, avg_fit, boards, starts, ix, G);
    CUDA_CHECK_KERNEL();
}

static int eval_mode(const char* path, int S) {
    if (S <= 0 || S > MAX_S) { printf("Error: S must be in 1..%d\n", MAX_S); return -1; }
    const int G = S * C + 1;
    FILE* in = fopen(path, "r");
    if (!in) { printf("Error: Cannot open %s\n", path); return -1; }
    std::vector<uint16_t> h_genes(G);
    int n = 0, a, s;
    while (n < G && fscanf(in, "%d %d", &a, &s) == 2) {
        if (a < 0 || a > 2 || s < 0 || s >= S) { printf("Error: invalid gene %d: %d %d\n", n, a, s); return -1; }
        h_genes[n++] = pack_gene(a, s);
    }
    fclose(in);
    if (n != G) { printf("Error: read %d genes, expected %d for S=%d\n", n, G, S); return -1; }

    std::vector<uint64_t> h_boards;
    std::vector<uint8_t> h_starts;
    load_boards(h_boards, h_starts);
    uint16_t* ix;
    init_tables(&ix);

    uint16_t* genes; float* avg_fit; uint64_t* boards; uint8_t* starts;
    CUDA_CHECK(cudaMalloc(&genes, G * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&avg_fit, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&boards, NUM_BOARDS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&starts, NUM_BOARDS * 10));
    CUDA_CHECK(cudaMemcpy(genes, h_genes.data(), G * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(boards, h_boards.data(), NUM_BOARDS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(starts, h_starts.data(), NUM_BOARDS * 10, cudaMemcpyHostToDevice));

    run_games(1, genes, avg_fit, boards, starts, ix, G);
    float h_avg;
    CUDA_CHECK(cudaMemcpy(&h_avg, avg_fit, sizeof(float), cudaMemcpyDeviceToHost));
    printf("%s (S=%d): average %.4f over %d boards\n", path, S, h_avg, TOTAL_CONFIGS);

    cudaFree(genes); cudaFree(avg_fit); cudaFree(boards); cudaFree(starts); cudaFree(ix);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && strcmp(argv[1], "--eval") == 0) return eval_mode(argv[2], atoi(argv[3]));

    const int N = (argc > 1) ? atoi(argv[1]) : 64;     // population size
    const int S = (argc > 2) ? atoi(argv[2]) : 16;     // states
    const int K = (argc > 3) ? atoi(argv[3]) : 500;    // generations
    const int L = (argc > 4) ? atoi(argv[4]) : 1;      // run ID
    const unsigned long long seed = (argc > 5) ? strtoull(argv[5], nullptr, 10) : (unsigned long long)time(0);

    if (N <= 0 || N % 4 != 0) { printf("Error: N must be a positive multiple of 4\n"); return -1; }
    if (S <= 0 || S > MAX_S) { printf("Error: S must be in 1..%d\n", MAX_S); return -1; }
    const int G = S * C + 1;
    const int groups = N / 4;

    printf("=== D0 All-Boards Training (v2) ===\n");
    printf("N=%d, S=%d, K=%d generations, G=%d genes, seed=%llu\n", N, S, K, G, seed);
    printf("Testing on ALL %d board configurations per individual\n", TOTAL_CONFIGS);
    printf("Using D0 crossover (uniform, p=0.5 per gene, complementary children)\n");

    std::vector<uint64_t> h_boards;
    std::vector<uint8_t> h_starts;
    load_boards(h_boards, h_starts);
    printf("Loaded %d board layouts\n", NUM_BOARDS);
    uint16_t* ix;
    init_tables(&ix);

    // Device memory
    const size_t n_genes = (size_t)N * G;
    uint16_t* genes; float* avg_fit;
    uint64_t* boards; uint8_t* starts; GroupInfo* d_groups;
    CUDA_CHECK(cudaMalloc(&genes, n_genes * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&avg_fit, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&boards, NUM_BOARDS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&starts, NUM_BOARDS * 10));
    CUDA_CHECK(cudaMalloc(&d_groups, groups * sizeof(GroupInfo)));
    CUDA_CHECK(cudaMemcpy(boards, h_boards.data(), NUM_BOARDS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(starts, h_starts.data(), NUM_BOARDS * 10, cudaMemcpyHostToDevice));
    printf("Memory allocated: genes=%lluMB\n",
        (unsigned long long)(n_genes * sizeof(uint16_t)) >> 20);

    // Output files
    char fname[64], bname[64];
    sprintf(fname, "txt/r-D0-%d-%d-%d-%d.txt", N, S, K, L);
    sprintf(bname, "txt/b-D0-%d-%d-%d-%d.txt", N, S, K, L);
    FILE* results = fopen(fname, "w");
    FILE* bestf = fopen(bname, "w");
    if (!results || !bestf) {
        printf("Error: Cannot create output files. Make sure 'txt' directory exists.\n");
        return -1;
    }
    fprintf(results, "generation,best,average\n");

    // Initialize (separate Philox key from crossover so the streams never overlap)
    {
        const int threads = 256;
        const unsigned blocks = (unsigned)((n_genes + threads - 1) / threads);
        init_population<<<blocks, threads>>>(genes, n_genes, S, seed ^ 0x9E3779B97F4A7C15ULL);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Initialization complete\n");

    std::mt19937 host_rng((unsigned)seed);
    std::vector<float> h_fit(N);
    std::vector<int> idx(N);
    std::vector<GroupInfo> h_groups(groups);
    std::vector<uint16_t> best(G, 0);

    float best_ind_fitness = 0.0f;   // f_m
    float best_gen_fitness = 0.0f;   // f_bar
    const dim3 cross_grid((G + CROSS_THREADS - 1) / CROSS_THREADS, groups);
    const auto start = std::chrono::steady_clock::now();
    auto last_report = start;

    // --- EVOLUTION LOOP ---
    for (int gen = 0; gen < K; gen++) {
        run_games(N, genes, avg_fit, boards, starts, ix, G);
        CUDA_CHECK(cudaMemcpy(h_fit.data(), avg_fit, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Find best individual
        float gen_best = 0.0f;
        double gen_sum = 0.0;
        int best_idx = 0;
        for (int i = 0; i < N; i++) {
            gen_sum += h_fit[i];
            if (h_fit[i] > gen_best) { gen_best = h_fit[i]; best_idx = i; }
        }
        const float gen_avg = (float)(gen_sum / N);

        if (gen_best > best_ind_fitness) {
            best_ind_fitness = gen_best;
            CUDA_CHECK(cudaMemcpy(best.data(), genes + (size_t)best_idx * G, G * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        }
        if (gen_avg > best_gen_fitness) best_gen_fitness = gen_avg;

        if (gen % 10 == 0 || gen == K - 1) {
            const auto now = std::chrono::steady_clock::now();
            const double secs = std::chrono::duration<double>(now - last_report).count();
            const int gens = (gen == 0) ? 1 : (gen % 10 == 0 ? 10 : gen % 10);
            last_report = now;
            printf("Gen %d: Best=%.4f Avg=%.4f (%.2f s/gen)\n", gen, gen_best, gen_avg, secs / gens);
            fprintf(results, "%d,%.4f,%.4f\n", gen, gen_best, gen_avg);
            fflush(results);
        }

        // Fisher-Yates shuffle into groups of 4
        for (int i = 0; i < N; i++) idx[i] = i;
        for (int i = N - 1; i > 0; i--) {
            std::uniform_int_distribution<int> pick(0, i);
            std::swap(idx[i], idx[pick(host_rng)]);
        }

        // Best two of each group are parents, worst two are children
        for (int g = 0; g < groups; g++) {
            int ids[4];
            float fits[4];
            for (int i = 0; i < 4; i++) { ids[i] = idx[g * 4 + i]; fits[i] = h_fit[ids[i]]; }
            for (int i = 0; i < 3; i++) {
                for (int j = i + 1; j < 4; j++) {
                    if (fits[j] > fits[i]) { std::swap(fits[i], fits[j]); std::swap(ids[i], ids[j]); }
                }
            }
            h_groups[g] = { ids[0], ids[1], ids[2], ids[3],
                mutation_rate(fits[0], best_ind_fitness, best_gen_fitness, G),
                mutation_rate(fits[1], best_ind_fitness, best_gen_fitness, G) };
        }
        CUDA_CHECK(cudaMemcpy(d_groups, h_groups.data(), groups * sizeof(GroupInfo), cudaMemcpyHostToDevice));

        crossover<<<cross_grid, CROSS_THREADS>>>(genes, d_groups, G, S, seed, gen);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- RESULTS ---
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("\n\nFinished!\nBest Individual: %.4f\nBest Gen Mean: %.4f\n", best_ind_fitness, best_gen_fitness);
    printf("Time: %lld seconds\n", (long long)total);

    for (int i = 0; i < G; i++) fprintf(bestf, "%d %d ", gene_action(best[i]), gene_state(best[i]));
    fclose(results);
    fclose(bestf);

    cudaFree(genes); cudaFree(avg_fit);
    cudaFree(boards); cudaFree(starts); cudaFree(d_groups); cudaFree(ix);
    return 0;
}
