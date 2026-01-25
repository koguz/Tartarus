#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

// --- Constant Memory ---
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

typedef struct gene_struct {
    int next_state;
    int action;
    int m_fitness;
    int fitness;
    int used;
} gene;

// Analysis kernel - collects detailed statistics
__global__ void analyze_solution(
    gene* solution,
    char* boards,
    int* ix,
    int G,
    int S,
    // Output arrays for analysis
    unsigned long long* state_combo_counts,  // S * 383 - heatmap
    unsigned long long* state_transitions,   // S * S - transition matrix
    unsigned long long* push_counts,         // S * 383 - successful pushes per state-combo
    unsigned long long* state_action_counts, // S * 3 - actions per state (aggregate)
    unsigned long long* combo_state_changes  // 383 - how often each combo triggers state change
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int M = 1869 * 10 * 4;  // 74,760 total configurations
    if (index >= M) return;

    int B = index / 40;
    int D = (index % 40) / 10;
    int P = (index % 40) % 10;

    const int coord[16] = { 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28 };

    char board[36];
    for (int i = 0; i < 36; i++) {
        board[i] = boards[B * 36 + i];
    }

    int dir = D;
    int cpx = 0, cpy = 0;
    int j = 0;
    for (int i = 0; i < 16; i++) {
        if (board[coord[i]] == 0) {
            if (j == P) {
                cpy = coord[i] / 6;
                cpx = coord[i] % 6;
                break;
            }
            j++;
        }
    }

    int cs = solution[G - 1].next_state;

    for (int i = 0; i < 80; i++) {
        // Compute sensor input
        int cc = 0;
        for (int m = 0; m < 8; m++) {
            int sx = cpx + SCAN_X[dir][m];
            int sy = cpy + SCAN_Y[dir][m];
            int val;
            if (sx < 0 || sy < 0 || sx >= 6 || sy >= 6) {
                val = 2;
            } else {
                // val = board[sx * 6 + sy];
                val = board[sy * 6 + sx];
            }
            cc += POW3[m] * val;
        }
        cc = ix[cc];

        // Record state-combination occurrence
        atomicAdd(&state_combo_counts[cs * 383 + cc], 1ULL);

        int action = solution[cs * 383 + cc].action;
        int next_state = solution[cs * 383 + cc].next_state;

        // Record state transition
        atomicAdd(&state_transitions[cs * S + next_state], 1ULL);

        // Record action per state
        atomicAdd(&state_action_counts[cs * 3 + action], 1ULL);

        // Check if state changed
        if (cs != next_state) {
            atomicAdd(&combo_state_changes[cc], 1ULL);
        }

        // Execute action and track pushes
        switch (action) {
        case 0: {  // Move forward
            int cx = cpx + DIR_X[dir];
            int cy = cpy + DIR_Y[dir];
            if (cx >= 0 && cy >= 0 && cx < 6 && cy < 6) {
                if (board[cy * 6 + cx] == 0) {
                    cpx = cx; cpy = cy;
                } else {
                    // There's a box - try to push
                    int dx = cx + DIR_X[dir];
                    int dy = cy + DIR_Y[dir];
                    if (dx >= 0 && dy >= 0 && dx < 6 && dy < 6 && board[dy * 6 + dx] == 0) {
                        board[cy * 6 + cx] = 0;
                        board[dy * 6 + dx] = 1;
                        cpx = cx; cpy = cy;
                        // Successful push!
                        atomicAdd(&push_counts[cs * 383 + cc], 1ULL);
                    }
                }
            }
            break;
        }
        case 1:
            dir = TURN_LEFT[dir];
            break;
        case 2:
            dir = TURN_RIGHT[dir];
            break;
        }

        cs = next_state;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <solution_file> <S>\n", argv[0]);
        return -1;
    }

    const int N = 1869;
    char* filename = argv[1];
    int S = atoi(argv[2]);
    int C = 383;
    int G = S * C + 1;

    printf("Analyzing solution: %s (S=%d)\n", filename, S);
    printf("This will generate detailed behavioral statistics.\n\n");

    // Load boards
    char* boards;
    cudaMallocManaged(&boards, N * 36 * sizeof(char));
    for (int i = 0; i < N * 36; i++) boards[i] = 0;

    const int coord[16] = { 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28 };
    FILE* bf = fopen("realboard.txt", "r");
    char b[17];
    int bi = 0;
    while (fscanf(bf, "%s", b) != EOF) {
        for (int j = 0; j < 16; j++) {
            boards[bi * 36 + coord[j]] = (b[j] == '1') ? 1 : 0;
        }
        bi++;
    }
    fclose(bf);

    // Load solution
    gene* solution;
    cudaMallocManaged(&solution, G * sizeof(gene));
    FILE* sf = fopen(filename, "r");
    if (!sf) {
        printf("Error: Cannot open %s\n", filename);
        return -1;
    }
    int si = 0;
    while (fscanf(sf, "%d %d", &solution[si].action, &solution[si].next_state) == 2) {
        si++;
    }
    fclose(sf);
    printf("Loaded %d genes\n", si);

    // Setup inverted index
    int iidx[383] = { 0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546 };
    int* ix;
    cudaMallocManaged(&ix, 6561 * sizeof(int));
    for (int i = 0; i < 6561; i++) ix[i] = 0;
    for (int i = 0; i < 383; i++) ix[iidx[i]] = i;

    // Allocate analysis arrays
    unsigned long long *state_combo_counts, *state_transitions, *push_counts;
    unsigned long long *state_action_counts, *combo_state_changes;

    cudaMallocManaged(&state_combo_counts, S * 383 * sizeof(unsigned long long));
    cudaMallocManaged(&state_transitions, S * S * sizeof(unsigned long long));
    cudaMallocManaged(&push_counts, S * 383 * sizeof(unsigned long long));
    cudaMallocManaged(&state_action_counts, S * 3 * sizeof(unsigned long long));
    cudaMallocManaged(&combo_state_changes, 383 * sizeof(unsigned long long));

    for (int i = 0; i < S * 383; i++) state_combo_counts[i] = 0;
    for (int i = 0; i < S * S; i++) state_transitions[i] = 0;
    for (int i = 0; i < S * 383; i++) push_counts[i] = 0;
    for (int i = 0; i < S * 3; i++) state_action_counts[i] = 0;
    for (int i = 0; i < 383; i++) combo_state_changes[i] = 0;

    cudaDeviceSynchronize();

    // Run analysis
    int block_size = 1024;
    int T = N * 4 * 10;
    int num_blocks = (T + block_size - 1) / block_size;
    printf("Running analysis on %d configurations...\n", T);

    analyze_solution<<<num_blocks, block_size>>>(
        solution, boards, ix, G, S,
        state_combo_counts, state_transitions, push_counts,
        state_action_counts, combo_state_changes
    );
    cudaDeviceSynchronize();

    // === OUTPUT RESULTS ===

    // 1. State-Combination Heatmap (CSV)
    char fname1[256];
    sprintf(fname1, "analysis_heatmap_%s.csv", filename);
    for (char* p = fname1; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
    FILE* f1 = fopen(fname1, "w");
    fprintf(f1, "state,combo,count\n");
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < 383; c++) {
            if (state_combo_counts[s * 383 + c] > 0) {
                fprintf(f1, "%d,%d,%llu\n", s, c, state_combo_counts[s * 383 + c]);
            }
        }
    }
    fclose(f1);
    printf("Saved: %s\n", fname1);

    // 2. State Transition Matrix (CSV)
    char fname2[256];
    sprintf(fname2, "analysis_transitions_%s.csv", filename);
    for (char* p = fname2; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
    FILE* f2 = fopen(fname2, "w");
    fprintf(f2, "from_state,to_state,count\n");
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            if (state_transitions[i * S + j] > 0) {
                fprintf(f2, "%d,%d,%llu\n", i, j, state_transitions[i * S + j]);
            }
        }
    }
    fclose(f2);
    printf("Saved: %s\n", fname2);

    // 3. Push counts per state-combo (CSV)
    char fname3[256];
    sprintf(fname3, "analysis_pushes_%s.csv", filename);
    for (char* p = fname3; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
    FILE* f3 = fopen(fname3, "w");
    fprintf(f3, "state,combo,push_count,total_count,push_rate\n");
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < 383; c++) {
            unsigned long long total = state_combo_counts[s * 383 + c];
            unsigned long long pushes = push_counts[s * 383 + c];
            if (total > 0) {
                fprintf(f3, "%d,%d,%llu,%llu,%.4f\n", s, c, pushes, total,
                    (float)pushes / (float)total);
            }
        }
    }
    fclose(f3);
    printf("Saved: %s\n", fname3);

    // 4. State action summary
    char fname4[256];
    sprintf(fname4, "analysis_state_actions_%s.csv", filename);
    for (char* p = fname4; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
    FILE* f4 = fopen(fname4, "w");
    fprintf(f4, "state,forward,turn_left,turn_right,total,forward_pct,left_pct,right_pct\n");
    for (int s = 0; s < S; s++) {
        unsigned long long fwd = state_action_counts[s * 3 + 0];
        unsigned long long left = state_action_counts[s * 3 + 1];
        unsigned long long right = state_action_counts[s * 3 + 2];
        unsigned long long total = fwd + left + right;
        if (total > 0) {
            fprintf(f4, "%d,%llu,%llu,%llu,%llu,%.2f,%.2f,%.2f\n", s, fwd, left, right, total,
                100.0f * fwd / total, 100.0f * left / total, 100.0f * right / total);
        }
    }
    fclose(f4);
    printf("Saved: %s\n", fname4);

    // 5. Combo state-change triggers
    char fname5[256];
    sprintf(fname5, "analysis_combo_triggers_%s.csv", filename);
    for (char* p = fname5; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
    FILE* f5 = fopen(fname5, "w");
    fprintf(f5, "combo,raw_index,state_changes\n");
    for (int c = 0; c < 383; c++) {
        fprintf(f5, "%d,%d,%llu\n", c, iidx[c], combo_state_changes[c]);
    }
    fclose(f5);
    printf("Saved: %s\n", fname5);

    // 6. Solution gene table (for reference)
    char fname6[256];
    sprintf(fname6, "analysis_genes_%s.csv", filename);
    for (char* p = fname6; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
    FILE* f6 = fopen(fname6, "w");
    fprintf(f6, "state,combo,action,next_state,same_state\n");
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < 383; c++) {
            int idx = s * 383 + c;
            fprintf(f6, "%d,%d,%d,%d,%d\n", s, c,
                solution[idx].action, solution[idx].next_state,
                (s == solution[idx].next_state) ? 1 : 0);
        }
    }
    fclose(f6);
    printf("Saved: %s\n", fname6);

    // === CONSOLE SUMMARY ===
    printf("\n=== ANALYSIS SUMMARY ===\n");

    // State usage summary
    printf("\nState Usage:\n");
    unsigned long long total_moves = (unsigned long long)N * 40 * 80;
    for (int s = 0; s < S; s++) {
        unsigned long long count = 0;
        for (int c = 0; c < 383; c++) count += state_combo_counts[s * 383 + c];
        printf("  State %2d: %10llu (%5.2f%%) | Fwd:%5.1f%% L:%5.1f%% R:%5.1f%%\n",
            s, count, 100.0f * count / total_moves,
            100.0f * state_action_counts[s*3+0] / (count > 0 ? count : 1),
            100.0f * state_action_counts[s*3+1] / (count > 0 ? count : 1),
            100.0f * state_action_counts[s*3+2] / (count > 0 ? count : 1));
    }

    // Push statistics
    unsigned long long total_pushes = 0;
    for (int i = 0; i < S * 383; i++) total_pushes += push_counts[i];
    printf("\nTotal successful pushes: %llu (%.2f per run)\n",
        total_pushes, (float)total_pushes / (N * 40));

    // Top pushing states
    printf("\nTop pushing states:\n");
    for (int s = 0; s < S; s++) {
        unsigned long long sp = 0;
        for (int c = 0; c < 383; c++) sp += push_counts[s * 383 + c];
        if (sp > total_pushes / S) {  // above average
            printf("  State %2d: %llu pushes (%.1f%%)\n", s, sp, 100.0f * sp / total_pushes);
        }
    }

    // Self-loop vs transition
    unsigned long long self_loops = 0, transitions_total = 0;
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            if (i == j) self_loops += state_transitions[i * S + j];
            transitions_total += state_transitions[i * S + j];
        }
    }
    printf("\nState transitions: %llu total, %llu self-loops (%.1f%%)\n",
        transitions_total, self_loops, 100.0f * self_loops / transitions_total);

    // Cleanup
    cudaFree(boards);
    cudaFree(solution);
    cudaFree(ix);
    cudaFree(state_combo_counts);
    cudaFree(state_transitions);
    cudaFree(push_counts);
    cudaFree(state_action_counts);
    cudaFree(combo_state_changes);

    printf("\nAnalysis complete!\n");
    return 0;
}
