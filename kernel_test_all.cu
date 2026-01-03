#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// Turn lookup tables
__constant__ int TURN_LEFT[4]  = { 3, 2, 0, 1 };  // 0→3, 1→2, 2→0, 3→1
__constant__ int TURN_RIGHT[4] = { 2, 3, 1, 0 };  // 0→2, 1→3, 2→1, 3→0

typedef struct gene_struct {
	int next_state;
	int action;
	int m_fitness;
	int fitness;
	int used;
} gene;

__global__ void run_boards(
	gene* solution,
	char* boards,
	char* scores,
	int* ix,
	int G,
	int S,
	unsigned long long* state_counts
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int M = 1869 * 10 * 4;  // 74,760 total configurations
	if (index >= M) return;

	// Decode index into board, direction, position
	int B = index / 40;           // board index (0-1868)
	int D = (index % 40) / 10;    // direction (0-3)
	int P = (index % 40) % 10;    // position (0-9)

	const int coord[16] = { 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28 };

	// Copy the board to local memory
	char board[36];
	for (int i = 0; i < 36; i++) {
		board[i] = boards[B * 36 + i];
	}

	// Direction as integer index (0-3)
	int dir = D;

	// Find the P-th empty cell for starting position
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

	// Get initial state from last gene
	int cs = solution[G - 1].next_state;

	// Run 80 moves
	for (int i = 0; i < 80; i++) {
		// Count state usage
		atomicAdd(&state_counts[cs], 1ULL);

		// Compute sensor input using lookup tables (no trig!)
		int cc = 0;
		for (int m = 0; m < 8; m++) {
			int sx = cpx + SCAN_X[dir][m];
			int sy = cpy + SCAN_Y[dir][m];
			int val;
			if (sx < 0 || sy < 0 || sx >= 6 || sy >= 6) {
				val = 2;  // wall
			} else {
				val = board[sx * 6 + sy];
			}
			cc += POW3[m] * val;
		}
		cc = ix[cc];  // convert to valid sensor index (0-382)

		int action = solution[cs * 383 + cc].action;
		cs = solution[cs * 383 + cc].next_state;

		// Execute action using lookup tables
		switch (action) {
		case 0: {  // Move forward
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
			break;
		}
		case 1:  // Turn left
			dir = TURN_LEFT[dir];
			break;
		case 2:  // Turn right
			dir = TURN_RIGHT[dir];
			break;
		default:
			break;
		}
	}

	// Calculate fitness (corners get +2, edges get +1)
	int f = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			if (board[i * 6 + j] == 1) {
				if (i == 0 || i == 5) f++;
				if (j == 0 || j == 5) f++;
			}
		}
	}
	scores[index] = f;
}


int main(int argc, char** argv) {
	const int N = 1869;  // unique board layouts from realboard.txt

	// New simple mode: kernel_test_all.exe <filename> <S>
	// Old mode: kernel_test_all.exe <N1> <S> <P> <T1> <L>
	char iname[256];
	int S;

	if (argc == 3) {
		// Simple mode: direct filename
		strcpy(iname, argv[1]);
		S = atoi(argv[2]);
	} else if (argc == 6) {
		// Legacy mode
		int N1 = 256 * atoi(argv[1]);
		S = atoi(argv[2]);
		int P = 128 * atoi(argv[3]);
		int T1 = atoi(argv[4]);
		int L = atoi(argv[5]);
		sprintf(iname, "txt/b-%d-%d-%d-%d-%d.txt", N1, P, S, T1, L);
	} else {
		printf("Usage: %s <filename> <S>\n", argv[0]);
		printf("   or: %s <N1> <S> <P> <T1> <L>\n", argv[0]);
		return -1;
	}

	printf("Testing solution: %s (S=%d)\n", iname, S);

	char* boards;
	//printf("int: %d\n, char: %d\n, bool %d\n", sizeof(int), sizeof(char), sizeof(bool));
	cudaError_t cudaStatus = cudaMallocManaged(&boards, N * 36 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (boards)\n");
		return -1;
	}
	for (int i = 0; i < N * 36; i++) boards[i] = 0;

	char* scores;
	cudaStatus = cudaMallocManaged(&scores, N * 40 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (scores)\n");
		return -1;
	}
	for (int i = 0; i < N * 40; i++) scores[i] = 0;
	const int coord[16] = { 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28 };
	
	FILE* test = fopen("realboard.txt", "r");
	char b[17];  // these are fixed... 
	int i = 0;
	while (fscanf(test, "%s", b) != EOF) {
		for (int j = 0; j < 16; j++) {
			b[j] == '0' ? boards[i * 36 + coord[j]] = 0 : boards[i * 36 + coord[j]] = 1;
			//printf("%d", boards[i * 36 + coord[j]]);
		}
		i++;
	}
	fclose(test);

	FILE* ind = fopen(iname, "r");
	if (!ind) {
		printf("Error: Cannot open %s\n", iname);
		return -1;
	}
	int C = 383;

	gene* solution;
	int G = S * C + 1;
	cudaStatus = cudaMallocManaged(&solution, G * sizeof(gene));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (gene)\n");
		return -1;
	}
	i = 0;
	while (
		fscanf(ind, "%d", &(solution[i].action)) != EOF &&
		fscanf(ind, "%d", &(solution[i].next_state)) != EOF) {
		// printf("%d - %d\n", solution[i].action, solution[i].next_state);
		solution[i].m_fitness = 0;
		solution[i].fitness = 0;
		solution[i].used = 0;
		i++;
	}
	fclose(ind);

	// Fixed iidx array - corrected values at indices 36-43 (was 242,243,245,246,251,252,254,255 -> now 243,244,246,247,252,253,255,256)
	int iidx[383] = { 0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546 };
	int* ix;
	cudaStatus = cudaMallocManaged(&ix, 6561 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (ix)\n");
		return -1;
	}
	for (int i = 0; i < 6561; i++) ix[i] = 0; // first make them all zero, then
	for (int i = 0; i < 383; i++) ix[iidx[i]] = i;  // create the inverted index


	// Allocate state usage counters
	unsigned long long* state_counts;
	cudaStatus = cudaMallocManaged(&state_counts, S * sizeof(unsigned long long));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (state_counts)\n");
		return -1;
	}
	for (int i = 0; i < S; i++) state_counts[i] = 0;

	// we are ready.
	// for each board, there should be 10 empty spaces, we can start at those spaces
	// each time with a different direction...

	cudaDeviceSynchronize(); // so that the managed memories are ready.
	// 1869 boards * 4 directions * 10 positions = 74,760 total configurations
	int block_size = 1024;
	int T = N * 4 * 10;
	int num_blocks = (T + block_size - 1) / block_size;
	printf("%d -> %d\n", num_blocks, block_size);
	run_boards<<<num_blocks, block_size>>>(solution, boards, scores, ix, G, S, state_counts);
	cudaDeviceSynchronize();

	char rname[256];
	sprintf(rname, "results_%s", iname);
	// Replace path separators with underscores
	for (int j = 0; rname[j]; j++) {
		if (rname[j] == '/' || rname[j] == '\\') rname[j] = '_';
	}

	FILE* results = fopen(rname, "w");
	int s = 0;
	int score_histogram[11] = {0};  // scores 0-10
	for (int i = 0; i < N * 40; i++) {
		fprintf(results, "%d ", (int)scores[i]);
		s += scores[i];
		if (scores[i] >= 0 && scores[i] <= 10) {
			score_histogram[(int)scores[i]]++;
		}
	}
	fclose(results);
	float avg_score = (float)s / (float)(N * 40);
	printf("average: %.4f\n", avg_score);

	// Print score histogram
	int total_boards = N * 40;
	printf("\nScore distribution (out of %d boards):\n", total_boards);
	for (int i = 10; i >= 0; i--) {
		float pct = 100.0f * score_histogram[i] / total_boards;
		int bar_len = (int)(pct / 2);  // 50 chars = 100%
		printf("  Score %2d: %6d (%5.2f%%) ", i, score_histogram[i], pct);
		for (int b = 0; b < bar_len; b++) printf("#");
		printf("\n");
	}

	// Print state usage statistics
	unsigned long long total_moves = (unsigned long long)N * 40 * 80;  // 74,760 configs * 80 moves
	printf("\nState usage (total moves: %llu):\n", total_moves);
	for (int i = 0; i < S; i++) {
		float pct = 100.0f * (float)state_counts[i] / (float)total_moves;
		printf("  State %2d: %12llu (%6.2f%%)\n", i, state_counts[i], pct);
	}

	// Save score distribution to file (txt/sc-*)
	char scname[256];
	sprintf(scname, "txt/sc_%s", iname);
	// Replace path separators with underscores (skip "txt/")
	for (char* p = scname + 4; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
	FILE* scfile = fopen(scname, "w");
	if (scfile) {
		fprintf(scfile, "score,count,percentage\n");
		for (int i = 0; i <= 10; i++) {
			fprintf(scfile, "%d,%d,%.4f\n", i, score_histogram[i], 100.0f * score_histogram[i] / total_boards);
		}
		fprintf(scfile, "average,%.4f\n", avg_score);
		fclose(scfile);
		printf("\nSaved: %s\n", scname);
	}

	// Save state distribution to file (txt/st-*)
	char stname[256];
	sprintf(stname, "txt/st_%s", iname);
	// Replace path separators with underscores (skip "txt/")
	for (char* p = stname + 4; *p; p++) if (*p == '/' || *p == '\\') *p = '_';
	FILE* stfile = fopen(stname, "w");
	if (stfile) {
		fprintf(stfile, "state,count,percentage\n");
		for (int i = 0; i < S; i++) {
			float pct = 100.0f * (float)state_counts[i] / (float)total_moves;
			fprintf(stfile, "%d,%llu,%.4f\n", i, state_counts[i], pct);
		}
		fclose(stfile);
		printf("Saved: %s\n", stname);
	}

	// Parse N, K, L from filename (format: txt/b-all-N-S-K-L.txt)
	int parsed_N = 0, parsed_S = 0, parsed_K = 0, parsed_L = 0;
	sscanf(iname, "txt/b-all-%d-%d-%d-%d.txt", &parsed_N, &parsed_S, &parsed_K, &parsed_L);

	// Check if complete_results.csv needs header
	FILE* arslt = fopen("complete_results.csv", "r");
	int needs_header = (arslt == NULL);
	if (arslt) {
		fseek(arslt, 0, SEEK_END);
		if (ftell(arslt) == 0) needs_header = 1;
		fclose(arslt);
	}

	arslt = fopen("complete_results.csv", "a");
	if (needs_header) {
		fprintf(arslt, "filename,N,S,K,L,average\n");
	}
	fprintf(arslt, "%s,%d,%d,%d,%d,%.4f\n", iname, parsed_N, S, parsed_K, parsed_L, avg_score);
	fclose(arslt);

	cudaFree(state_counts);
	return 0;
}
