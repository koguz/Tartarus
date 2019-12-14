#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

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

__device__  void rotate_ccw(pos* c, double d) {
	pos r;
	r.x = (int)nearbyintf(c->x * cosf(CR_CUDART_PI / d) - c->y * sinf(CR_CUDART_PI / d));
	r.y = (int)nearbyintf(c->x * sinf(CR_CUDART_PI / d) + c->y * cosf(CR_CUDART_PI / d));
	c->x = r.x; c->y = r.y;
}

__global__ void run_boards(
	gene* solution,
	char* boards,
	char* scores,
	int* ix,
	int G
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int M = 7426 * 10 * 4;
	if (index > M) return; 
	// divide the index by 40 to find the board number
	int B = index / 40;  // the board
	// divide the remainder by 10 to find the direction
	int D = (index % 40) / 10; // the direction
	// the remainder gives the position... 
	int P = (index % 40) % 10; // the position
	// use inverted index to quickly check the next position.
	const int coord[16] = { 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28 };
	
	// we have to copy the board... 
	char board[36];
	for (int i = 0; i < 36; i++) {
		board[i] = boards[B * 36 + i];
	}
	
	
	pos cd;
	switch (D) {
	case 0:
		cd.x = 0; cd.y = -1;
		break;
	case 1:
		cd.x = 0; cd.y = 1;
		break;
	case 2:
		cd.x = 1; cd.y = 0;
		break;
	default:
		cd.x = -1; cd.y = 0;
		break;
	}

	pos cp;
	int j = 0;
	for (int i = 0; i < 16; i++) {
		if (board[coord[i]] == 0) {
			if (j == P) {
				cp.y = coord[j] / 6;
				cp.x = coord[j] % 6;
			}
			else {
				j++;
			}
		} 
	}
	
	int cs = solution[G - 1].next_state;
	
	for (int i = 0; i < 80; i++) {
		int cc = 0;
		float cct = 0.0f;
		for (int m = 0; m < 8; m++) {  // m is for the 8-neighborhood
			int cx = cp.x + cd.x; int cy = cp.y + cd.y;
			if (cx < 0 || cy < 0 || cx >= 6 || cy >= 6) {
				// then it is a wall (wall = 2)
				// cc += powf(3, m) * 2; // this has rounding errors... 
				cct += powf(3, m) * 2.0f;
			}
			else {
				cct += powf(3, m) * (float)board[cx * 6 + cy];
			}
			rotate_ccw(&cd, 4);
		}
		cc = ix[(int)cct];
		int action = solution[cs * 383 + cc].action;
		cs = solution[cs * 383 + cc].next_state;

		int cx, cy, dx, dy;
		switch (action) {
		case 0:
			cx = cp.x + cd.x;
			cy = cp.y + cd.y;
			if (cx >= 0 && cy >= 0 && cx < 6 && cy < 6) {
				if (board[cx * 6 + cy] == 0) { // nothing in front of us... move...
					cp.x = cx; cp.y = cy;
				}
				else { // there is a box...
					dx = cx + cd.x;
					dy = cy + cd.y;
					if (dx >= 0 && dy >= 0 && dx < 6 && dy < 6 && board[dx * 6 + dy] == 0) {
						board[cx * 6 + cy] = 0;
						board[dx * 6 + dy] = 1;
						cp.x = cx; cp.y = cy;  // update your position
					}
				}
			} // else it's a wall, we cannot do anything
			break;
		case 1:
			rotate_ccw(&cd, 0.66);
			break;
		case 2:
			rotate_ccw(&cd, 2);
			break;
		default:
			break;
		}
	}
	
	// find the fitness
	int f = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			int ii = i * 6 + j;
			if (board[ii] == 1) {
				if (i % 6 == 0 || i % 5 == 0) f++;
				if (j % 6 == 0 || j % 5 == 0) f++;
			}
		}
	}
	scores[index] = f;
	printf(".");
}


int main(int argc, char** argv) {
	const int N = 7426;
	int N1 = 256 * atoi(argv[1]);
	int S = atoi(argv[2]);
	int P = 128 * atoi(argv[3]);
	int T1 = atoi(argv[4]);
	int L = atoi(argv[5]);

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
	
	FILE* test = fopen("boards.txt", "r");
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

	// get both file name and number of sizes as arguments to main... 
	//char iname[100] = argv[1]; // "b-1280-128-8-1-1.txt";
	char iname[100];
	sprintf(iname, "txt/b-%d-%d-%d-%d-%d.txt", N1, P, S, T1, L);
	FILE* ind = fopen(iname, "r");  // argv[1]
	//int S = atoi(argv[2]); // 12;  // atoi(argv[2]);
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

	int iidx[383] = { 0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,242,243,245,246,251,252,254,255,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546 };
	int* ix;
	cudaStatus = cudaMallocManaged(&ix, 6561 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (ix)\n");
		return -1;
	}
	for (int i = 0; i < 6561; i++) ix[i] = 0; // first make them all zero, then
	for (int i = 0; i < 383; i++) ix[iidx[i]] = i;  // create the inverted index


	// we are ready. 
	// for each board, there should be 10 empty spaces, we can start at those spaces
	// each time with a different direction... 

	cudaDeviceSynchronize(); // so that the managed memories are ready.
	// 7426 boards * 4 direction * in 10 spaces
	int block_size = 1024;
	int T = N * 4 * 10; 
	int num_blocks = (T + block_size - 1) / block_size;
	printf("%d -> %d", num_blocks, block_size);
	run_boards<<<num_blocks, block_size>>>(solution, boards, scores, ix, G);
	cudaDeviceSynchronize();

	char rname[100];
	sprintf(rname, "results_b-%d-%d-%d-%d-%d.txt", N1, P, S, T1, L);
	//sprintf(rname, "results_%s", argv[1]);
	
	FILE* results = fopen(rname, "w");
	int s = 0;
	for (int i = 0; i < N * 40; i++) {
		fprintf(results, "%d ", (int)scores[i]);
		s += scores[i];
	}
	fclose(results);
	printf("average: %.2f", (float)s / (float)(N * 40));

	FILE* arslt = fopen("complete_results.csv", "a");
	fprintf(arslt, "%d,%d,%d,%d,%d,%0.4f\n", N1, P, S, T1, L, (float)s / (float)(N * 40));
	fclose(arslt);

	return 0;
}
