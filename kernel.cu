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

__global__ void setup_states(curandState* state, int* Q) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(Q[id], 0, 0, &state[id]);
}

__global__ void init_population(gene* pop, int G, int S, curandState* state) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = state[id];
	int s = id * G;
	for (int i = s; i < s + G; i++) {
		int myrand = curand(&localState) % 100;
		if (myrand < 40) pop[i].action = 0;
		else if (myrand < 70) pop[i].action = 1;
		else pop[i].action = 2;
		//pop[i].action = curand(&localState) % 3;
		pop[i].next_state = curand(&localState) % S;
		pop[i].fitness = 0;
		pop[i].m_fitness = 0;
		pop[i].used = 0;
	}
	state[id] = localState;
}

__global__ void generate_boards(int* boards, curandState* state, int R) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = state[id];

	int s = id * R * R;
	while (1) {
		for (int i = s; i < s + (R * R); i++) boards[i] = 0; // first let all be 0
		int i = 0;
		do {
			int x = (curand(&localState) % (R - 2)) + 1;
			int y = (curand(&localState) % (R - 2)) + 1;
			if (boards[s + (x * R + y)] == 0) {
				boards[s + (x * R + y)] = 1;
				i++;
			}
		} while (i < 6);  // magic number 6 -> number of boxes... 

		int repeat = 0;
		for (int i = s; i < s + (R * R); i++) {
			if (
				boards[i] == 1 &&
				boards[i + 1] == 1 &&
				boards[i + R] == 1 &&
				boards[i + R + 1] == 1
				) {
				repeat = 1;
				break;
			}
		}
		if (repeat == 0) break;
	}
	state[id] = localState;
}

__global__ void average_fitness(int P, int* F, float* avg_fit, int G) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// we get the current thread with id - it will be as many as N
	// F is N * P, so we skip ahead P 
	int s = id * P;
	int sum = 0;
	for (int i = s; i < s + P; i++) {
		sum += F[i];
	}
	avg_fit[id] = (float)sum / (float)P;
}

__global__ void run_boards(
	gene* pop,
	int* boards,
	curandState* state,
	int R,
	int G,
	int M,
	int C,
	int* F,
	int* ix,
	int* sum_occ,
	int* statistics) { //,
	//int* nstats) {
	// blockIdx.x is the individual out of N individuals
	// threadIdx.x is the board out of P boards for that individual... 
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = state[id]; // used to be blockIdx.x

	// get the individual
	int ind = blockIdx.x * G;
	// get the board... 
	int brd = id * R * R;

	// find a random position on board
	pos cp; cp.x = -1; cp.y = -1;
	while (1) {
		cp.x = (curand(&localState) % (R - 2)) + 1;
		cp.y = (curand(&localState) % (R - 2)) + 1;
		if (boards[brd + (cp.x * R + cp.y)] == 0)
			break;
	}
	// get a random direction
	pos cd;
	switch (curand(&localState) % 4) {
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

	// Have to use this
	const int X = 383 * 16 + 1;
	bool used[X];
	for (int i = 0; i < X; i++) used[i] = false;

	int cs = pop[ind + G - 1].next_state;
	for (int i = 0; i < M; i++) {  // perform M moves (default 80)
		int cc = 0;
		//float cct = 0.0f;
		int cct2 = 0;
		for (int m = 0; m < 8; m++) {  // m is for the 8-neighborhood
			int cx = cp.x + cd.x; int cy = cp.y + cd.y;
			if (cx < 0 || cy < 0 || cx >= R || cy >= R) {
				// then it is a wall (wall = 2)
				// cc += powf(3, m) * 2; // this has rounding errors... 
				int tt = 1;
				for (int j = 0; j < m; j++) tt = tt * 3;
				cct2 += tt * 2;
				//cct += powf(3, m) * 2.0f;
			}
			else {
				int tt = 1;
				for (int j = 0; j < m; j++) tt = tt * 3;
				cct2 += tt * boards[brd + (cx * R + cy)];
				//cct += powf(3, m) * (float)boards[brd + (cx * R + cy)];
			}
			rotate_ccw(&cd, 4);
		}
		//if ((int)cct != cct2) {
			//printf("%d != %d\n", (int)cct, cct2);
		//}
		//cc = ix[(int)cct]; 
		cc = ix[cct2];
		int action = pop[ind + (cs * C + cc)].action;
		//if (pop[ind + (cs * C + cc)].used == 0) { 
			// should occur very infrequently, therefore we should be safe.
			// update! now it will occur frequently
			// atomicAdd(&(pop[ind + (cs * C + cc)]).used, 1); 
			// update! now it will not run at all... 
		//}
		cs = pop[ind + (cs * C + cc)].next_state;
		statistics[blockIdx.x * C + cc]++;
		used[cs * C + cc] = true;
		// occ[id * G + cc]++;
		// atomic
		atomicAdd(&sum_occ[blockIdx.x * G + cc], 1);
		//atomicAdd(&nstats[cct2], 1);

		int cx, cy, dx, dy;
		switch (action) {
		case 0:
			cx = cp.x + cd.x;
			cy = cp.y + cd.y;
			if (cx >= 0 && cy >= 0 && cx < R && cy < R) {
				if (boards[brd + (cx * R + cy)] == 0) { // nothing in front of us... move...
					cp.x = cx; cp.y = cy;
				}
				else { // there is a box...
					dx = cx + cd.x;
					dy = cy + cd.y;
					if (dx >= 0 && dy >= 0 && dx < R && dy < R && boards[brd + (dx * R + dy)] == 0) {
						boards[brd + (cx * R + cy)] = 0;
						boards[brd + (dx * R + dy)] = 1;
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
	}  // end of moves loop

	// find the fitness
	int f = 0;
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < R; j++) {
			int ii = brd + (i * R + j);
			if (boards[ii] == 1) {
				if (i % R == 0 || i % (R - 1) == 0) f++;
				if (j % R == 0 || j % (R - 1) == 0) f++;
			}
		}
	}
	F[id] = f;
	// store this for each gene struct
	for (int i = 0; i < G; i++) {
		if (used[i]) {
			atomicAdd(&(pop[ind + i]).fitness, f);
			atomicAdd(&(pop[ind + i]).used, 1);
		}
		while (true) {
			int tm = pop[ind + i].m_fitness;
			if (f > tm) {
				atomicCAS(&(pop[ind + i]).m_fitness, tm, f);
			}
			else {
				break;
			}
		}
	}
	state[id] = localState;
}

__global__ void crossover(
	int* idx,
	float* f,
	gene* pop,
	int G,
	int S,
	curandState* state,
	float ftmax,
	float ftbar,
	int* sum_occ,
	int option
) {
	// id will be N/4, so we multiply by 4
	int id = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
	curandState localState = state[id];

	float max1 = -1.0f, max2 = -1.0f;
	int idx1 = -1, idx2 = -1;
	for (int i = id; i < id + 4; i++) {
		if (f[idx[i]] > max1&& f[idx[i]] > max2) {
			max2 = max1;
			idx2 = idx1;
			max1 = f[idx[i]];
			idx1 = idx[i];
		}
		else if (f[idx[i]] > max2) {
			max2 = f[idx[i]];
			idx2 = idx[i];
		}
	}
	int t = 0, min1 = -1, min2 = -1;
	for (int i = id; i < id + 4; i++) {
		if (idx[i] != idx1 && idx[i] != idx2) {
			if (t == 0) {
				min1 = idx[i];
				t = 1;
			}
			else {
				min2 = idx[i];
			}
		}
	}
	// idx1 is the maximum, idx2 is the second maximum
	// uniform crossover using idxes and overwrite min1 and min2
	// printf("%d - %d - %d - %d\n\n", idx1, idx2, min1, min2);
	/*if (idx1 == -1 || idx2 == -1 || min1 == -1 || min2 == -1) {
		printf("%d=4 x (%d x %d + %d)\n\n", id, blockIdx.x, blockDim.x, threadIdx.x);
		printf("%d - %d - %d - %d - %d - %d \n\n", idx1, idx2, min1, min2, max1, max2);
		printf("%d - %d\n\n", idx[id], f[idx[id]]);
		for (int i = 0; i < 10; i++) printf("%d ", f[idx[i]]); 
	}*/
	int k = 10;
	float pm1 = k * 0.5 / G; float pm2 = k * 0.5 / G;
	if (f[idx1] >= ftbar) pm1 = k * (ftmax - f[idx1]) / (ftmax - ftbar) / G;
	if (f[idx2] >= ftbar) pm2 = k * (ftmax - f[idx2]) / (ftmax - ftbar) / G;
	for (int i = 0; i < G; i++) {
		float v1 = 0; float v2 = 0;
		if (option == 1) {
			v1 = sum_occ[idx1 * G + i];
			v2 = sum_occ[idx2 * G + i];
		}
		else if (option == 2) {
			if (pop[idx1 * G + i].used != 0)
				v1 = pop[idx1 * G + i].m_fitness;
			// v1 = (float)pop[idx1 * G + i].fitness / (float)pop[idx1 * G + i].used;
			if (pop[idx2 * G + i].used != 0)
				v2 = pop[idx2 * G + i].m_fitness;
			// v2 = (float)pop[idx2 * G + i].fitness / (float)pop[idx2 * G + i].used;
		/*if (v1 < 7 && v2 < 7) {
			v1 = 0; v2 = 0;
		}*/
		}
		// if (curand(&localState) % 2 == 0) {
		if (v1 > v2 || (v1 == v2 && curand(&localState) % 2 == 0)) {
			if (curand_uniform(&localState) <= pm1) {
				pop[min1 * G + i].action = curand(&localState) % 3;
				pop[min1 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min1 * G + i].action = pop[idx1 * G + i].action;
				pop[min1 * G + i].next_state = pop[idx1 * G + i].next_state;
			}
			if (curand_uniform(&localState) <= pm2) {
				pop[min2 * G + i].action = curand(&localState) % 3;
				pop[min2 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min2 * G + i].action = pop[idx2 * G + i].action;
				pop[min2 * G + i].next_state = pop[idx2 * G + i].next_state;
			}
		}
		else {
			if (curand_uniform(&localState) <= pm1) {
				pop[min1 * G + i].action = curand(&localState) % 3;
				pop[min1 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min1 * G + i].action = pop[idx2 * G + i].action;
				pop[min1 * G + i].next_state = pop[idx2 * G + i].next_state;
			}
			if (curand_uniform(&localState) <= pm2) {
				pop[min2 * G + i].action = curand(&localState) % 3;
				pop[min2 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min2 * G + i].action = pop[idx1 * G + i].action;
				pop[min2 * G + i].next_state = pop[idx1 * G + i].next_state;
			}
		}
		pop[idx1 * G + i].fitness = 0; pop[idx1 * G + i].used = 0; pop[idx1 * G + i].m_fitness = 0;
		pop[idx2 * G + i].fitness = 0; pop[idx2 * G + i].used = 0; pop[idx2 * G + i].m_fitness = 0;
		pop[min1 * G + i].fitness = 0; pop[min1 * G + i].used = 0; pop[min1 * G + i].m_fitness = 0;
		pop[min2 * G + i].fitness = 0; pop[min2 * G + i].used = 0; pop[min2 * G + i].m_fitness = 0;

	}
	state[id] = localState;
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

int main(int argc, char** argv) {
	//argv[1] = "5"; argv[2] = "4"; argv[3] = "1"; argv[4] = "0"; argv[5] = "2";
	//argv[1] = "5"; argv[2] = "10"; argv[3] = "11";
	//1 4 1 0 1 
	//argv[1] = "1"; argv[2] = "4"; argv[3] = "1"; argv[4] = "0"; argv[5] = "1";
	time_t dur = time(0);
	// generate N number of random values and pass them to GPU as initial seeds
	srand(time(0));

	// various variables
	bool writeToFile = true;			// write to file
	int block_size = 256;				// number of threads in a block
	int K = 2000;						// minimum number of generations
	int N = block_size * atoi(argv[1]);	// number of individuals in population
	int S = atoi(argv[2]);				// number of states
	int P = 128 * atoi(argv[3]);		// number of boards for each individual
	int T = atoi(argv[4]);				// 0 fully random, 1 sum occ, 2 gene fitness
	int L = atoi(argv[5]);				// run no
	int C = 383;						// number of combinations -- (int)pow(3, 8);
	int G = S * C + 1;					// number of genes in the individual
	int* Q;								// generate N number of random seeds on host
	int* F;								// fitness matrix
	float* arr_avgfit;					// average fitnesses for each individual
	int M = 80;							// number of moves allowed
	int R = 6;							// size of board
	int* idx;
	// bool B = (T == 0) ? true : false;

	char fname[50];
	char bname[60];
	char sname[50];
	char gname[50];
	//char nname[50];
	sprintf(fname, "txt/r-%d-%d-%d-%d-%d.txt", N, P, S, T, L);
	sprintf(bname, "txt/b-%d-%d-%d-%d-%d.txt", N, P, S, T, L);
	sprintf(sname, "txt/s-%d-%d-%d-%d-%d.txt", N, P, S, T, L);
	sprintf(gname, "txt/f-%d-%d-%d-%d-%d.txt", N, P, S, T, L);
	//sprintf(nname, "txt/N-%d-%d-%d-%d-%d.txt", N, P, S, T, L);
	FILE* results, * rsltall, * bestf, * statf, * genef, * nstatf;

	if (writeToFile) {
		results = fopen(fname, "w");
		rsltall = fopen("txt/results-f.csv", "a");
		bestf = fopen(bname, "w");
		statf = fopen(sname, "w");
		genef = fopen(gname, "w");
		//nstatf = fopen(nname, "w");
	}

	gene* pop;
	cudaError_t cudaStatus = cudaMallocManaged(&pop, N * G * sizeof(gene));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (pop)\n");
		return -1;
	}

	gene* best;
	cudaStatus = cudaMallocManaged(&best, G * sizeof(gene));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (best)\n");
		return -1;
	}

	int iidx[383] = { 0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31, 36, 37, 39, 40, 78, 79, 81, 82, 84, 85, 90, 91, 93, 94, 108, 109, 111, 112, 117, 118, 120, 121, 159, 160, 243, 244, 246, 247, 252, 253, 255, 256, 270, 271, 273, 274, 279, 280, 282, 283, 321, 322, 324, 325, 327, 328, 333, 334, 336, 337, 351, 352, 354, 355, 360, 361, 363, 364, 402, 403, 702, 703, 705, 706, 711, 712, 714, 715, 726, 727, 729, 730, 732, 733, 738, 739, 741, 742, 756, 757, 759, 760, 765, 766, 768, 769, 807, 808, 810, 811, 813, 814, 819, 820, 822, 823, 837, 838, 840, 841, 846, 847, 849, 850, 888, 972, 973, 975, 976, 981, 982, 984, 985, 999, 1000, 1002, 1003, 1008, 1009, 1011, 1012, 1050, 1051, 1053, 1054, 1056, 1057, 1062, 1063, 1065, 1066, 1080, 1081, 1083, 1084, 1089, 1090, 1092, 1131, 1431, 1432, 1434, 1435, 1440, 1443, 1455, 2187, 2188, 2190, 2191, 2196, 2197, 2199, 2200, 2214, 2215, 2217, 2218, 2223, 2224, 2226, 2227, 2265, 2266, 2268, 2269, 2271, 2272, 2277, 2278, 2280, 2281, 2295, 2296, 2298, 2299, 2304, 2305, 2307, 2308, 2346, 2347, 2430, 2431, 2433, 2434, 2439, 2440, 2442, 2443, 2457, 2458, 2460, 2461, 2466, 2467, 2469, 2470, 2508, 2509, 2511, 2512, 2514, 2515, 2520, 2521, 2523, 2524, 2538, 2539, 2541, 2542, 2547, 2548, 2550, 2589, 2590, 2889, 2890, 2892, 2893, 2898, 2899, 2901, 2902, 2913, 2914, 2916, 2917, 2919, 2920, 2925, 2926, 2928, 2929, 2943, 2944, 2946, 2947, 2952, 2953, 2955, 2956, 2994, 2995, 2997, 2998, 3000, 3001, 3006, 3007, 3009, 3010, 3024, 3025, 3027, 3028, 3033, 3034, 3036, 3075, 3159, 3160, 3162, 3163, 3168, 3169, 3171, 3172, 3186, 3187, 3189, 3190, 3195, 3196, 3198, 3237, 3238, 3240, 3241, 3243, 3244, 3249, 3250, 3252, 3267, 3268, 3270, 3276, 3318, 3618, 3619, 3621, 3622, 3627, 3630, 3642, 4382, 4391, 4409, 4418, 4454, 4463, 4472, 4490, 4499, 4535, 4625, 4634, 4652, 4661, 4697, 4706, 4715, 4733, 4742, 4778, 5111, 5120, 5138, 5147, 5183, 5192, 5219, 5354, 5363, 5381, 5390, 5426, 5435, 5462, 6318, 6319, 6321, 6322, 6326, 6327, 6328, 6330, 6331, 6335, 6345, 6346, 6348, 6349, 6353, 6354, 6355, 6357, 6358, 6362, 6399, 6400, 6402, 6403, 6407, 6408, 6411, 6426, 6427, 6429, 6430, 6434, 6435, 6438, 6534, 6535, 6537, 6538, 6543, 6546 };
	int* ix;
	cudaStatus = cudaMallocManaged(&ix, 6561 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (ix)\n");
		return -1;
	}
	for (int i = 0; i < 6561; i++) ix[i] = 0; // first make them all zero, then
	for (int i = 0; i < 383; i++) ix[iidx[i]] = i;  // create the inverted index

	// random seeds for each board of each individual
	cudaStatus = cudaMallocManaged(&Q, N * P * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (Q_1)\n");
		return -1;
	}
	for (int i = 0; i < N * P; i++) Q[i] = rand() % INT_MAX;

	curandState* devStates;
	cudaMalloc((void**)&devStates, N * P * sizeof(curandState));

	int num_blocks = ((N * P) + block_size - 1) / block_size;
	setup_states <<<num_blocks, block_size >>> (devStates, Q);
	cudaFree(Q);

	cudaStatus = cudaMallocManaged(&F, N * P * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (F)\n");
		return -1;
	}

	cudaStatus = cudaMallocManaged(&arr_avgfit, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (arr_avgfit)\n");
		return -1;
	}

	cudaStatus = cudaMallocManaged(&idx, N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (idx)\n");
		return -1;
	}

	// consecutive kernel calls do not require cudaDeviceSynchronize since they are queued... 
	num_blocks = (N + block_size - 1) / block_size;
	init_population <<<num_blocks, block_size >>> (pop, G, S, devStates);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Error initializing kernel 'init_population'\nErr: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// we need a memory block for the boards... The size is R*R to N*P
	int* boards;
	cudaStatus = cudaMallocManaged(&boards, N * P * R * R * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged of boards\n");
		return -1;
	}

	int* statistics;
	cudaStatus = cudaMallocManaged(&statistics, N * C * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (statistics) ");
		return -1;
	}

	/*int* nstats;
	cudaStatus = cudaMallocManaged(&nstats, 6561 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (nstats) ");
		return -1;
	}
	cudaDeviceSynchronize();
	for (int i = 0; i < 6561; i++) nstats[i] = 0;
	cudaDeviceSynchronize();*/

	int* sum_occ;
	cudaStatus = cudaMallocManaged(&sum_occ, N * G * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (sum_occ) ");
		return -1;
	}
	cudaDeviceSynchronize(); // we need this so that the GPU can allocate memory for statistics and occ. 
	for (int i = 0; i < N * C; i++) statistics[i] = 0;
	for (int i = 0; i < N * G; i++) sum_occ[i] = 0;

	float* gene_fitness = (float*)malloc(G * sizeof(float));

	float best_ind_fitness = 0.0f;		// BEST individual
	float best_gen_fitness = 0.0f;		// BEST generation fitness
	float convergence = 0.0f;
	int i = 0;

	while (i < K || convergence > 0.97) {
		// N number of blocks, each having P number of threads... 
		// first generate N * P number of boards 
		cudaDeviceSynchronize();
		for (int il = 0; il < N * G; il++) sum_occ[il] = 0;
		cudaDeviceSynchronize();
		generate_boards <<<N, P >>> (boards, devStates, R);
		run_boards <<<N, P >>> (pop, boards, devStates, R, G, M, C, F, ix, sum_occ, statistics);// , nstats);
		num_blocks = (N + block_size - 1) / block_size;
		average_fitness <<<num_blocks, block_size >>> (P, F, arr_avgfit, G);
		float gen_fitness = 0.0f;
		float ind_fitness = 0.0f;
		cudaDeviceSynchronize();
		for (int j = 0; j < N; j++) {
			gen_fitness += arr_avgfit[j];
			if (ind_fitness < arr_avgfit[j]) {
				ind_fitness = arr_avgfit[j];
			}
			if (best_ind_fitness < arr_avgfit[j]) {
				best_ind_fitness = arr_avgfit[j];
				int k = 0;
				for (int tj = j * G; tj < (j + 1) * G; tj++) {
					best[k].action = pop[tj].action;
					best[k].next_state = pop[tj].next_state;
					k++;
				}
			}
		}

		gen_fitness = gen_fitness / (float)N;
		if (best_gen_fitness < gen_fitness) {
			best_gen_fitness = gen_fitness;
			// store the averages for this generation. 
			for (int j = 0; j < N; j++) {
				if (j > 0) {
					for (int k = 0; k < G; k++) {
						if (pop[j * G + k].used > 0)
							gene_fitness[k] += (float)pop[j * G + k].fitness / (float)pop[j * G + k].used;
						//printf("%f ", gene_fitness[k]);
					}
				}
				else {
					for (int k = 0; k < G; k++) {
						if (pop[j * G + k].used > 0)
							gene_fitness[k] = (float)pop[j * G + k].fitness / (float)pop[j * G + k].used;
						else gene_fitness[k] = 0.0f;
						//printf("%f ", gene_fitness[k]);
					}
				}
			}
			for (int k = 0; k < G; k++) {
				gene_fitness[k] = gene_fitness[k] / (float)N;
				// printf("(%f) ", gene_fitness[k]);
			}
		}
		// progress bar
		if (i % 1000 == 0) printf("M");
		else if (i % 100 == 0) printf("C");
		else printf(".");
		convergence = gen_fitness / ind_fitness;
		if (writeToFile) fprintf(results, "%0.2f ", gen_fitness);
		// shuffle... 
		shuffle(idx, N);
		int original_block_size = block_size;
		if (N / 4 <= block_size) {
			// 1 block yeterli... 
			//printf("bir blok yeterli");
			num_blocks = 1; block_size = N / 4;
		}
		else {
			num_blocks = ((N / 4) + block_size - 1) / block_size;
			block_size = N / (4 * num_blocks);
			//printf("bir blok yeterli degil!: %d\n\n", num_blocks);
		}
		//cudaDeviceSynchronize();
		crossover <<<num_blocks, block_size >>> (idx, arr_avgfit, pop, G, S, devStates, best_ind_fitness, best_gen_fitness, sum_occ, T);
		//cudaDeviceSynchronize();
		block_size = original_block_size;
		i++;
	}
	printf("\nBEST INDIVIDUAL: %0.4f ", best_ind_fitness);
	printf("\nBEST GEN FIT: %0.4f ", best_gen_fitness);
	dur = time(0) - dur;
	printf("\nCompleted in %d seconds\n", dur);
	cudaDeviceSynchronize();
	if (writeToFile)
	{
		fprintf(rsltall, "%d,%d,%d,%d,%d,%0.4f,%0.4f,%d\n", N, P, S, T, L, best_gen_fitness, best_ind_fitness, dur);
		for (int i = 0; i < G; i++) {
			fprintf(bestf, "%d %d ", best[i].action, best[i].next_state);
		}

		int* final_statistics = (int*)malloc(C * sizeof(int));
		for (int i = 0; i < C; i++) final_statistics[i] = 0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < C; j++) {
				final_statistics[j] += statistics[i * C + j];
			}
		}
		for (int i = 0; i < C; i++) {
			fprintf(statf, "%d ", final_statistics[i]);
		}

		float* final_gene_fitness = (float*)malloc(C * sizeof(float));
		for (int i = 0; i < C; i++) final_gene_fitness[i] = 0.0f;

		for (int i = 0; i < S; i++) {
			for (int j = 0; j < C; j++) {
				if (gene_fitness[i * C + j] > final_gene_fitness[j])
					final_gene_fitness[j] = gene_fitness[i * C + j];
			}
		}
		for (int i = 0; i < C; i++) {
			fprintf(genef, "%0.4f ", final_gene_fitness[i]); // /(float)S
		}

		//for (int i = 0; i < 6561; i++)
		//	fprintf(nstatf, "%d ", nstats[i]);


		fclose(results);
		fclose(rsltall);
		fclose(bestf);
		fclose(statf);
		fclose(genef);
		//fclose(nstatf);
	}

	return 0;
}