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
		pop[i].action = curand(&localState) % 3;
		pop[i].next_state = curand(&localState) % S;
	}
	state[id] = localState;
}

__global__ void generate_boards(int* boards, curandState* state, int R) {
	int id = blockIdx.x * blockDim.x + threadIdx.x; 
	// curandState localState = state[blockIdx.x];
	curandState localState = state[id];

	int s = id * R * R; 
	//int* board = (int*)((char*)boards + id * pitch);
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
	/*if (id == 100) {   // check a board... 
		for (int i = 0; i < 36; i++) {
			printf("%d ", board[i]);
			if ((i + 1) % 6 == 0) printf("\n");
		}
	}*/
}

__global__ void average_fitness(int P, int* F, float* avg_fit) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// we get the current thread with id - it will be as many as N
	// F is N * P, so we skip ahead P 
	int s = id * P; 
	int sum = 0;
	for (int i = s; i < s + P; i++) {
		sum += F[i];
	}
	avg_fit[id] = (float)sum / (float)P;
	//printf("%f \n", avg_fit[id]);
}

__global__ void run_boards(
	gene* pop, 
	int* boards, 
	curandState* state,
	int R,
	int G,
	int M,
	int C,
	int* F) {
	// blockIdx.x is the individual out of N individuals
	// threadIdx.x is the board out of P boards for that individual... 
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = state[id]; // used to be blockIdx.x

	// get the individual
	int ind = blockIdx.x * G; 
	// gene* ind = (gene*)((char*)pop + blockIdx.x * pitch); 
	// get the board... 
	int brd = id * R * R; 
	// int* board = (int*)((char*)boards + id * board_pitch);

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
	default:  // case 3 and others... 
		cd.x = -1; cd.y = 0;
		break;
	}

	int cs = pop[ind + G - 1].next_state;
	for (int i = 0; i < M; i++) {  // perform M moves (default 80)
		int cc = 0;
		for (int m = 0; m < 8; m++) {  // m is for the 8-neighborhood
			int cx = cp.x + cd.x; int cy = cp.y + cd.y; 
			if (cx < 0 || cy < 0 || cx >= R || cy >= R) {
				// then it is a wall (wall = 2)
				cc += powf(3, m) * 2;
			}
			else cc += powf(3, m) * boards[brd + (cx * R + cy)];
			rotate_ccw(&cd, 4);
		}
		int action = pop[ind + (cs * C + cc)].action;
		cs = pop[ind + (cs * C + cc)].next_state;

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
	state[id] = localState;
	// if(blockIdx.x == 0) printf("%d ", F[id]);
	// if(blockIdx.x == 0 || blockIdx.x == 1)
		// printf("blockIdx: %d, threadIdx: %d\n", blockIdx.x, threadIdx.x); 
}

__global__ void crossover(int* idx, float* f, gene* pop, int G, int S, curandState* state) {
	// id will be N/4, so we multiply by 4
	int id = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
	// curandState localState = state[blockIdx.x];
	curandState localState = state[id];

	float max1 = -1.0f, max2 = -1.0f;
	int idx1 = -1, idx2 = -1;
	for (int i = id; i < id + 4; i++) {
		if (f[idx[i]] > max1 && f[idx[i]] > max2) {
			max2 = max1;
			idx2 = idx1;
			max1 = f[idx[i]];
			idx1 = idx[i];
		} else if (f[idx[i]] > max2) {
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
	int X = 1000;
	for (int i = 0; i < G; i++) {
		if (curand(&localState) % 2 == 0) {
			if (curand(&localState) % X == 19) {
				pop[min1 * G + i].action = curand(&localState) % 3;
				pop[min1 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min1 * G + i].action = pop[idx1 * G + i].action;
				pop[min1 * G + i].next_state = pop[idx1 * G + i].next_state;
			}
			if (curand(&localState) % X == 20) {
				pop[min2 * G + i].action = curand(&localState) % 3;
				pop[min2 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min2 * G + i].action = pop[idx2 * G + i].action;
				pop[min2 * G + i].next_state = pop[idx2 * G + i].next_state;
			}
		}
		else {
			if (curand(&localState) % X == 21) {
				pop[min1 * G + i].action = curand(&localState) % 3;
				pop[min1 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min1 * G + i].action = pop[idx2 * G + i].action;
				pop[min1 * G + i].next_state = pop[idx2 * G + i].next_state;
			}
			if (curand(&localState) % X == 22) {
				pop[min2 * G + i].action = curand(&localState) % 3;
				pop[min2 * G + i].next_state = curand(&localState) % S;
			}
			else {
				pop[min2 * G + i].action = pop[idx1 * G + i].action;
				pop[min2 * G + i].next_state = pop[idx1 * G + i].next_state;
			}
		}
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
	// generate N number of random values and pass them to GPU as initial seeds
	srand(time(0));

	// various variables
	int block_size = 256;				// number of threads in a block
	int K = 1000;						// number of generations
	int N = block_size * 4;				// number of individuals in population
	int P = 128;						// number of boards for each individual
	int S = 10;							// number of states
	int C = (int)pow(3, 8);				// number of combinations
	int G = S * C + 1;					// number of genes in the individual
	int* Q;								// generate N number of random seeds on host
	int* F;								// fitness matrix
	float* avg_fit;						// average fitnesses
	int M = 80;							// number of moves allowed
	int R = 6;							// size of board
	int* idx;

	char fname[50];
	sprintf(fname, "r-%d-%d-%d.txt", N, S, 10);
	FILE* results = fopen(fname, "w");

	gene* pop; 
	// instead of pitch, let's try managed... 
	cudaError_t cudaStatus = cudaMallocManaged(&pop, N * G * sizeof(gene));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocPitch\n");
		return -1;
	}
	
	// random seeds for each board of each individual
	cudaStatus = cudaMallocManaged(&Q, N * P * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (Q_1)\n");
		return -1;
	}
	for (int i = 0; i < N * P; i++) Q[i] = rand() % INT_MAX;

	curandState* devStates;
	cudaMalloc((void**)& devStates, N * P * sizeof(curandState));
	
	int num_blocks = ((N * P) + block_size - 1) / block_size;
	setup_states<<<num_blocks, block_size>>>(devStates, Q);
	cudaFree(Q);

	cudaStatus = cudaMallocManaged(&F, N * P * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (F)\n");
		return -1;
	}

	cudaStatus = cudaMallocManaged(&avg_fit, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (avg_fit)\n");
		return -1;
	}

	cudaStatus = cudaMallocManaged(&idx, N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (idx)\n");
		return -1;
	}

	// consecutive kernel calls do not require cudaDeviceSynchronize since they are queued... 
	num_blocks = (N + block_size - 1) / block_size;
	init_population<<<num_blocks, block_size>>>(pop, G, S, devStates);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Error initializing kernel 'init_population'\nErr: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// we need a memory block for the boards... The size is R*R to N*P
	int* boards;
	cudaStatus = cudaMallocManaged(&boards, N * P * R * R * sizeof(int)); 
	// cudaMallocPitch(&boards, &board_pitch, R * R * sizeof(int), N * P);
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocPitch of boards\n");
		return -1;
	}
	
	float topfit = 0.0f;

	for (int i = 0; i < K; i++) {					// loop generations
		// N number of blocks, each having P number of threads... 
		// first generate N * P number of boards 
		generate_boards<<<N, P>>>(boards, devStates, R);
		run_boards<<<N, P>>>(pop, boards, devStates, R, G, M, C, F);
		num_blocks = (N + block_size - 1) / block_size;
		average_fitness<<<num_blocks, block_size>>>(P, F, avg_fit);
		float gen_fitness = 0.0f;
		cudaDeviceSynchronize();
		for (int j = 0; j < N; j++) {
			gen_fitness += avg_fit[j];
			if (topfit < avg_fit[j]) 
				topfit = avg_fit[j];
		}
		printf("%0.2f ", gen_fitness / (float)N);
		fprintf(results, "%0.2f ", gen_fitness / (float)N);

		// add mutation adaptation here

		// moving on to crossover...
		// shuffle... 
		shuffle(idx, N);
		int original_block_size = block_size;
		if (N/4 <= block_size) {
			// 1 block yeterli... 
			num_blocks = 1; block_size = N / 4;
		}
		else {
			num_blocks = ((N / 4) + block_size - 1) / block_size;
		}
		crossover<<<num_blocks, block_size>>>(idx, avg_fit, pop, G, S, devStates);
		block_size = original_block_size;
	}
	printf("BEST: %0.4f ", topfit);
	fprintf(results, "\n\nBEST%0.4f", topfit);
	return 0;
}