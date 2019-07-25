#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

constexpr auto M_PI = 3.14159265358979323846264338327950288;

typedef struct gene_struct {
	int next_state;
	int action; 
} gene;

typedef struct position {
	int x;
	int y;
} pos;

__global__ void setup_states(curandState* state, int* Q) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(Q[id], id, 0, &state[id]);
}

__global__ void init_population(gene* pop, size_t pitch, int G, int S, curandState* state) { 
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = state[id];

	gene* ind = (gene*)((char*)pop + id * pitch);
	for (int i = 0; i < G; i++) {
		ind[i].action = curand(&localState) % 3;
		ind[i].next_state = curand(&localState) % S; 
		// check randomness if necessary
		/*if (id == 200)
			printf("(%d, %d)\n", ind[i].action, ind[i].next_state);*/
	}
}

__global__ void generate_boards(int* boards, size_t pitch, curandState* state, int R) {
	int id = blockIdx.x * blockDim.x + threadIdx.x; 
	curandState localState = state[blockIdx.x];

	int* board = (int*)((char*)boards + id * pitch);
	while (1) {
		for (int i = 0; i < R * R; i++) board[i] = 0; // first let all be 0
		int i = 0;
		do {
			int x = (curand(&localState) % (R - 2)) + 1;
			int y = (curand(&localState) % (R - 2)) + 1;
			if (board[x * R + y] == 0) {
				board[x * R + y] = 1;
				i++;
			}
		} while (i < 6);  // magic number 6 -> number of boxes... 

		int repeat = 0;
		for (int i = 0; i < R * R; i++) {
			if (
				board[i] == 1 &&
				board[i + 1] == 1 &&
				board[i + R] == 1 &&
				board[i + R + 1] == 1
				) {
				repeat = 1;
				break;
			}
		}
		if (repeat == 0) break;
	}
	/*if (id == 100) {   // check a board... 
		for (int i = 0; i < 36; i++) {
			printf("%d ", board[i]);
			if ((i + 1) % 6 == 0) printf("\n");
		}
	}*/
}

__global__ void run_board(
	gene* pop, 
	size_t pitch, 
	int* boards, 
	size_t board_pitch, 
	curandState* state,
	int R) {
	// blockIdx.x is the individual out of N individuals
	// threadIdx.x is the board out of P boards for that individual... 
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = state[blockIdx.x];

	// get the individual
	gene* ind = (gene*)((char*)pop + blockIdx.x * pitch); 
	// get the board... 
	int* board = (int*)((char*)boards + id * board_pitch);
	// find a random position on board
	pos cp; cp.x = -1; cp.y = -1;
	while (1) {
		cp.x = (curand(&localState) % (R - 2)) + 1;
		cp.y = (curand(&localState) % (R - 2)) + 1;
		if (board[cp.x * R + cp.y] == 0)
			break;
	}
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
	
	// if(blockIdx.x == 0 || blockIdx.x == 1)
		// printf("blockIdx: %d, threadIdx: %d\n", blockIdx.x, threadIdx.x); 
}

int main(int argc, char** argv) {
	/* Pseudocode 
	 * 1 - init population: N number of individuals that are made up of G 
	 *     number of genes. --- DONE --- 
	 * 2 - Loop generations. This cannot be in parallel, has to be sequential.
	 *   a - Loop individuals - we can make this run in parallel, however each
	         individual will run on several boards, which can also be in 
			 parallel. 
			 Number of blocks -> number of individuals
			 Number of threads per block -> number of boards 
		 b - We need a random board, a random position, and a random direction
		 c - The moves of the bulldozer must be in parallel, so no parallelism
		 d - The rotation code can be an inline function (check docs)
		 e - Find the average fitness for individual by averaging fitness 
		     values for the boards. 
	     f - Find the average fitness of generation by averaging average 
		     fitness of each individual 
	     g - Generating new individuals using uniform crossover. This can be in
		     parallel. Groups of four, so we can have number of individuals / 4
			 number of threads in as many blocks as possible. 
	 */
	
	// generate N number of random values and pass them to GPU as initial seeds
	srand(time(0));

	// Let's try to use cudaMallocPitch to create a two dimensional array for 
	// the population. 
	int N = 256;						// number of individuals in population
	int S = 4;							// number of states
	int C = (int)pow(3, 8);				// number of combinations
	int G = S * C + 1;					// number of genes in the individual
	int* Q;								// generate N number of random seeds on host
	int P = 128;						// number of boards for each individual
	int M = 80;							// number of moves allowed
	int R = 6;							// size of board
	// generations
	int K = 1000;						// number of generations
	
	gene* pop; 
	size_t pitch;
	cudaError_t cudaStatus = cudaMallocPitch(&pop, &pitch, G * sizeof(gene), N);
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocPitch\n");
		return -1;
	}
	
	int block_size = 256;				// number of threads in a block
	int num_blocks = (N + block_size - 1) / block_size;

	curandState* devStates;
	cudaMalloc((void**)& devStates, N * sizeof(curandState));

	cudaStatus = cudaMallocManaged(&Q, N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocManaged (Q_1)\n");
		return -1;
	}
	for (int i = 0; i < N; i++) Q[i] = rand() % 65536;
	setup_states<<<num_blocks, block_size>>>(devStates, Q);
	cudaFree(Q);

	// consecutive kernel calls do not require cudaDeviceSynchronize since they are queued... 
	init_population<<<num_blocks, block_size>>>(pop, pitch, G, S, devStates);// , G, N);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Error initializing kernel 'init_population'\nErr: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// we need a memory block for the boards... The size is R*R to N*P
	int* boards;
	size_t board_pitch;
	cudaStatus = cudaMallocPitch(&boards, &board_pitch, R * R * sizeof(int), N * P);
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocPitch of boards\n");
		return -1;
	}

	K = 1;
	for (int i = 0; i < K; i++) {					// loop generations
		// N number of blocks, each having P number of threads... 
		// first generate N * P number of boards 
		//generate_boards<<<N, P>>>(boards, board_pitch, xxx);
		generate_boards<<<N, P>>>(boards, board_pitch, devStates, R);
		run_board<<<N, P>>>(pop, pitch, boards, board_pitch, devStates, R);
	}
	
	return 0;
}