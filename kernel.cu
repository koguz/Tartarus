
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
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

__global__ void init_population(gene* pop, size_t pitch, int G, int S, curandState* state) { 
	int id = blockIdx.x; 
	curand_init(1234, id, 0, &state[id]);
	gene* ind = (gene*)((char*)pop + blockIdx.x * pitch);
	curandState localState = state[id];
	for (int i = 0; i < G; i++) {
		ind[i].action = curand(&localState) % 3;
		ind[i].next_state = curand(&localState) % S; 
		// printf("ind[%d] = (%d, %d) - ", i, ind[i].action, ind[i].next_state);
	}
	// printf("Individual %d, gene %d - %d\n", blockIdx.x, threadIdx.x, ind[100].action);
}

int main(int argc, char** argv) {
	/* Pseudocode 
	 * 1 - init population: N number of individuals that are made up of G 
	 *     number of genes. 
	 * 2 - Loop generations. This cannot be in parallel, has to be sequential.
	 *   a - Loop individuals - we can make this run in parallel, however each
	         individual will run on several boards, which can also be in 
			 parallel. 
			 Number of blocks -> number of individuals
			 Number of threads per block -> number of boards 
		 b - We need a random board, a random position, and a random direction
		 c - The moves of the bulldozer must be in parallel, so no parallelism
		 d - The rotation code could be an inline function (check docs)
		 e - Find the average fitness for individual by averaging fitness 
		     values for the boards. 
	     f - Find the average fitness of generation by averaging average 
		     fitness of each individual 
	     g - Generating new individuals using uniform crossover. This can be in
		     parallel. Groups of four, so we can have number of individuals / 4
			 number of threads in as many blocks as possible. 
	 */
	

	// Let's try to use cudaMallocPitch to create a two dimensional array for 
	// the population. 
	int N = 200;						// number of individuals in population
	int S = 4;							// number of states
	int C = (int)pow(3, 8);				// number of combinations
	int G = S * C + 1;					// number of genes in the individual
	gene* pop; 
	size_t pitch;
	cudaError_t cudaStatus = cudaMallocPitch(&pop, &pitch, G * sizeof(gene), N);
	if (cudaStatus != cudaSuccess) {
		printf("error in initialization of cudaMallocPitch\n");
		return -1;
	}
	curandState* devStates;
	cudaMalloc((void**)& devStates, N * sizeof(curandState));
	// int block_size = 1024;				// number of threads in a block
	// int num_blocks = ((N * G) + block_size - 1) / block_size;
	init_population <<<N, 1>>> (pop, pitch, G, S, devStates);// , G, N);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Error initializing kernel 'init_population'\nErr: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	
	return 0;
}