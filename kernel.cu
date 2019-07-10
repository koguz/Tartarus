
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define M_PI 3.14159265358979323846264338327950288

struct gene {
	int next_state;
	int action;
};

struct pos {
	int x;
	int y;
};

void rotate_ccw(struct pos* c, double d) {
	struct pos r;
	r.x = (int)nearbyint(c->x * cos(M_PI / d) - c->y * sin(M_PI / d));
	r.y = (int)nearbyint(c->x * sin(M_PI / d) + c->y * cos(M_PI / d));
	c->x = r.x; c->y = r.y;
}

struct pos random_direction() {
	struct pos d;
	switch (rand() % 4) {
	case 0:
		d.x = 0; d.y = -1;
		break;
	case 1:
		d.x = 0; d.y = 1;
		break;
	case 2:
		d.x = 1; d.y = 0;
		break;
	case 3:
		d.x = -1; d.y = 0;
		break;
	}
	return d;
};

struct pos random_position(int** board, int R) {
	struct pos p;
	p.x = -1; p.y = -1;
	while (1) {
		p.x = (rand() % (R - 2)) + 1;
		p.y = (rand() % (R - 2)) + 1;
		if (board[p.x][p.y] == 0)
			break;
	}
	return p;
};

int** create_board(int R, int B) {
	int i, j;
	int** board = (int**)malloc(sizeof(int*) * R);
	for (i = 0; i < R; i++) {
		board[i] = (int*)malloc(sizeof(int) * R);
	}
	while (1) {
		for (i = 0; i < R; i++) {
			for (j = 0; j < R; j++)
				board[i][j] = 0;
		}
		i = 0;
		do {
			int x = (rand() % (R - 2)) + 1;
			int y = (rand() % (R - 2)) + 1;
			if (board[x][y] == 0) {
				board[x][y] = 1;
				i++;
			}
		} while (i < B);

		int repeat = 0;
		for (i = 0; i < R; i++) {
			for (j = 0; j < R; j++) {
				if (board[i][j] == 1 &&
					board[i + 1][j] == 1 &&
					board[i][j + 1] == 1 &&
					board[i + 1][j + 1] == 1
					) {
					repeat = 1;
					break;
				}
			}
			if (repeat == 1) break;
		}
		if (repeat == 0) break;
	}
	return board;
}

void free_board(int** board, int R) {
	int i;
	for (i = 0; i < R; i++)
		free(board[i]);
	free(board);
}

void print_board(int** board, int R) {
	int i, j;
	for (i = 0; i < R; i++) {
		for (j = 0; j < R; j++)
			printf("%d ", board[i][j]);
		printf("\n");
	}
}

int fitness(int** board, int R) {
	int i, j;
	int f = 0;
	for (i = 0; i < R; i++) {
		for (j = 0; j < R; j++) {
			if (board[i][j] == 1) {
				if (i == 0 || i == R - 1) f++;
				if (j == 0 || j == R - 1) f++;
			}
		}
	}
	return f;
}

void shuffle(int* arr, int S) {  // Fisher-Yates Shuffle
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
	srand(time(0));
	int S = atoi(argv[1]);
	int T = atoi(argv[2]);
	char fname[50];
	printf("%d", atoi(argv[1]));
	sprintf(fname, "adaptive_results-%d-%d.txt", S, T);
	FILE* results;              // save the results to a file
	int i, j, k, m, n;          // variables of /for/ loops
	int B = 6;                  // number of boxes
	int R = 6;                  // size of board
	int N = 800;                // number of individuals in the population
	//int S = 7;                  // number of states
	int C = (int)pow(3, 8);    // combinations of wall, box and empty cells
	int G = S * C + 1;          // number of genes in the chromosome
	int K = 1000;               // number of generations
	int M = 80;                 // number of moves allowed
	int P = 100;                // number of boards for each individual
	int A = 1000;               // mutation rate; one in an A (one in a 1000)
	int D = 50;                 // adaptive check interval
	float* f = (float*)malloc(sizeof(float) * N);
	float* d = (float*)malloc(sizeof(float) * K);
	float fg = 0.0f;
	int* idx = (int*)malloc(sizeof(int) * N);
	float alimit = 0.1f;

	struct gene** pop = (struct gene**)malloc(sizeof(struct gene*) * N);
	for (i = 0; i < N; i++) {
		pop[i] = (struct gene*)malloc(sizeof(struct gene) * G);
		for (j = 0; j < G; j++) {
			pop[i][j].next_state = rand() % S;
			pop[i][j].action = rand() % 3;
		}
	}

	results = fopen(fname, "w");
	for (i = 0; i < K; i++) {              // loop generations
		for (j = 0; j < N; j++) {          // loop individuals
			f[j] = 0;
			for (n = 0; n < P; n++) {       // loop boards
				int** board = create_board(R, B);
				struct pos cp = random_position(board, B);  // initial position
				struct pos cd = random_direction();         // initial direction
				int cs = pop[j][G - 1].next_state;            // initial state
				for (k = 0; k < M; k++) {                          // moves
					int cc = 0;
					for (m = 0; m < 8; m++) {   // magic number 8 is for 8-neighborhood
						int cx = cp.x + cd.x; int cy = cp.y + cd.y;
						if (cx < 0 || cy < 0 || cx >= R || cy >= R) {
							// then it is a wall (wall = 2)
							cc += pow(3, m) * 2;
						}
						else cc += pow(3, m) * board[cx][cy];
						rotate_ccw(&cd, 4);
					}
					int action = (pop[j][(cs * C) + cc]).action;  // action id
					cs = (pop[j][(cs * C) + cc]).next_state;

					int cx, cy, dx, dy;
					switch (action) {
					case 0:
						cx = cp.x + cd.x;
						cy = cp.y + cd.y;
						if (cx >= 0 && cy >= 0 && cx < R && cy < R) {
							if (board[cx][cy] == 0) { // nothing in front of us... move...
								cp.x = cx; cp.y = cy;
							}
							else {
								// there is a box... if the box can move in the cd direction,
								// then move it and update your own position.
								// else, we have nothing to do.
								dx = cx + cd.x;
								dy = cy + cd.y;
								if (dx >= 0 && dy >= 0 && dx < R && dy < R && board[dx][dy] == 0) {
									// empty! move the box there...
									board[cx][cy] = 0;
									board[dx][dy] = 1;
									cp.x = cx; cp.y = cy;  // update your position
								}
							}
						} // else, it is a wall... do nothing...
						break;
					case 1:  // rotate left
						rotate_ccw(&cd, 0.66);
						break;
					case 2:  // rotate right
						rotate_ccw(&cd, 2);
						break;
					default:
						printf("Invalid action!\n");
						break;
					}
				}
				f[j] += fitness(board, R);
				free_board(board, R);
			} // end of boards
			f[j] = f[j] / P;
			fg += f[j];
		} // j -> end of individual fitness
		fg = fg / N;
		d[i] = fg;
		if (i > 0 && i % D == 0) {
			double fark = atan((d[i] - d[i - D]) / (float)D) * 180 / M_PI;
			printf("rate of change (%d to %d): %0.2f\n", i, i - D, fark);
			if (fark < alimit) {
				A = A * 9 / 10;
				alimit = alimit / 2.0f;
				printf("mutation rate set to: %d\n", A);
			}
		}
		printf("Generation fitness: %0.2f\n", fg);
		fprintf(results, "%0.2f ", fg);
		shuffle(idx, N);
		// The population is shuffled into 50 groups of four individuals. The fittest
		// of two are recombined to replace the weakest two...
		for (k = 0; k < N; k = k + 4) {
			// indices are in idx
			float max1 = -1;
			int idx1 = -1;
			float max2 = -1;
			int idx2 = -1;
			for (m = k; m < k + 4; m++) {
				if (f[idx[m]] > max1 && f[idx[m]] > max2) {
					max2 = max1;
					idx2 = idx1;
					max1 = f[idx[m]];
					idx1 = idx[m];
				}
				else if (f[idx[m]] > max2) {
					max2 = f[idx[m]];
					idx2 = idx[m];
				}
			}
			if (idx1 == idx2) {
				printf("idx1=idx2\n");
			}
			// max1 is the maximum, max2 is the second maximum
			// uniform crossover using idxes...
			struct gene* offspring1 = (struct gene*)malloc(sizeof(struct gene) * G);
			struct gene* offspring2 = (struct gene*)malloc(sizeof(struct gene) * G);
			for (m = 0; m < G; m++) {   // uniform crossover
				if (rand() % 2 == 0) {
					offspring1[m].action = pop[idx1][m].action;
					offspring1[m].next_state = pop[idx1][m].next_state;
					if (rand() % A == 19) { // mutation
						offspring1[m].action = rand() % 3;
						offspring1[m].next_state = rand() % S;
					}
					offspring2[m].action = pop[idx2][m].action;
					offspring2[m].next_state = pop[idx2][m].action;
					if (rand() % A == 20) { // mutation
						offspring2[m].action = rand() % 3;
						offspring2[m].next_state = rand() % S;
					}
				}
				else {
					offspring1[m].action = pop[idx2][m].action;
					offspring1[m].next_state = pop[idx2][m].next_state;
					if (rand() % A == 21) { // mutation
						offspring1[m].action = rand() % 3;
						offspring1[m].next_state = rand() % S;
					}
					offspring2[m].action = pop[idx1][m].action;
					offspring2[m].next_state = pop[idx1][m].action;
					if (rand() % A == 22) { // mutation
						offspring2[m].action = rand() % 3;
						offspring2[m].next_state = rand() % S;
					}
				}
			}
			int t = 0;
			for (m = k; m < k + 4; m++) {
				// replace least 2 with offsprings
				if (idx[m] != idx1 && idx[m] != idx2) {
					if (t == 0) {
						free(pop[idx[m]]);
						pop[idx[m]] = offspring1;
						if (idx1 == idx2) t = 2;
						else t = 1;
					}
					else {
						free(pop[idx[m]]);
						pop[idx[m]] = offspring2;
					}
				}
			}
		}
	} // i -> end of generations
	fclose(results);
	return 0;
}

