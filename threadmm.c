#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

//Defining size of square matrices, tile, and thread amount
#define N 512
#define MAX_THREADS 16
#define BLOCK_SIZE 64

int A[N][N], B[N][N], C[N][N];

//Timer Function
double CLOCK(){
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return(t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

//Sets elements of A matrix to 1, B matrix to 2
void matrix_initialize(){
  int i, j;
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      A[i][j] = 1;
      B[i][j] = 2;
      C[i][j] = 0;
    }
  }
}

//Tiled Matrix Multiplication
void matrix_multiply_tiled() {
  int i, ii, j, jj, k, kk;
  for (ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (jj = 0; jj < N; jj += BLOCK_SIZE) {
          for (kk = 0; kk < N; kk += BLOCK_SIZE) {
                // Perform multiplication on the current tile
                for (i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                        for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

//Threaded Matrix Multiplication with 16 threads
void* matrix_multi(void* arg){
  int tid = *(int*)arg;
  int row_per_thread = N / MAX_THREADS;
  int start_row = tid * row_per_thread;
  int end_row = start_row + row_per_thread;

  int i, j, k;
  for(i = start_row; i < end_row; i++){
    for(j = 0; j < N; j++){
      for(k = 0; k < N; k++){
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  pthread_exit(NULL);
}


int main(){

  //initialize matrices
  matrix_initialize();

  int i, j;
  double start, end, start1, end1;

  int num_threads = MAX_THREADS;

  /Timing the tiled matrix multiplication
  start1 = CLOCK();
  matrix_multiply_tiled();
  end1 = CLOCK();

  printf("%d is the final element val for each\n", C[0][0]);
  double time1 = end1 - start1;
  printf("Time is %f\n", time1);

  //restart matrix
  matrix_initialize();

  pthread_t threads[num_threads];
  int thread_ids[num_threads];

  //Timing the threaded matrix multiplication
  start = CLOCK();

  for(i = 0; i < num_threads; i++){
    thread_ids[i] = i;
    pthread_create(&threads[i], NULL, matrix_multi, &thread_ids[i]);
  }
  for(i = 0; i < num_threads; i++){
    pthread_join(threads[i], NULL);
  }

  end = CLOCK();

  double time = end - start;
  printf("Time is %f\n", time);
  printf("%d is the final element val for each", C[0][0]);


  return 0;
}
