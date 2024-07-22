#include <stdio.h>
#include<stdlib.h>
#include <time.h>

double CLOCK() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC,  &t);
  return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


#define M 512
#define tile_size 64

main(int argc, char **argv)
{
  int i,j,k;
  double start, finish, total1, total2, total3, sum;
  float a[M][M], b[M][M], c[M][M];

  //Setting values for each matrix
  for (i=0; i<M; i++)
    for (j=0; j<M; j++)
      a[i][j] = 1;

  for (i=0; i<M; i++)
    for (j=0; j<M; j++)
      b[i][j] = 2;

  for (i=0; i<M; i++)
    for (j=0; j<M; j++)
      c[i][j] = 0;


  //Regular Matrix Multiplication
  start = CLOCK();
  for (j=0; j<M; j++)
    for (i=0; i<M; i++)
      for (k = 0; k<M; k++) {
        c[i][j] = c[i][j] + a[i][k] * b[k][j];
      }
  finish = CLOCK();
  total1 = finish - start;
  printf("Time for the loop = %f\n", total1);
  printf("Sum is %f\n", c[99][99]);

  //Reseting the resultant matrix element values
  for(i = 0; i<M; i++)
    for(j = 0; j<M; j++){
      c[i][j]=0;
    }

  //Tiled Matrix Multiplication with Tile Size of
  start=CLOCK();
  int ii, jj, kk;
  for(ii = 0; ii < M; ii+=tile_size)
    for(jj = 0; jj < M; jj+=tile_size)
      for(kk = 0; kk < M; kk+=tile_size)
        for(i = ii; i < M && i < ii+tile_size; i++)
          for(j = jj; j < M && j < jj+tile_size; j++)
            for(k = kk; k < M && k < kk+tile_size; k++){
              c[i][j] += a[i][k] * b[k][j];
            }
  finish = CLOCK();
  total3 = finish - start;
  printf("Time for tiled loop = %f\n", total3);
  printf("Sum is %f\n", c[99][99]);

  return 0;
}






