#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define DEFAULT_H 1000
#define DEFAULT_W 1000
#define DEFAULT_CNT 200
#define DEFAULT_FILENAME "julia"
#define PARALLEL_FILENAME "julia_parallel"
#define NUM_OF_GPU_THREADS 1024

int main (int argc, char *argv[]);
unsigned char *julia_set ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt );
void tga_write ( int w, int h, unsigned char rgb[], char *filename );
void timestamp ( );

unsigned char *julia_set_cuda ( int w, int h, int cnt, float xl, float xr, float yb, float yt, double *elapsed_seconds );
void update_filename(int w, int h, int cnt, char buffer[256], char *filename);
void compare_results(int w, int h, unsigned char rgb[], unsigned char rgb_cuda[]);

__device__ int julia_device ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt )
{
  float ai;
  float ar;
  float ci = 0.156f;
  float cr = -0.8f;
  int k;
  float t;
  float x;
  float y;

  x = ( ( float ) ( w - i - 1 ) * xl
      + ( float ) (     i     ) * xr )
      / ( float ) ( w     - 1 );

  y = ( ( float ) ( h - j - 1 ) * yb
      + ( float ) (     j     ) * yt )
      / ( float ) ( h     - 1 );

  ar = x;
  ai = y;

  for ( k = 0; k < cnt; k++ )
  {
    /* explicit non-fused IEEE ops so results match the CPU's separate mul+add
       (nvcc otherwise contracts a*b+c into a single fma instruction) */
    t  = __fadd_rn ( __fsub_rn ( __fmul_rn ( ar, ar ), __fmul_rn ( ai, ai ) ), cr );
    ai = __fadd_rn ( __fadd_rn ( __fmul_rn ( ar, ai ), __fmul_rn ( ai, ar ) ), ci );
    ar = t;

    if ( 1000 < __fadd_rn ( __fmul_rn ( ar, ar ), __fmul_rn ( ai, ai ) ) )
    {
      return 0;
    }
  }

  return 1;
}

__global__ void julia_kernel ( unsigned char *rgb, int w, int h, int cnt, float xl, float xr, float yb, float yt )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i < w && j < h )
  {
    int juliaValue = julia_device ( w, h, xl, xr, yb, yt, i, j, cnt );
    int k = 3 * ( j * w + i );

    rgb[k]   = 255 * ( 1 - juliaValue );
    rgb[k+1] = 255 * ( 1 - juliaValue );
    rgb[k+2] = 255;
  }
}

int main (int argc, char *argv[] )  {
  int h = DEFAULT_H;
  int w = DEFAULT_W;
  int cnt = DEFAULT_CNT;
  char filename[256] = DEFAULT_FILENAME;
  char cuda_filename[256] = PARALLEL_FILENAME;
  char buffer[256];
  unsigned char *rgb, *rgb_cuda;
  std::chrono::high_resolution_clock::time_point time_start;
  double time_elapsed_seq, time_elapsed_cuda;
  float xl = - 1.5;
  float xr = + 1.5;
  float yb = - 1.5;
  float yt = + 1.5;

  if (argc == 4) {
    h = atoi(argv[1]);
    w = atoi(argv[2]);
    cnt = atoi(argv[3]);
    if (!h || !w || !cnt) return 1;
  }

  timestamp();

  printf ( "\n" );
  printf ( "JULIA Set\n" );
  printf ( "  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n" );

  // Sequential
  time_start = std::chrono::high_resolution_clock::now();
  rgb = julia_set ( w, h, cnt, xl, xr, yb, yt );
  time_elapsed_seq = std::chrono::duration<double>( std::chrono::high_resolution_clock::now() - time_start ).count();

  printf("Non parallel time:\t %f\n", time_elapsed_seq);

  // CUDA
  rgb_cuda = julia_set_cuda ( w, h, cnt, xl, xr, yb, yt, &time_elapsed_cuda );

  printf("CUDA time:\t %f\n", time_elapsed_cuda);

  printf("Time for CUDA is %.2f times faster\n", time_elapsed_seq/time_elapsed_cuda);

  update_filename(w, h, cnt, buffer, filename);
  update_filename(w, h, cnt, buffer, cuda_filename);

  tga_write ( w, h, rgb, filename );
  tga_write ( w, h, rgb_cuda, cuda_filename );

  compare_results(w, h, rgb, rgb_cuda);

  free(rgb);
  free(rgb_cuda);

  timestamp();

  printf ( "\n" );
  printf ( "JULIA set:\n" );
  printf ( "-------Normal end of execution.-------\n" );

  return 0;
}

unsigned char *julia_set ( int w, int h, int cnt, float xl, float xr, float yb, float yt )
{
  int i;
  int j;
  int juliaValue;
  int k;
  unsigned char *rgb;

  rgb = ( unsigned char * ) malloc ( w * h * 3 * sizeof ( unsigned char ) );

  for ( j = 0; j < h; j++ )
  {
    for ( i = 0; i < w; i++ )
    {
    juliaValue = julia ( w, h, xl, xr, yb, yt, i, j, cnt );

    k = 3 * ( j * w + i );

    rgb[k]   = 255 * ( 1 - juliaValue );
    rgb[k+1] = 255 * ( 1 - juliaValue );
    rgb[k+2] = 255;
    }
  }

  return rgb;
}

unsigned char *julia_set_cuda ( int w, int h, int cnt, float xl, float xr, float yb, float yt, double *elapsed_seconds )
{
  unsigned char *rgb;
  unsigned char *d_rgb;
  size_t size = ( size_t ) w * h * 3 * sizeof ( unsigned char );

  rgb = ( unsigned char * ) malloc ( size );

  cudaError_t err;

  cudaEvent_t start, stop;
  cudaEventCreate ( &start );
  cudaEventCreate ( &stop );

  cudaEventRecord ( start );

  err = cudaMalloc ( ( void ** ) &d_rgb, size );
  if ( err != cudaSuccess )
  {
    printf ( "cudaMalloc failed: %s\n", cudaGetErrorString ( err ) );
  }

  int blockDim2D = ( int ) sqrt ( ( double ) NUM_OF_GPU_THREADS );
  dim3 blockSize ( blockDim2D, blockDim2D );
  dim3 gridSize ( ( w + blockSize.x - 1 ) / blockSize.x,
                  ( h + blockSize.y - 1 ) / blockSize.y );

  julia_kernel<<<gridSize, blockSize>>> ( d_rgb, w, h, cnt, xl, xr, yb, yt );

  err = cudaGetLastError();
  if ( err != cudaSuccess )
  {
    printf ( "Kernel launch failed: %s\n", cudaGetErrorString ( err ) );
  }

  cudaMemcpy ( rgb, d_rgb, size, cudaMemcpyDeviceToHost );

  cudaEventRecord ( stop );
  cudaEventSynchronize ( stop );

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime ( &elapsed_ms, start, stop );
  *elapsed_seconds = ( double ) elapsed_ms / 1000.0;

  cudaEventDestroy ( start );
  cudaEventDestroy ( stop );

  cudaFree ( d_rgb );

  return rgb;
}

int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt )
{
  float ai;
  float ar;
  float ci = 0.156;
  float cr = -0.8;
  int k;
  float t;
  float x;
  float y;

  x = ( ( float ) ( w - i - 1 ) * xl
      + ( float ) (     i     ) * xr )
      / ( float ) ( w     - 1 );

  y = ( ( float ) ( h - j - 1 ) * yb
      + ( float ) (     j     ) * yt )
      / ( float ) ( h     - 1 );

	  ar = x;
  ai = y;

  for ( k = 0; k < cnt; k++ )
  {
    t  = ar * ar - ai * ai + cr;
    ai = ar * ai + ai * ar + ci;
    ar = t;

    if ( 1000 < ar * ar + ai * ai )
    {
      return 0;
    }
  }

  return 1;
}

void tga_write ( int w, int h, unsigned char rgb[], char *filename )
{
  FILE *file_unit;
  unsigned char header1[12] = { 0,0,2,0,0,0,0,0,0,0,0,0 };
  unsigned char header2[6] = { w%256, w/256, h%256, h/256, 24, 0 };

  file_unit = fopen ( filename, "wb" );

  fwrite ( header1, sizeof ( unsigned char ), 12, file_unit );
  fwrite ( header2, sizeof ( unsigned char ), 6, file_unit );

  fwrite ( rgb, sizeof ( unsigned char ), 3 * w * h, file_unit );

  fclose ( file_unit );

  return;
}

void timestamp ( void )
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

void compare_results(int w, int h, unsigned char rgb[], unsigned char rgb_cuda[])
{
  for(int i = 0; i < 3*w*h; i++)
  {
    if(rgb[i] != rgb_cuda[i])
    {
      printf("Test FAILED\n");
      return;
    }
  }
  printf("Test PASSED\n");
}

void update_filename(int w, int h, int cnt, char buffer[256], char *filename)
{
  strcat(filename, "_");
  sprintf(buffer, "%d", h);
  strcat(filename, buffer);
  strcat(filename, "_");
  sprintf(buffer, "%d", w);
  strcat(filename, buffer);
  strcat(filename, "_");
  sprintf(buffer, "%d", cnt);
  strcat(filename, buffer);
  strcat(filename, ".tga");
}