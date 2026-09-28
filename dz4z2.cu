#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

// nvcc -O2 -lm -arch=sm_60 feyman.cu -o feyman 

#define ACCURACY 0.01
#define NUM_OF_GPU_THREADS 1024

void compare_results(double err, double err_cuda);

int i4_ceiling(double x)
{
  int value = (int)x;
  if (value < x)
    value = value + 1;
  return value;
}

int i4_min(int i1, int i2)
{
  int value;
  if (i1 < i2)
    value = i1;
  else
    value = i2;
  return value;
}

double potential(double a, double b, double c, double x, double y, double z)
{
  return 2.0 * (pow(x / a / a, 2) + pow(y / b / b, 2) + pow(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}

double r8_uniform_01(int *seed)
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
  r = (double)(*seed) * 4.656612875E-10;

  return r;
}

__device__ float potential_device(float a, float b, float c, float x, float y, float z)
{
  return 2.0f * (powf(x / a / a, 2) + powf(y / b / b, 2) + powf(z / c / c, 2)) + 1.0f / a / a + 1.0f / b / b + 1.0f / c / c;
}

__device__ float r8_uniform_01_device(int *seed)
{
  int k;
  float r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
  r = (float)(*seed) * 4.656612875E-10f;

  return r;
}

__host__ __device__ inline double sq(double v) { return v * v; }
__device__ inline float sqf(float v) { return v * v; }

__global__ void feynman_points_kernel(int *inside_flag, double *w_exact_arr,
                                       int ni, int nj, int nk,
                                       float a, float b, float c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = ni * nj * nk;

  if (idx >= total) return;

  int i = idx / (nj * nk);
  int j = (idx / nk) % nj;
  int k = idx % nk;

  float x = ((float)(ni - (i + 1)) * (-a) + (float)(i) * a) / (float)(ni - 1);
  float y = ((float)(nj - (j + 1)) * (-b) + (float)(j) * b) / (float)(nj - 1);
  float z = ((float)(nk - (k + 1)) * (-c) + (float)(k) * c) / (float)(nk - 1);

  float chk = sqf(x / a) + sqf(y / b) + sqf(z / c);

  if (1.0f < chk)
  {
    inside_flag[idx] = 0;
    w_exact_arr[idx] = 0.0;
    return;
  }

  inside_flag[idx] = 1;
  w_exact_arr[idx] = (double) expf(sqf(x / a) + sqf(y / b) + sqf(z / c) - 1.0f);
}

__global__ void feynman_trials_kernel(double *wt_sum, int *inside_flag,
                                       int ni, int nj, int nk,
                                       float a, float b, float c,
                                       float stepsz, float h, int N)
{
  long long gid = ( long long ) blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;
  long long total_work = ( long long ) total_points * N;

  if (gid >= total_work) return;

  int idx = ( int ) (gid / N);
  int trial = ( int ) (gid % N);

  if (!inside_flag[idx]) return;

  int i = idx / (nj * nk);
  int j = (idx / nk) % nj;
  int k = idx % nk;

  float x = ((float)(ni - (i + 1)) * (-a) + (float)(i) * a) / (float)(ni - 1);
  float y = ((float)(nj - (j + 1)) * (-b) + (float)(j) * b) / (float)(nj - 1);
  float z = ((float)(nk - (k + 1)) * (-c) + (float)(k) * c) / (float)(nk - 1);

  int local_seed = 123456789 + idx * 1000003 + trial * 17 + (i + 1) * 100 + (j + 1) * 10 + (k + 1);

  float x1 = x;
  float x2 = y;
  float x3 = z;
  float w = 1.0f;
  float chk = 0.0f;

  while (chk < 1.0f)
  {
    float dx, dy, dz, ut, us, vs, vh, we;

    ut = r8_uniform_01_device(&local_seed);
    if (ut < 1.0f / 3.0f)
    {
      us = r8_uniform_01_device(&local_seed) - 0.5f;
      dx = (us < 0.0f) ? -stepsz : stepsz;
    }
    else
      dx = 0.0f;

    ut = r8_uniform_01_device(&local_seed);
    if (ut < 1.0f / 3.0f)
    {
      us = r8_uniform_01_device(&local_seed) - 0.5f;
      dy = (us < 0.0f) ? -stepsz : stepsz;
    }
    else
      dy = 0.0f;

    ut = r8_uniform_01_device(&local_seed);
    if (ut < 1.0f / 3.0f)
    {
      us = r8_uniform_01_device(&local_seed) - 0.5f;
      dz = (us < 0.0f) ? -stepsz : stepsz;
    }
    else
      dz = 0.0f;

    vs = potential_device(a, b, c, x1, x2, x3);
    x1 = x1 + dx;
    x2 = x2 + dy;
    x3 = x3 + dz;

    vh = potential_device(a, b, c, x1, x2, x3);

    we = (1.0f - h * vs) * w;
    w = w - 0.5f * h * (vh * we + vs * w);

    chk = sqf(x1 / a) + sqf(x2 / b) + sqf(x3 / c);
  }

  atomicAdd(&wt_sum[idx], (double) w);
}

double feynman_cuda(int ni, int nj, int nk, double a, double b, double c,
                     double stepsz, double h, int N, double *elapsed_seconds)
{
  int total = ni * nj * nk;
  long long total_work = ( long long ) total * N;

  double *w_exact_arr = (double *) malloc(total * sizeof(double));
  int *inside_flag = (int *) malloc(total * sizeof(int));
  double *wt_sum = (double *) calloc(total, sizeof(double));

  double *d_w_exact_arr, *d_wt_sum;
  int *d_inside_flag;

  cudaError_t err;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaMalloc((void **)&d_w_exact_arr, total * sizeof(double));
  cudaMalloc((void **)&d_inside_flag, total * sizeof(int));
  cudaMalloc((void **)&d_wt_sum, total * sizeof(double));
  cudaMemset(d_wt_sum, 0, total * sizeof(double));

  int blockSize = NUM_OF_GPU_THREADS;

  float af = (float) a, bf = (float) b, cf = (float) c;
  float stepszf = (float) stepsz, hf = (float) h;

  int pointsGridSize = (total + blockSize - 1) / blockSize;
  feynman_points_kernel<<<pointsGridSize, blockSize>>>(d_inside_flag, d_w_exact_arr,
                                                        ni, nj, nk, af, bf, cf);

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("feynman_points_kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  long long trialsGridSize = (total_work + blockSize - 1) / blockSize;
  feynman_trials_kernel<<<( unsigned int ) trialsGridSize, blockSize>>>(d_wt_sum, d_inside_flag,
                                                                          ni, nj, nk, af, bf, cf,
                                                                          stepszf, hf, N);

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("feynman_trials_kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(w_exact_arr, d_w_exact_arr, total * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(inside_flag, d_inside_flag, total * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(wt_sum, d_wt_sum, total * sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  *elapsed_seconds = (double) elapsed_ms / 1000.0;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_w_exact_arr);
  cudaFree(d_inside_flag);
  cudaFree(d_wt_sum);

  double err_sum = 0.0;
  int n_inside = 0;

  for (int idx = 0; idx < total; idx++)
  {
    if (inside_flag[idx])
    {
      double wt = wt_sum[idx] / (double)(N);
      err_sum += sq(w_exact_arr[idx] - wt);
      n_inside++;
    }
  }

  free(w_exact_arr);
  free(inside_flag);
  free(wt_sum);

  return sqrt(err_sum / (double)(n_inside));
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

int main(int arc, char **argv)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  double chk;
  int dim = 3;
  double dx;
  double dy;
  double dz;
  double err;
  double h = 0.001;
  int i;
  int j;
  int k;
  int n_inside;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int seed = 123456789;
  int steps;
  int steps_ave;
  int trial;
  double us;
  double ut;
  double vh;
  double vs;
  double x;
  double x1;
  double x2;
  double x3;
  double y;
  double w;
  double w_exact;
  double we;
  double wt;
  double z;

  std::chrono::high_resolution_clock::time_point time_start;
  double time_elapsed_seq, time_elapsed_cuda;

  int N = atoi(argv[1]);
  timestamp();

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c))
  {
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c))
  {
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else
  {
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  err = 0.0;
  n_inside = 0;

  // Sequential
  time_start = std::chrono::high_resolution_clock::now();
  for (i = 1; i <= ni; i++)
  {
    x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);

    for (j = 1; j <= nj; j++)
    {
      y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);

      for (k = 1; k <= nk; k++)
      {
        z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
          w_exact = 1.0;
          wt = 1.0;
          steps_ave = 0;
          continue;
        }

        n_inside++;

        w_exact = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);

        wt = 0.0;
        steps = 0;
        for (trial = 0; trial < N; trial++)
        {
          x1 = x;
          x2 = y;
          x3 = z;
          w = 1.0;
          chk = 0.0;
          while (chk < 1.0)
          {
            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dx = -stepsz;
              else
                dx = stepsz;
            }
            else
              dx = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dy = -stepsz;
              else
                dy = stepsz;
            }
            else
              dy = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dz = -stepsz;
              else
                dz = stepsz;
            }
            else
              dz = 0.0;

            vs = potential(a, b, c, x1, x2, x3);
            x1 = x1 + dx;
            x2 = x2 + dy;
            x3 = x3 + dz;

            steps++;

            vh = potential(a, b, c, x1, x2, x3);

            we = (1.0 - h * vs) * w;
            w = w - 0.5 * h * (vh * we + vs * w);

            chk = pow(x1 / a, 2) + pow(x2 / b, 2) + pow(x3 / c, 2);
          }
          wt = wt + w;
        }
        wt = wt / (double)(N);
        steps_ave = steps / (double)(N);

        err = err + pow(w_exact - wt, 2);
      }
    }
  }
  err = sqrt(err / (double)(n_inside));
  time_elapsed_seq = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time_start).count();

  printf("Non parallel time:\t %f\n", time_elapsed_seq);

  // CUDA
  double err_cuda = feynman_cuda(ni, nj, nk, a, b, c, stepsz, h, N, &time_elapsed_cuda);

  printf("CUDA time:\t %f\n", time_elapsed_cuda);
  printf("Seq err: %f \t CUDA err: %f\n", err, err_cuda);

  compare_results(err, err_cuda);

  printf("Time for CUDA is %.2f times faster\n", time_elapsed_seq / time_elapsed_cuda);

  timestamp();

  printf("---------------------------\n");

  return 0;
}

void compare_results(double err, double err_cuda)
{
  if (fabs(err - err_cuda) < ACCURACY)
  {
    printf("Test PASSED\n");
  }
  else
  {
    printf("Test FAILED\n");
  }
}