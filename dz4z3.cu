#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>


// nvcc -O2 -lm -arch=sm_60 feyman.cu -o feyman 

#define ACCURACY 0.01
#define NUM_OF_GPU_THREADS 1024
#define COMPUTE_BLOCK_SIZE 128

int main ( int argc, char *argv[] );
void compute ( int np, int nd, double pos[], double vel[],
  double mass, double f[], double *pot, double *kin );
double cpu_time ( );
double dist ( int nd, double r1[], double r2[], double dr[] );
void initialize ( int np, int nd, double pos[], double vel[], double acc[] );
void r8mat_uniform_ab ( int m, int n, double a, double b, int *seed, double r[] );
void timestamp ( );
void update ( int np, int nd, double pos[], double vel[], double f[],
  double acc[], double mass, double dt );

void compute_cuda ( int np, int nd, double *d_pos, double *d_vel, double mass,
  double *d_force, double *pot, double *kin );
void initialize_cuda ( int np, int nd, double *d_vel, double *d_acc );
void update_cuda ( int np, int nd, double *d_pos, double *d_vel, double *d_force,
  double *d_acc, double mass, double dt );
int compare_results ( double *arr1, double *arr2, int n );

__device__ double dist_device ( int nd, double *r1, double *r2, double dr[3] )
{
  double d = 0.0;
  for ( int i = 0; i < nd; i++ )
  {
    dr[i] = r1[i] - r2[i];
    d = d + dr[i] * dr[i];
  }
  return sqrt ( d );
}

__global__ void compute_kernel ( int np, int nd, double *pos, double *vel, double mass,
                                  double *f, double *pot_accum, double *kin_accum )
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if ( k >= np ) return;

  double PI2 = 3.141592653589793 / 2.0;
  double rij[3];
  double pe = 0.0;
  double ke = 0.0;

  for ( int i = 0; i < nd; i++ )
  {
    f[i + k * nd] = 0.0;
  }

  for ( int j = 0; j < np; j++ )
  {
    if ( k != j )
    {
      double d = dist_device ( nd, pos + k * nd, pos + j * nd, rij );

      double d2 = ( d < PI2 ) ? d : PI2;

      double sin_d2, cos_d2;
      sincos ( d2, &sin_d2, &cos_d2 );

      pe = pe + 0.5 * sin_d2 * sin_d2;

      double sin_2d2 = 2.0 * sin_d2 * cos_d2;

      for ( int i = 0; i < nd; i++ )
      {
        f[i + k * nd] = f[i + k * nd] - rij[i] * sin_2d2 / d;
      }
    }
  }

  for ( int i = 0; i < nd; i++ )
  {
    ke = ke + vel[i + k * nd] * vel[i + k * nd];
  }
  ke = ke * 0.5 * mass;

  atomicAdd ( pot_accum, pe );
  atomicAdd ( kin_accum, ke );
}

__global__ void zero_kernel ( int n, double *vel, double *acc )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if ( idx >= n ) return;

  vel[idx] = 0.0;
  acc[idx] = 0.0;
}

__global__ void update_kernel ( int n, double *pos, double *vel, double *f,
                                 double *acc, double rmass, double dt )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if ( idx >= n ) return;

  pos[idx] = pos[idx] + vel[idx] * dt + 0.5 * acc[idx] * dt * dt;
  vel[idx] = vel[idx] + 0.5 * dt * ( f[idx] * rmass + acc[idx] );
  acc[idx] = f[idx] * rmass;
}

void compute_cuda ( int np, int nd, double *d_pos, double *d_vel, double mass,
                     double *d_force, double *pot, double *kin )
{
  double *d_pot, *d_kin;
  cudaMalloc ( ( void ** ) &d_pot, sizeof ( double ) );
  cudaMalloc ( ( void ** ) &d_kin, sizeof ( double ) );
  cudaMemset ( d_pot, 0, sizeof ( double ) );
  cudaMemset ( d_kin, 0, sizeof ( double ) );

  int blockSize = COMPUTE_BLOCK_SIZE;
  int gridSize = ( np + blockSize - 1 ) / blockSize;

  compute_kernel<<<gridSize, blockSize>>> ( np, nd, d_pos, d_vel, mass, d_force, d_pot, d_kin );

  cudaError_t err = cudaGetLastError ();
  if ( err != cudaSuccess )
  {
    printf ( "compute_kernel launch failed: %s\n", cudaGetErrorString ( err ) );
  }

  cudaMemcpy ( pot, d_pot, sizeof ( double ), cudaMemcpyDeviceToHost );
  cudaMemcpy ( kin, d_kin, sizeof ( double ), cudaMemcpyDeviceToHost );

  cudaFree ( d_pot );
  cudaFree ( d_kin );
}

void initialize_cuda ( int np, int nd, double *d_vel, double *d_acc )
{
  int n = nd * np;
  int blockSize = NUM_OF_GPU_THREADS;
  int gridSize = ( n + blockSize - 1 ) / blockSize;

  zero_kernel<<<gridSize, blockSize>>> ( n, d_vel, d_acc );

  cudaError_t err = cudaGetLastError ();
  if ( err != cudaSuccess )
  {
    printf ( "zero_kernel launch failed: %s\n", cudaGetErrorString ( err ) );
  }
}

void update_cuda ( int np, int nd, double *d_pos, double *d_vel, double *d_force,
                    double *d_acc, double mass, double dt )
{
  int n = nd * np;
  double rmass = 1.0 / mass;
  int blockSize = NUM_OF_GPU_THREADS;
  int gridSize = ( n + blockSize - 1 ) / blockSize;

  update_kernel<<<gridSize, blockSize>>> ( n, d_pos, d_vel, d_force, d_acc, rmass, dt );

  cudaError_t err = cudaGetLastError ();
  if ( err != cudaSuccess )
  {
    printf ( "update_kernel launch failed: %s\n", cudaGetErrorString ( err ) );
  }
}

int main ( int argc, char *argv[] )
{
  double *acc;
  double ctime;
  double dt;
  double e0;
  double *force;
  double kinetic;
  double mass = 1.0;
  int nd;
  int np;
  double *pos;
  double potential;
  int step;
  int step_num;
  int step_print;
  int step_print_index;
  int step_print_num;
  double *vel;

  double time_start_seq;
  double time_elapsed_seq, time_elapsed_cuda;

  double *kinetic_energy_results;
  double *potentional_energy_results;
  double *relative_energy_error_results;

  double *cuda_kinetic_energy_results;
  double *cuda_potentional_energy_results;
  double *cuda_relative_energy_error_results;

  timestamp ( );
  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  CUDA version\n" );
  printf ( "  A molecular dynamics program.\n" );

  if ( 1 < argc )
  {
    nd = atoi ( argv[1] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter ND, the spatial dimension (2 or 3).\n" );
    scanf ( "%d", &nd );
  }

  if ( 2 < argc )
  {
    np = atoi ( argv[2] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter NP, the number of particles (500, for instance).\n" );
    scanf ( "%d", &np );
  }

  if ( 3 < argc )
  {
    step_num = atoi ( argv[3] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter STEP_NUM, the number of time steps (500 or 1000, for instance).\n" );
    scanf ( "%d", &step_num );
  }

  if ( 4 < argc )
  {
    dt = atof ( argv[4] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter DT, the size of the time step (0.1, for instance).\n" );
    scanf ( "%lf", &dt );
  }

  printf ( "\n" );
  printf ( "  ND, the spatial dimension, is %d\n", nd );
  printf ( "  NP, the number of particles in the simulation, is %d\n", np );
  printf ( "  STEP_NUM, the number of time steps, is %d\n", step_num );
  printf ( "  DT, the size of each time step, is %f\n", dt );

  acc = ( double * ) malloc ( nd * np * sizeof ( double ) );
  force = ( double * ) malloc ( nd * np * sizeof ( double ) );
  pos = ( double * ) malloc ( nd * np * sizeof ( double ) );
  vel = ( double * ) malloc ( nd * np * sizeof ( double ) );

  printf ( "\n" );
  printf ( "  At each step, we report the potential and kinetic energies.\n" );
  printf ( "  The sum of these energies should be a constant.\n" );
  printf ( "  As an accuracy check, we also print the relative error\n" );
  printf ( "  in the total energy.\n" );
  printf ( "\n" );
  printf ( "      Step      Potential       Kinetic        (P+K-E0)/E0\n" );
  printf ( "                Energy P        Energy K       Relative Energy Error\n" );
  printf ( "\n" );

  step_print = 0;
  step_print_index = 0;
  step_print_num = 10;

  kinetic_energy_results        = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  potentional_energy_results    = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  relative_energy_error_results = ( double * ) malloc ( step_print_num * sizeof ( double ) );

  cuda_kinetic_energy_results        = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  cuda_potentional_energy_results    = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  cuda_relative_energy_error_results = ( double * ) malloc ( step_print_num * sizeof ( double ) );

  ctime = cpu_time ( );

  // Sequential
  time_start_seq = cpu_time ( );
  for ( step = 0; step <= step_num; step++ )
  {
    if ( step == 0 )
    {
      initialize ( np, nd, pos, vel, acc );
    }
    else
    {
      update ( np, nd, pos, vel, force, acc, mass, dt );
    }

    compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );

    if ( step == 0 )
    {
      e0 = potential + kinetic;
    }

    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
       ( potential + kinetic - e0 ) / e0 );
      kinetic_energy_results[step_print_index] = kinetic;
      potentional_energy_results[step_print_index] = potential;
      relative_energy_error_results[step_print_index] = ( potential + kinetic - e0 ) / e0;
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }
  }
  time_elapsed_seq = cpu_time ( ) - time_start_seq;
  printf ( "Non parallel time:\t %f\n", time_elapsed_seq );

  //--------------------------------------------------------------------------------------------
  // CUDA
  step_print = 0;
  step_print_index = 0;

  double *d_pos, *d_vel, *d_force, *d_acc;
  cudaMalloc ( ( void ** ) &d_pos, nd * np * sizeof ( double ) );
  cudaMalloc ( ( void ** ) &d_vel, nd * np * sizeof ( double ) );
  cudaMalloc ( ( void ** ) &d_force, nd * np * sizeof ( double ) );
  cudaMalloc ( ( void ** ) &d_acc, nd * np * sizeof ( double ) );

  cudaEvent_t start, stop;
  cudaEventCreate ( &start );
  cudaEventCreate ( &stop );
  cudaEventRecord ( start );

  for ( step = 0; step <= step_num; step++ )
  {
    if ( step == 0 )
    {
      int seed = 123456789;
      r8mat_uniform_ab ( nd, np, 0.0, 10.0, &seed, pos );
      cudaMemcpy ( d_pos, pos, nd * np * sizeof ( double ), cudaMemcpyHostToDevice );
      initialize_cuda ( np, nd, d_vel, d_acc );
    }
    else
    {
      update_cuda ( np, nd, d_pos, d_vel, d_force, d_acc, mass, dt );
    }

    compute_cuda ( np, nd, d_pos, d_vel, mass, d_force, &potential, &kinetic );

    if ( step == 0 )
    {
      e0 = potential + kinetic;
    }

    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
       ( potential + kinetic - e0 ) / e0 );
      cuda_kinetic_energy_results[step_print_index] = kinetic;
      cuda_potentional_energy_results[step_print_index] = potential;
      cuda_relative_energy_error_results[step_print_index] = ( potential + kinetic - e0 ) / e0;
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }
  }

  cudaEventRecord ( stop );
  cudaEventSynchronize ( stop );
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime ( &elapsed_ms, start, stop );
  time_elapsed_cuda = ( double ) elapsed_ms / 1000.0;
  cudaEventDestroy ( start );
  cudaEventDestroy ( stop );

  cudaFree ( d_pos );
  cudaFree ( d_vel );
  cudaFree ( d_force );
  cudaFree ( d_acc );

  printf ( "CUDA time:\t %f\n", time_elapsed_cuda );
  printf ( "Time for CUDA is %.2f times faster\n", time_elapsed_seq / time_elapsed_cuda );

  ctime = cpu_time ( ) - ctime;
  printf ( "\n" );
  printf ( "  Elapsed cpu time: %f seconds.\n", ctime );

  if ( compare_results ( kinetic_energy_results, cuda_kinetic_energy_results, step_print_num ) &&
       compare_results ( potentional_energy_results, cuda_potentional_energy_results, step_print_num ) &&
       compare_results ( relative_energy_error_results, cuda_relative_energy_error_results, step_print_num ) )
  {
    printf ( "Test PASSED\n" );
  }
  else
  {
    printf ( "Test FAILED\n" );
  }

  free ( acc );
  free ( force );
  free ( pos );
  free ( vel );

  free ( kinetic_energy_results );
  free ( potentional_energy_results );
  free ( relative_energy_error_results );

  free ( cuda_kinetic_energy_results );
  free ( cuda_potentional_energy_results );
  free ( cuda_relative_energy_error_results );

  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}

void compute ( int np, int nd, double pos[], double vel[], double mass,
  double f[], double *pot, double *kin )
{
  double d;
  double d2;
  int i;
  int j;
  int k;
  double ke;
  double pe;
  double PI2 = 3.141592653589793 / 2.0;
  double rij[3];

  pe = 0.0;
  ke = 0.0;

  for ( k = 0; k < np; k++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      f[i+k*nd] = 0.0;
    }

    for ( j = 0; j < np; j++ )
    {
      if ( k != j )
      {
        d = dist ( nd, pos+k*nd, pos+j*nd, rij );

        if ( d < PI2 )
        {
          d2 = d;
        }
        else
        {
          d2 = PI2;
        }

        pe = pe + 0.5 * pow ( sin ( d2 ), 2 );

        for ( i = 0; i < nd; i++ )
        {
          f[i+k*nd] = f[i+k*nd] - rij[i] * sin ( 2.0 * d2 ) / d;
        }
      }
    }

    for ( i = 0; i < nd; i++ )
    {
      ke = ke + vel[i+k*nd] * vel[i+k*nd];
    }
  }

  ke = ke * 0.5 * mass;

  *pot = pe;
  *kin = ke;

  return;
}

double cpu_time ( )
{
  double value;

  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;

  return value;
}

double dist ( int nd, double r1[], double r2[], double dr[] )
{
  double d;
  int i;

  d = 0.0;
  for ( i = 0; i < nd; i++ )
  {
    dr[i] = r1[i] - r2[i];
    d = d + dr[i] * dr[i];
  }
  d = sqrt ( d );

  return d;
}

void initialize ( int np, int nd, double pos[], double vel[], double acc[] )
{
  int i;
  int j;
  int seed;

  seed = 123456789;
  r8mat_uniform_ab ( nd, np, 0.0, 10.0, &seed, pos );

  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      vel[i+j*nd] = 0.0;
    }
  }

  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      acc[i+j*nd] = 0.0;
    }
  }

  return;
}

void r8mat_uniform_ab ( int m, int n, double a, double b, int *seed, double r[] )
{
  int i;
  const int i4_huge = 2147483647;
  int j;
  int k;

  if ( *seed == 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8MAT_UNIFORM_AB - Fatal error!\n" );
    fprintf ( stderr, "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      k = *seed / 127773;

      *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

      if ( *seed < 0 )
      {
        *seed = *seed + i4_huge;
      }
      r[i+j*m] = a + ( b - a ) * ( double ) ( *seed ) * 4.656612875E-10;
    }
  }

  return;
}

void timestamp ( )
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

void update ( int np, int nd, double pos[], double vel[], double f[],
  double acc[], double mass, double dt )
{
  int i;
  int j;
  double rmass;

  rmass = 1.0 / mass;

  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      pos[i+j*nd] = pos[i+j*nd] + vel[i+j*nd] * dt + 0.5 * acc[i+j*nd] * dt * dt;
      vel[i+j*nd] = vel[i+j*nd] + 0.5 * dt * ( f[i+j*nd] * rmass + acc[i+j*nd] );
      acc[i+j*nd] = f[i+j*nd] * rmass;
    }
  }

  return;
}

int compare_results ( double *arr1, double *arr2, int n )
{
  for ( int i = 0; i < n; i++ )
  {
    double scale = fabs ( arr1[i] );
    double diff = fabs ( arr1[i] - arr2[i] );

    if ( scale > 1.0 )
    {
      if ( diff / scale > ACCURACY )
      {
        return 0;
      }
    }
    else
    {
      if ( diff > ACCURACY )
      {
        return 0;
      }
    }
  }
  return 1;
}