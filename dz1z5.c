# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <math.h>
#include <omp.h>


#define ACCURACY 0.01

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


void compute_optimized ( int np, int nd, double pos[], double vel[], double mass, double f[], double *pot, double *kin );
void initialize_optimized ( int np, int nd, double pos[], double vel[], double acc[] );
void update_optimized ( int np, int nd, double pos[], double vel[], double f[], 
  double acc[], double mass, double dt );
int compare_results(double* arr1, double* arr2, int n);

int main ( int argc, char *argv[] )
{
  double *acc;
  double ctime;
  double dt;
  double e0;
  double *force;
  int i;
  int id;
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

  double time_start;
  double time_elapsed_seq, time_elapsed_par;

  double *kinetic_energy_results;
  double *potentional_energy_results;
  double *relative_energy_error_results;

  double *parallel_kinetic_energy_results;
  double *parallel_potentional_energy_results;
  double *parallel_relative_energy_error_results;

  timestamp ( );
  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  C version\n" );
  printf ( "  A molecular dynamics program.\n" );
/*
  Get the spatial dimension.
*/
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
//
//  Get the number of particles.
//
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
//
//  Get the number of time steps.
//
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
//
//  Get the time steps.
//
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
/*
  Report.
*/
  printf ( "\n" );
  printf ( "  ND, the spatial dimension, is %d\n", nd );
  printf ( "  NP, the number of particles in the simulation, is %d\n", np );
  printf ( "  STEP_NUM, the number of time steps, is %d\n", step_num );
  printf ( "  DT, the size of each time step, is %f\n", dt );
/*
  Allocate memory.
*/
  // Sequential
  acc = ( double * ) malloc ( nd * np * sizeof ( double ) );
  force = ( double * ) malloc ( nd * np * sizeof ( double ) );
  pos = ( double * ) malloc ( nd * np * sizeof ( double ) );
  vel = ( double * ) malloc ( nd * np * sizeof ( double ) );

/*
  This is the main time stepping loop:
    Compute forces and energies,
    Update positions, velocities, accelerations.
*/
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
  


  kinetic_energy_results         = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  potentional_energy_results     = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  relative_energy_error_results  = ( double * ) malloc ( step_print_num * sizeof ( double ) );

  parallel_kinetic_energy_results         = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  parallel_potentional_energy_results     = ( double * ) malloc ( step_print_num * sizeof ( double ) );
  parallel_relative_energy_error_results  = ( double * ) malloc ( step_print_num * sizeof ( double ) );

  ctime = cpu_time ( );

time_start = omp_get_wtime();
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
  time_elapsed_seq = omp_get_wtime() - time_start;
  printf("Non parallel time:\t %f\n", time_elapsed_seq);

//--------------------------------------------------------------------------------------------
  step_print = 0;
  step_print_index = 0;

  time_start = omp_get_wtime();
  for ( step = 0; step <= step_num; step++ )
  {
    if ( step == 0 )
    {
      initialize_optimized ( np, nd, pos, vel, acc );
    }
    else
    {
      update_optimized ( np, nd, pos, vel, force, acc, mass, dt );
    }

    compute_optimized ( np, nd, pos, vel, mass, force, &potential, &kinetic );

    if ( step == 0 )
    {
      e0 = potential + kinetic;
    }

    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
       ( potential + kinetic - e0 ) / e0 );
      parallel_kinetic_energy_results[step_print_index] = kinetic;
      parallel_potentional_energy_results[step_print_index] = potential;
      parallel_relative_energy_error_results[step_print_index] = ( potential + kinetic - e0 ) / e0;
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }

  }
  time_elapsed_par = omp_get_wtime() - time_start;
  printf("Parallel time:\t %f\n", time_elapsed_par);
/*
  Report timing.
*/
  printf("Time for Parallel is %.2f times faster\n", time_elapsed_seq/time_elapsed_par);

  ctime = cpu_time ( ) - ctime;
  printf ( "\n" );
  printf ( "  Elapsed cpu time: %f seconds.\n", ctime );

  if(compare_results(kinetic_energy_results,parallel_kinetic_energy_results, step_print_num) && 
     compare_results(potentional_energy_results,parallel_potentional_energy_results, step_print_num) && 
     compare_results(relative_energy_error_results,parallel_relative_energy_error_results, step_print_num))
  {
    printf("Test PASSED\n");
  }
  else
  {
    printf("Test FAILED\n");
  }

/*
  Free memory.
*/
  free ( acc );
  free ( force );
  free ( pos );
  free ( vel );

  free(kinetic_energy_results);
  free(potentional_energy_results);
  free(relative_energy_error_results);

  free(parallel_kinetic_energy_results);
  free(parallel_potentional_energy_results);
  free(parallel_relative_energy_error_results);
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}

void compute_optimized ( int np, int nd, double pos[], double vel[], double mass, 
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


  pe = 0.0;
  ke = 0.0;

#pragma omp parallel for      \
        private(k,i,j,d,d2)   \
        reduction(+:pe,ke)
  for ( k = 0; k < np; k++ )
  {
    double rij[3];
/*
  Compute the potential energy and forces.
*/
    for ( i = 0; i < nd; i++ )
    {
      f[i+k*nd] = 0.0;
    }

    for ( j = 0; j < np; j++ )
    {
      if ( k != j )
      {
        d = dist ( nd, pos+k*nd, pos+j*nd, rij );
        
/*
  Attribute half of the potential energy to particle J.
*/
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
/*
  Compute the kinetic energy.
*/
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
/******************************************************************************/

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
/******************************************************************************/

void initialize_optimized ( int np, int nd, double pos[], double vel[], double acc[] )
{
  int i;
  int j;
  int seed;
/*
  Set positions.
*/
  seed = 123456789;
  r8mat_uniform_ab ( nd, np, 0.0, 10.0, &seed, pos );
/*
  Set velocities. 
  Set accelerations.
*/
#pragma omp parallel for private(j,i)
  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      vel[i+j*nd] = 0.0;
      acc[i+j*nd] = 0.0;
    }
  }

  return;
}

void update_optimized ( int np, int nd, double pos[], double vel[], double f[], 
  double acc[], double mass, double dt )
{
  int i;
  int j;
  double rmass;

  rmass = 1.0 / mass;
#pragma omp parallel for default(none) \
            private(j,i)   \
            shared(pos, acc, np, nd, rmass, f, vel, dt)
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
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

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
/*
  Compute the potential energy and forces.
*/
    for ( i = 0; i < nd; i++ )
    {
      f[i+k*nd] = 0.0;
    }

    for ( j = 0; j < np; j++ )
    {
      if ( k != j )
      {
        d = dist ( nd, pos+k*nd, pos+j*nd, rij );
/*
  Attribute half of the potential energy to particle J.
*/
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
/*
  Compute the kinetic energy.
*/
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


void initialize ( int np, int nd, double pos[], double vel[], double acc[] )
{
  int i;
  int j;
  int seed;
/*
  Set positions.
*/
  seed = 123456789;
  r8mat_uniform_ab ( nd, np, 0.0, 10.0, &seed, pos );
/*
  Set velocities.
*/
  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      vel[i+j*nd] = 0.0;
    }
  }
/*
  Set accelerations.
*/
  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      acc[i+j*nd] = 0.0;
    }
  }

  return;
}



int compare_results(double* arr1, double* arr2, int n)
{
  for(int i = 0; i < n; i++)
  {
    if(fabs(arr1[i] - arr2[i]) > ACCURACY)
    {
      return 0;
    }
  }
  return 1;
}