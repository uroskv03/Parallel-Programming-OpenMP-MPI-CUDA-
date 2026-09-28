#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <stddef.h>

#define ACCURACY 0.01

#define MASTER_RANK 0

typedef struct {
    double err;
    int inside;
} result_t;

void compare_results(double err, double err_parallel);

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

// print na stdout upotrebiti u validaciji paralelnog resenja
int main(int argc, char **argv)
{
  int rank, size;

  if(argc != 2) {
    printf("Provide correct arguments");
    return 1;
  }
    
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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

  double time_start;
  double time_elapsed_seq, time_elapsed_par;

  int N = atoi(argv[1]);

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
  if (rank == MASTER_RANK) {
    timestamp();
    
    printf("A = %f\n", a);
    printf("B = %f\n", b);
    printf("C = %f\n", c);
    printf("N = %d\n", N);
    printf("H = %6.4f\n", h);
    time_start = MPI_Wtime();
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
            // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
            //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

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

          // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
          //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);
        }
      }
    }
    err = sqrt(err / (double)(n_inside));
    time_elapsed_seq = MPI_Wtime() - time_start;
    printf("Non parallel time:\t %f\n", time_elapsed_seq);
    printf("\n\nRMS absolute error in solution = %e\n", err);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // Parallel
  int local_seed = 123456789 + rank * 1000;
  double err_parallel = 0.0;
  n_inside = 0;
  time_start = MPI_Wtime();

  for (i = rank+1; i <= ni; i+=size)
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
          // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
          //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

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
            ut = r8_uniform_01(&local_seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&local_seed) - 0.5;
              if (us < 0.0)
                dx = -stepsz;
              else
                dx = stepsz;
            }
            else
              dx = 0.0;

            ut = r8_uniform_01(&local_seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&local_seed) - 0.5;
              if (us < 0.0)
                dy = -stepsz;
              else
                dy = stepsz;
            }
            else
              dy = 0.0;

            ut = r8_uniform_01(&local_seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&local_seed) - 0.5;
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

        err_parallel = err_parallel + pow(w_exact - wt, 2);

        // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
        //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);
      }
    }
  }

  MPI_Win win;
  result_t result;

  if (rank == MASTER_RANK) {
    result.err = err_parallel;
    result.inside = n_inside;
  }

  MPI_Win_create(
    (rank == MASTER_RANK) ? &result : NULL,
    (rank == MASTER_RANK) ? sizeof(result_t) : 0,
    1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  MPI_Aint err_disp = offsetof(result_t, err);
  MPI_Aint inside_disp = offsetof(result_t, inside);

  MPI_Win_fence(0, win);

  if (rank != MASTER_RANK) {
    MPI_Accumulate( &err_parallel, 1, MPI_DOUBLE, MASTER_RANK, err_disp, 1, MPI_DOUBLE, MPI_SUM, win);
    MPI_Accumulate( &n_inside, 1, MPI_INT, MASTER_RANK, inside_disp, 1, MPI_INT, MPI_SUM, win);
  }

  MPI_Win_fence(0, win);

  if(rank == MASTER_RANK){
    err_parallel = sqrt(result.err / (double)result.inside);
    time_elapsed_par = MPI_Wtime() - time_start;
    printf("Parallel time:\t %f\n", time_elapsed_par);

    printf("\n\nRMS absolute error in solution = %e\n", err_parallel);
    printf("Seq err: %f \t Par err: %f\n",err,err_parallel);

    compare_results(err, err_parallel);

    printf("Time for Parallel is %.2f times faster\n", time_elapsed_seq/time_elapsed_par);
    timestamp();

    printf("---------------------------\n");
  }
  MPI_Win_free(&win);

  MPI_Finalize();
  return 0;
}

void compare_results(double err, double err_parallel)
{
  if(fabs(err - err_parallel) < ACCURACY)
  {
    printf("Test PASSED\n");
  }
  else
  {
    printf("Test FAILED\n");
  }
}