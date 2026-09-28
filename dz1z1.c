# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>
# include <omp.h>

# define DEFAULT_H 1000
# define DEFAULT_W 1000
# define DEFAULT_CNT 200
# define DEFAULT_FILENAME "julia"
#define PARALLEL_FILENAME "julia_parallel"

int main (int argc, char *argv[]);
unsigned char *julia_set ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt );
void tga_write ( int w, int h, unsigned char rgb[], char *filename );
void timestamp ( );


unsigned char *julia_set_parallel ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
void update_filename(int w, int h, int cnt, char buffer[256], char *filename);
void compare_results(int w, int h, unsigned char rgb[], unsigned char rgb_parallel[]);

int main (int argc, char *argv[] )  {
  int h = DEFAULT_H;
  int w = DEFAULT_W;
  int cnt = DEFAULT_CNT;
  char filename[256] = DEFAULT_FILENAME; 
  char parallel_filename[256] = PARALLEL_FILENAME;
  char buffer[256];
  unsigned char *rgb, *rgb_parallel;
  double time_start;
  double time_elapsed_seq, time_elapsed_par;
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
  time_start = omp_get_wtime();
  rgb = julia_set ( w, h, cnt, xl, xr, yb, yt );
  time_elapsed_seq = omp_get_wtime() - time_start;

  printf("Non parallel time:\t %f\n", time_elapsed_seq);

  // Parallel
  time_start = omp_get_wtime();
  rgb_parallel = julia_set_parallel( w, h, cnt, xl, xr, yb, yt );
  time_elapsed_par = omp_get_wtime() - time_start;

  printf("Parallel time:\t %f\n", time_elapsed_par);

  printf("Time for Parallel is %.2f times faster\n", time_elapsed_seq/time_elapsed_par);

  update_filename(w, h, cnt, buffer, filename);
  update_filename(w, h, cnt, buffer, parallel_filename);

  tga_write ( w, h, rgb, filename );
  tga_write ( w, h, rgb_parallel, parallel_filename);

  compare_results(w, h, rgb, rgb_parallel);

  free(rgb);
  free(rgb_parallel);

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
  int nextRow = 0;
  rgb = ( unsigned char * ) malloc ( w * h * 3 * sizeof ( unsigned char ) );

  while(1)
  {
    j = nextRow++;
    if (j >= h) break;
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

unsigned char *julia_set_parallel ( int w, int h, int cnt, float xl, float xr, float yb, float yt )
{
  int i;
  int j;
  int juliaValue;
  int k;
  unsigned char *rgb;

  // Parallel
  int id, nthreads, start, end, chunk;

  rgb = ( unsigned char * ) malloc ( w * h * 3 * sizeof ( unsigned char ) );
  int nextRow=0;
#pragma omp parallel default(none) \
        shared(w, h, cnt, xl, xr, yb, yt, rgb, nextRow) \
        private(i, j, k, juliaValue)
{
    
  while(1)
  {
#pragma omp atomic capture
    j = nextRow++;
    if (j >= h) break;
    for ( i = 0; i < w; i++ )
    {
    juliaValue = julia ( w, h, xl, xr, yb, yt, i, j, cnt );

    k = 3 * ( j * w + i );

    rgb[k]   = 255 * ( 1 - juliaValue );
    rgb[k+1] = 255 * ( 1 - juliaValue );
    rgb[k+2] = 255;
    }
  }
}

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

  /*printf ( "\n" );
  printf ( "TGA_WRITE:\n" );
  printf ( "  Graphics data saved as '%s'\n", filename );
  */
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

void compare_results(int w, int h, unsigned char rgb[], unsigned char rgb_parallel[])
{
  for(int i = 0; i < 3*w*h; i++)
  {
    if(rgb[i] != rgb_parallel[i])
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