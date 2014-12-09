#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "mt19937ar.h"

#define NUM_SPINS L*L*L
#define L 32	//has to be a multiple of 2*S_B
#define S_B 2	//Number of spins in block side
#define S_T 2	//Spins per thread (in x-direction), only 2 is supported yet
#define NS_PER_S 1000000000ULL

__shared__ int s_s[S_B+2][S_B+2][S_B+2];	//2 ghost cells
__shared__ float s_J[S_B][S_B][S_B+1];	//just one GC
__shared__ float s_K[S_B][S_B+1][S_B];	//just one GC
__shared__ float s_I[S_B+1][S_B][S_B];	//just one GC


/* function to initialize shared memory for each block of threads */
__device__ void setSharedCell(unsigned x0, unsigned y0, unsigned z0, int *s, float *I, float *J, float *K)
{
  unsigned x,y,z,xyz,k;

  for (k = 0; k < S_T; k++)
  {
    x = x0 + k;
    y = y0;
    z = z0;

    /************* GHOST CELLS *************/
    if (k == 0 && threadIdx.x == 0)	//First column thread
    {
      xyz = (x-1+L) % L + y * L + z * L*L;
      s_s[threadIdx.z + 1][threadIdx.y + 1][0] = s[xyz];
      s_J[threadIdx.z][threadIdx.y][0] = J[xyz];
    }
    else if (k == 1 && threadIdx.x == blockDim.x-1)	//Last column thread
    {
      s_s[threadIdx.z + 1][threadIdx.y + 1][S_B+1] = s[(x+1)%L + y * L + z * L*L];
    }

    if (threadIdx.y == 0)	//First row thread
    {
      xyz = x + ((y-1+L) % L) * L + z * L*L;
      s_s[threadIdx.z + 1][0][threadIdx.x * S_T + 1 + k] = s[xyz];
      s_K[threadIdx.z][0][threadIdx.x * S_T + k] = K[xyz];
    }
    else if (threadIdx.y == blockDim.y-1)	//Last row thread
    {
      s_s[threadIdx.z + 1][S_B+1][threadIdx.x * S_T + 1 + k] = s[x + ((y+1)%L) * L + z * L*L];
    }

    if (threadIdx.z == 0)	//First row thread
    {
      xyz = x + y * L + ((z-1+L) % L) * L*L;
      s_s[0][threadIdx.y + 1][threadIdx.x * S_T + 1 + k] = s[xyz];
      s_I[0][threadIdx.y][threadIdx.x * S_T + k] = I[xyz];
    }
    else if (threadIdx.z == blockDim.z-1)	//Last row thread
    {
      s_s[S_B+1][threadIdx.y + 1][threadIdx.x * S_T + 1 + k] = s[x + y * L + ((z+1)%L) * L*L];
    }

    /************* INNER CELLS *************/
    xyz = x + y * L + z * L*L;

    s_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x * S_T + 1 + k] = s[xyz];

    s_J[threadIdx.z][threadIdx.y][threadIdx.x * S_T + 1 + k] = J[xyz];
    s_K[threadIdx.z][threadIdx.y + 1][threadIdx.x * S_T + k] = K[xyz];
    s_I[threadIdx.z + 1][threadIdx.y][threadIdx.x * S_T + k] = I[xyz];
  }
}


/* device kernel to flip spins */
__global__ void minimize(int *s, float *I, float *J, float *K, int bo, int* flipped)
{
  unsigned x,x0, y,y0, z,z0, k;	//Indices
  __shared__ bool s_flipped;
  float de;

  s_flipped = false;

  if ((blockIdx.y + blockIdx.z) % 2 == 0)
    x0 = blockDim.x * S_T * ( 2 * blockIdx.x + bo) + threadIdx.x * S_T;	//Columns
  else
    x0 = blockDim.x * S_T * ( 2 * blockIdx.x + 1-bo) + threadIdx.x * S_T;	//Columns

  y0  = blockDim.y * blockIdx.y  + threadIdx.y;	//Rows
  z0  = blockDim.z * blockIdx.z  + threadIdx.z;	//Depth

  setSharedCell(x0, y0, z0, s, I, J, K);

  __syncthreads();	//wait until shared memory is initialized

  /************* ACTUAL CALCULATIONS *************/
  for (k = 0; k < 2; k++)
  {
    z = threadIdx.z + 1;
    y = threadIdx.y + 1;

    if ((y0+z0) % 2 == 0)
      x = threadIdx.x * S_T + 1 + k;
    else
      x = threadIdx.x * S_T + 2 - k;

    de = s_s[z][y][x] *
    (
       s_s[z][y][x-1] * s_J[z-1][y-1][x-1] +
       s_s[z][y][x+1] * s_J[z-1][y-1][x]   +

       s_s[z][y-1][x] * s_K[z-1][y-1][x-1] +
       s_s[z][y+1][x] * s_K[z-1][y][x-1]   +

       s_s[z-1][y][x] * s_I[z-1][y-1][x-1] +
       s_s[z+1][y][x] * s_I[z][y-1][x-1]
    );

    if (de > 0)
    {
      s_s[z][y][x] = -s_s[z][y][x];
      if (! s_flipped)	//if no spin flip in this block has been accepted yet set global variable
      {
        s_flipped = true;
        flipped[0] = 1;
      }
    }

    if ((y0+z0) % 2 == 0)
      s[x0+k + L*y0 + L*L*z0] = s_s[z][y][x];
    else
      s[x0+1-k + L*y0 + L*L*z0] = s_s[z][y][x];

    __syncthreads();	//wait for all threads to finish
  }
}


/* device kernel to calculate the energy of the system */
__global__ void calcEnergy(int *s, float *I, float *J, float *K, float *E)
{
  unsigned x,x0, y,y0, z,z0, k;	//Indices
  float e = 0.0;

  x0 = blockDim.x * S_T * blockIdx.x + threadIdx.x * S_T;	//Columns
  y0  = blockDim.y * blockIdx.y  + threadIdx.y;	//Rows
  z0  = blockDim.z * blockIdx.z  + threadIdx.z;	//Depth

  setSharedCell(x0, y0, z0, s, I, J, K);

  __syncthreads();	//wait until shared memory is initialized

  /************* ACTUAL CALCULATIONS *************/
  for (k = 0; k < 2; k++)
  {
    z = threadIdx.z + 1;
    y = threadIdx.y + 1;

    if ((y+z) % 2 == 1)
        x = threadIdx.x * S_T + 1 + k;
    else
        x = threadIdx.x * S_T + 2 - k;

    e += s_s[z][y][x] *
    (
       s_s[z][y][x-1] * s_J[z-1][y-1][x-1] +
       s_s[z][y][x+1] * s_J[z-1][y-1][x]   +

       s_s[z][y-1][x] * s_K[z-1][y-1][x-1] +
       s_s[z][y+1][x] * s_K[z-1][y][x-1]   +

       s_s[z-1][y][x] * s_I[z-1][y-1][x-1] +
       s_s[z+1][y][x] * s_I[z][y-1][x-1]
    ) * 0.5;
  }
  E[(x0 + y0*L + z0*L*L)/S_T] = e;
}


/* Generates a random Gaussian distributed number with mean zero and variance
   one using the Box-Muller method. genrand_real3 creates a random number in
   the intervall (0,1) */
float rand_gauss(void)
{
  return (float) ( sqrt(-2 * log(genrand_real3())) * cos(2 * M_PI * genrand_real3()) );
}


/* Return true if an error occured in CUDA */
bool cudaErr(cudaError_t status)
{
  if (status == cudaSuccess)
    return(false);
  else
  {
    fprintf(stderr, "CUDA ERROR: %s\n\n", cudaGetErrorString(status));
    return(true);
  }
}


void usage(char *cmd)
{
  fprintf(stderr, "usage:\n\t%s gpu_id minN maxN Ncouplings\n\n", cmd);
  exit(1);
}


int main(int argc, char *argv[])
{
  /* Declare and initialize variables */
  unsigned i, k, Nquenches, seed;
  unsigned long long tdiff, counter = 0;
  int len, j, block_offset, cuda_dev, num_dev, nc, minN, maxN, N, Nmaxsteps, Ncouplings, h_flipped, thread_num, req_threads;
  int *h_s = NULL, *d_s = NULL, *d_flipped = NULL;
  char buffer[500], buf[2];
  char *filename = NULL;
  FILE *f = NULL;
  float *h_J = NULL, *h_Kpbc = NULL, *h_Kapbc = NULL, *h_I = NULL, *h_E = NULL;
  float *d_J = NULL, *d_Kpbc = NULL, *d_Kapbc = NULL, *d_I = NULL, *d_E = NULL;
  float Epbc, Eapbc, E;
  struct timespec starttime, endtime;

  cudaError_t status;

  const dim3 num_blocks1(L/S_B/2, L/S_B,  L/S_B);
  const dim3 num_blocks2(L/S_B,   L/S_B,  L/S_B);
  const dim3 block_size(S_B/S_T,  S_B,    S_B);

  Nmaxsteps = 1000;  //max. number of lattice sweeps (if we don't break the loop because no more spins get flipped)

  if (argc != 5)
    usage(argv[0]);

  req_threads = strlen(argv[1]);	//get number of requested threads/GPUs (first argument can be 01 to request GPUs 0 and 1)
  minN = abs(atoi(argv[2]));
  maxN = abs(atoi(argv[3]));
  Ncouplings = abs(atoi(argv[4]));

  cudaGetDeviceCount(&num_dev);

  if (minN > maxN)
    usage(argv[0]);

  if (num_dev == 0)
  {
    fprintf(stderr, "ERROR: no GPU detected.\n");
    exit(1);
  }
  else if (num_dev < req_threads)
  {
    fprintf(stderr, "ERROR: %d threads requested but only %d GPU(s) detected.\n", req_threads, num_dev);
    exit(1);
  }

#pragma omp parallel num_threads(req_threads) private(status, cuda_dev, thread_num, buf, h_s, d_s, h_E, d_E, d_J, d_Kpbc, d_Kapbc, d_I, d_flipped, h_flipped, nc, N, i, j, k, E, block_offset) shared(counter, Nquenches, Ncouplings, Nmaxsteps, minN, maxN, f, filename, len, buffer, h_J, h_Kpbc, h_Kapbc, h_I, Epbc, Eapbc, seed, starttime, endtime)
{
  thread_num = omp_get_thread_num();
  buf[0] = argv[1][thread_num];
  buf[1] = 0;	//terminate string
  cuda_dev = atoi(buf);
  status = cudaSetDevice(cuda_dev);	//request respective GPU in every thread
  if ( cudaErr(status) )
    exit(1);


  /**************  MEMORY ALLOCATION  **************/
  cudaMalloc((void**)&d_flipped, sizeof(int));

  h_s = (int*) malloc(NUM_SPINS * sizeof(int));
  cudaMalloc((void**)&d_s, NUM_SPINS * sizeof(int));

  h_E = (float*) malloc(NUM_SPINS / 2 * sizeof(float));
  cudaMalloc((void**)&d_E, NUM_SPINS / 2 * sizeof(float));

  /* for shared arrays only one thread should execute malloc */
  #pragma omp master
  {
    h_J = (float*) malloc(NUM_SPINS * sizeof(float));
    h_Kpbc = (float*) malloc(NUM_SPINS * sizeof(float));
    h_Kapbc = (float*) malloc(NUM_SPINS * sizeof(float));
    h_I = (float*) malloc(NUM_SPINS * sizeof(float));
  }
  cudaMalloc((void**)&d_J, NUM_SPINS * sizeof(float));
  cudaMalloc((void**)&d_Kpbc, NUM_SPINS * sizeof(float));
  cudaMalloc((void**)&d_Kapbc, NUM_SPINS * sizeof(float));
  cudaMalloc((void**)&d_I, NUM_SPINS * sizeof(float));

  #pragma omp barrier

  if (h_s == NULL || h_I == NULL || h_J == NULL || h_Kpbc == NULL || h_Kapbc == NULL || h_E == NULL )
  {
    fprintf(stderr, "ERROR: Cannot allocate memory.\n");
    exit(1);
  }

  if ( cudaErr(cudaGetLastError()) )
    exit(1);

  #pragma omp master
  clock_gettime(CLOCK_REALTIME, &starttime);

  /***************************  READY, SET, GO!  ****************************/
  for(N = minN; N <= maxN; N++)
  {
    if (N != 2 && N != 3 && N != 4 && N != 6 && N != 8 && N != 10 && N != 15)
      continue;

    /* shared variables only need to be set by one thread */
    #pragma omp master
    {
      seed = time(NULL);

      len = sprintf(buffer, "data3dL%dN%d", L, N)+1;
      filename = (char*) malloc(len*sizeof(char));
      strncpy(filename, buffer, len);
      f = fopen(filename, "a");
      if (f == NULL)
      {
        fprintf(stderr, "Error opening file!\n");
        exit(1);
      }

      Nquenches = pow(10,N);

      printf("Start writing N%d\n", N);
      fprintf(f, "# seed: %u, Ncouplings: %d\n", seed, Ncouplings);
      fflush(f);
    }

    #pragma omp barrier

    srand(seed + thread_num);	//each thread uses different seed
    init_genrand(seed + thread_num);

    for (nc = 0; nc < Ncouplings; nc++)
    {
      #pragma omp master
      {
        Epbc = 9999999.0;
        Eapbc = 9999999.0;
      }

      /* initialize random couplings using all threads */
      #pragma omp for
      for (k = 0; k < NUM_SPINS; k++)
      {
        h_J[k] = rand_gauss();
        h_Kpbc[k] = rand_gauss();
        h_Kapbc[k] = h_Kpbc[k];
        h_I[k] = rand_gauss();
      }


      /* anti-periodic BC, intoducing domain wall next to the first plane */
      #pragma omp for
      for (i = 0; i < L*L; i++)
        h_Kapbc[i] = -h_Kpbc[i];

      cudaMemcpy(d_J, h_J, sizeof(float) * NUM_SPINS, cudaMemcpyHostToDevice);
      cudaMemcpy(d_Kpbc, h_Kpbc, sizeof(float) * NUM_SPINS, cudaMemcpyHostToDevice);
      cudaMemcpy(d_Kapbc, h_Kapbc, sizeof(float) * NUM_SPINS, cudaMemcpyHostToDevice);
      cudaMemcpy(d_I, h_I, sizeof(float) * NUM_SPINS, cudaMemcpyHostToDevice);

      /* this is the main loop that is split amongst the requested GPUs */
      #pragma omp for
      for (i = 0; i < Nquenches; i++)
      {
        /* initialize random spins */
        for (j = 0; j < NUM_SPINS; j++)
        {
          if(genrand_real1() > 0.5)
            h_s[j] = 1;
          else
            h_s[j] = -1;
        }

        cudaMemcpy(d_s, h_s, sizeof(int) * NUM_SPINS, cudaMemcpyHostToDevice);

	for (j = 0; j < Nmaxsteps; j++)
        {
	  h_flipped = 0;
          cudaMemcpy(d_flipped, &h_flipped, sizeof(int), cudaMemcpyHostToDevice);

          for (block_offset = 0; block_offset < 2; block_offset++) //Switching block "color"
          {
            minimize<<< num_blocks1, block_size >>>(d_s, d_I, d_J, d_Kpbc, block_offset, d_flipped);
            cudaDeviceSynchronize();
          }

          cudaMemcpy(&h_flipped, d_flipped, sizeof(int), cudaMemcpyDeviceToHost);
          /* if no spin flip was accepted break the loop */
          if (h_flipped == 0)
          {
            #pragma omp atomic
            counter += j;
            break;
          }
        }

        calcEnergy<<< num_blocks2, block_size >>>(d_s, d_I, d_J, d_Kpbc, d_E);
        cudaMemcpy(h_E, d_E, sizeof(float) * NUM_SPINS / 2, cudaMemcpyDeviceToHost);

        E = 0.0;
	for (j = 0; j < NUM_SPINS / 2; j++)
        {
          E += h_E[j];
        }

        /* omp critical to avoid race condition */
        #pragma omp critical
        {
          if (E < Epbc)
            Epbc = E;
        }


        /************************  Switch to Anti-PBC  ************************/

        cudaMemcpy(d_s, h_s, sizeof(int) * NUM_SPINS, cudaMemcpyHostToDevice);

        for (j = 0; j < Nmaxsteps; j++)
	{
	  h_flipped = 0;
          cudaMemcpy(d_flipped, &h_flipped, sizeof(int), cudaMemcpyHostToDevice);

          for (block_offset = 0; block_offset < 2; block_offset++) //Switching block "color"
          {
            minimize<<< num_blocks1, block_size >>>(d_s, d_I, d_J, d_Kapbc, block_offset, d_flipped);
            cudaDeviceSynchronize();
          }

          cudaMemcpy(&h_flipped, d_flipped, sizeof(int), cudaMemcpyDeviceToHost);
          /* if no spin flip was accepted break the loop */
          if (h_flipped == 0)
          {
            #pragma omp atomic
            counter += j;
            break;
          }
        }

        calcEnergy<<< num_blocks2, block_size >>>(d_s, d_I, d_J, d_Kapbc, d_E);
        cudaMemcpy(h_E, d_E, sizeof(float) * NUM_SPINS / 2, cudaMemcpyDeviceToHost);

        E = 0.0;
	for (j = 0; j < NUM_SPINS / 2; j++)
        {
          E += h_E[j];
        }

        /* omp critical to avoid race condition */
        #pragma omp critical
        {
          if (E < Eapbc)
            Eapbc = E;
        }

        cudaErr(cudaGetLastError());	//check for CUDA errors before starting next round
      }

      #pragma omp master
      {
        fprintf(f, "%10.20f\t%10.20f\n", Epbc, Eapbc);
        fflush(f);
      }

      #pragma omp barrier
    }

    #pragma omp master
    {
      printf("Done writing N%d\n", N);
      fclose(f);
      free(filename);
    }

    #pragma omp barrier
  }

  /* free allocated memory */
  free(h_E);
  free(h_s);
  #pragma omp master
  {
    free(h_J);
    free(h_Kpbc);
    free(h_Kapbc);
    free(h_I);
  }

  cudaFree(d_E);
  cudaFree(d_s);
  cudaFree(d_J);
  cudaFree(d_Kpbc);
  cudaFree(d_Kapbc);
  cudaFree(d_I);

}	//parallel section ends here

  clock_gettime(CLOCK_REALTIME, &endtime);
  tdiff = (endtime.tv_sec-starttime.tv_sec)*NS_PER_S + endtime.tv_nsec-starttime.tv_nsec;
  printf("%.4lf spin updates per nanosecond (%llu lattice sweeps in %llu ns).\n", (double) NUM_SPINS * counter / tdiff, counter,  tdiff);

  return 0;
}

/********************************  END OF FILE  *******************************/
