#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"
#include <cuda_runtime.h>

__global__ void compute(vector3* hPos, vector3* hVel, double* mass, vector3* matrix, int N);
__global__ void sumMatrix(vector3* hPos, vector3* hVel, vector3* matrix, int N); 
// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *h_mass, *d_mass;
vector3 *d_matrix; 

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	h_mass = (double *)malloc(sizeof(double) * numObjects);
}

int initDeviceMemory(int numObjects)
{
	cudaError_t result; 
	int sizeVector = sizeof(vector3) * numObjects; 
	int sizeDouble = sizeof(double) * numObjects; 
	int sizeMatrix = sizeof(vector3) * numObjects * numObjects; 
	result = cudaMalloc(&d_hPos, sizeVector); 
	if (result != cudaSuccess){ goto error; }
	result = cudaMalloc(&d_hVel, sizeVector); 
	if (result != cudaSuccess){ goto error;  }
	result = cudaMalloc(&d_mass, sizeDouble); 
	if (result != cudaSuccess){ goto error; }
	result = cudaMalloc(&d_matrix, sizeMatrix); 
	return 0; 
error:
	printf("Error allocating on device: %s\n", cudaGetErrorString(result)); 
	return 1; 
}

void copyToDevice(int numObjects) { 
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_hVel, hVel, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_mass, h_mass, sizeof(double) * numObjects, cudaMemcpyHostToDevice); 
}
void copyToHost(int numObjects) { 
	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost); 
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost); 
}
//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(h_mass);
}

void freeDeviceMemory() 
{
	cudaFree(d_hVel); 
	cudaFree(d_hPos); 
	cudaFree(d_mass); 
}
//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		h_mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			h_mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",h_mass[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	int	err = initDeviceMemory(NUMENTITIES); 
	if(err != 0) { 
		printf("Error in device memory allocation"); 
 	} 
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	copyToDevice(NUMENTITIES); 
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	#endif
	int threadsPerBlock = 256; 
	int blocksPerGrid = (NUMENTITIES + threadsPerBlock - 1) / threadsPerBlock;
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute<<<blocksPerGrid,threadsPerBlock>>>(d_hPos, d_hVel, d_mass, d_matrix, NUMENTITIES); 
		cudaDeviceSynchronize(); 
		sumMatrix<<<NUMENTITIES,256>>>(d_hPos, d_hVel, d_matrix, NUMENTITIES); 	
		cudaDeviceSynchronize(); 
	}
	copyToHost(NUMENTITIES); 
	clock_t t1=clock()-t0;
#ifdef DEBUG
	printf("--------- COMPUTED SYSTEM ----------\n"); 
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
	freeDeviceMemory(); 
}
