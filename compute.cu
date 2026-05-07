#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h> 

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
// we're on the device, have access to pos, vel, mass and a NxN matrix to store the compute result. 
__global__ void compute(vector3* pos, vector3* vel, double* mass, vector3* matrix, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x ; 
	if (i < N) { 
		for(int j = 0; j < N; j++) { 
			int matIndex = i * N + j; 
			if(i == j) {	
				FILL_VECTOR(matrix[matIndex],0,0,0); 
			}
			else { 
				vector3 distance;
				for (int k=0;k<3;k++) { 
					distance[k]=pos[i][k]-pos[j][k]; 
				} 
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(matrix[matIndex],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
}
__global__ void sumMatrix(vector3* hPos, vector3* hVel, vector3* matrix, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N) {
		vector3 accel_sum={0,0,0}; 
		for(int j = 0; j < N; j++) {
			int matIndex = i * N + j; 
			accel_sum[0] += matrix[matIndex][0]; 
			accel_sum[1] += matrix[matIndex][1]; 
			accel_sum[2] += matrix[matIndex][2]; 
		}
		for (int k = 0; k<3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL; 
			hPos[i][k] += hVel[i][k] * INTERVAL; 
		}
	}
}
#define SIZE 256
#define SHM_SIZE 256 // assuming 1 block of 1024 threads 
/*
Sums up a column of the matrix based
__global__ void sumMatrix(vector3* hPos, vector3* hVel, vector3* matrix, int N) { 
	__shared__ vector3 sharedSum[SHM_SIZE]; 
	vector3 runningSum = {0,0,0}; 
	for (int i = threadIdx.x; i < N; i += 256) { 
		int j = blockIdx.x * blockDim.x + i; 
		if(i < N) { 
			runningSum[0] += matrix[j][0] ;
			runningSum[1] += matrix[j][1] ;
			runningSum[2] += matrix[j][2] ;
		}
	}
	sharedSum[threadIdx.x][0] = runningSum[0]; 
	sharedSum[threadIdx.x][1] = runningSum[1]; 
	sharedSum[threadIdx.x][2] = runningSum[2]; 
	__syncthreads(); 
	for (int s = 1; s < blockDim.x; s *= 2) {
		if(threadIdx.x % (2 * s) == 0) { 
			sharedSum[threadIdx.x][0] += sharedSum[threadIdx.x + s][0]; 
			sharedSum[threadIdx.x][1] += sharedSum[threadIdx.x + s][1]; 
			sharedSum[threadIdx.x][2] += sharedSum[threadIdx.x + s][2]; 
		}
		__syncthreads(); 
	}
	if(threadIdx.x == 0) { 
		for (int k = 0; k<3; k++) {
			hVel[blockIdx.x * blockDim.x + threadIdx.x][k] += sharedSum[0][k] * INTERVAL; 
			hPos[blockIdx.x * blockDim.x + threadIdx.x][k] += hVel[blockIdx.x * blockDim.x + threadIdx.x][k] * INTERVAL; 
		}
	}
}
*/
