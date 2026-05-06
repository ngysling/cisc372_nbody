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
	int i = threadIdx.x + blockIdx.x + blockDim.x; 
	int j = threadIdx.y + blockIdx.y + blockDim.y; 
	if (i == j) { 
		FILL_VECTOR(matrix[(i * NUMENTITIES) + j],0,0,0); 
	} else { 
		vector3 distance;
		for (int k=0;k<3;k++) { distance[k]=pos[i * NUMENTITIES + k]-pos[j * NUMENTITIES + k]; } 
   		double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
		double magnitude=sqrt(magnitude_sq);
		double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
		FILL_VECTOR(matrix[i * NUMENTITIES + j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	}
}

/* 
__global__ void sumMatrix(vector3* hPos, vector3* hVel, vector3* matrix, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; // i is our thread index.
	if(i < N) {
		vector3 accel_sum={0,0,0}; 
		int row = i * N; 
		for(int j = 0; j < N; j++) {
			accel_sum[0] += matrix[row + j][0]; 
			accel_sum[1] += matrix[row + j][1]; 
			accel_sum[2] += matrix[row + j][2]; 
		}
		for (int k = 0; k<3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL; 
			hPos[i][k] += hVel[i][k] * INTERVAL; 
		}
	}
}
*/ 

#define SIZE 256
#define SHM_SIZE 256 // assuming 1 block of 1024 threads 
/*
Sums up a column of the matrix based
*/ 
__global__ void sumMatrix(vector3* hPos, vector3* hVel, vector3* matrix, int N) { 
	__shared__ vector3 sharedSum[SHM_SIZE]; 
	vector3 runningSum = {0,0,0}; 
	for (int i = threadIdx.x; i < N; i += 256) { 
		int j = blockIdx.x * blockDim.x + threadIdx.x; // col
		if(i < N) { 
			runningSum[0] += matrix[j][0] ;
			runningSum[1] += matrix[j][1] ;
			runningSum[2] += matrix[j][2] ;
		}
	}
	sharedSum[threadIdx.x] = runningSum; 
	__syncthreads(); 
	for (int s = 1; s < blockDim.x; s *= 2) {
		if(threadIdx.x % (2 * s) == 0) { 
			runningSum[threadIdx.x] += runningSum[threadIdx.x + s]; 
		}
		__syncthreads(); 
	}
	if(threadIdx.x == 0) { 
		sum = &runningSum[0]; 
		for (int k = 0; k<3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL; 
			hPos[i][k] += hVel[i][k] * INTERVAL; 
		}
	}
}
