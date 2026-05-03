#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

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
    