#pragma once

// Include CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Particle.h"
#include "ParticleGrid.h"
#include "Simulation.h"

// funkce pro osetreni chyb
static void HandleError(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		//cout << cudaGetErrorString( error ) " in " << file << " at line " << line;
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		//scanf(" ");
		exit(EXIT_FAILURE);
	}
}
#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )


struct MyCudaWrapper {
	Simulation* simulation;

	MyCudaWrapper() {
		simulation = nullptr;
	}

	void init(Simulation& simulation);

};

__global__ void densityKernel(int particleCount, Particle* particles, ParticleGrid grid);

__global__ void velocityKernel(int particleCount, Particle* particles, ParticleGrid grid);

__global__ void integrateKernel(int particleCount, Particle* particles, float timeStep);