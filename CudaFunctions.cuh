#pragma once

// Include CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Particle.h"
#include "ParticleGrid.h"

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

struct Simulation;

struct MyCudaWrapper {
	MyCudaWrapper() { }

	void Init(Simulation& simulation);

	void Update(Simulation& simulation, float timeStep);

	void CopyParticlesHostToDevice(Simulation& simulation);

	void CopyParticlesDeviceToHost(Simulation& simulation);

	void CopyGridHostToDevice(Simulation& simulation);

};

__global__ void densityPressureKernel(int particleCount, Particle* particles, ParticleGrid grid, float MASS, float GAS_CONST, float REST_DENS);

__global__ void forceKernel(int particleCount, Particle* particles, ParticleGrid grid, float MASS, float VISC, MyVec2 G);

__global__ void integrateKernel(int particleCount, Particle* particles, float timeStep, float BOUND_DAMPING, float VIEW_WIDTH, float VIEW_HEIGHT);