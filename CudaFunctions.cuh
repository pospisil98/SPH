#pragma once

// Include CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Particle.h"
#include "ParticleGrid.h"


/// <summary> Simple macro for easier CUDA error handling (USAGE: call method in that) </summary>
#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )

// Forward declaration - solution of circullar dependency
struct Simulation;


/// <summary>
/// Struct wrapping all CUDA related functions and calls
/// </summary>
struct MyCudaWrapper {
	MyCudaWrapper() { }

	/// <summary>
	/// Initialize CUDA Wrapper (memory registrations, allocations)
	/// </summary>
	/// <param name="simulation">Reference to simulation class</param>
	void Init(Simulation& simulation);

	/// <summary>
	/// Clears all memory and so on.
	/// </summary>
	/// <param name="simulation">Reference to simulation class</param>
	void Finalize(Simulation& simulation);

	/// <summary>
	/// Updates simulation on GPU (assuming that grid has been updated already)
	/// </summary>
	/// <param name="simulation">Reference to simulation</param>
	/// <param name="timeStep">Simulation timestep duration</param>
	void Update(Simulation& simulation, float timeStep);

	/// <summary>
	/// Wrapper for copying particles from HOST to DEVICE
	/// </summary>
	/// <param name="simulation">Reference to simulation</param>
	void CopyParticlesHostToDevice(Simulation& simulation);

	/// <summary>
	/// Wrapper for copying particles from DEVICE to HOST
	/// </summary>
	/// <param name="simulation">Reference to simulation</param>
	void CopyParticlesDeviceToHost(Simulation& simulation);

	/// <summary>
	/// Wrapper for copying spatial grid from HOST to DEVICE
	/// </summary>
	/// <param name="simulation">Reference to simulation</param>
	void CopyGridHostToDevice(Simulation& simulation);

	/// <summary>
	/// Method for updating CUDA related structures on window size change
	/// /// <param name="simulation">Reference to simulation</param>
	/// </summary>
	void WindowSizeChange(Simulation& simulation);
};

/// <summary>
/// Kernel for computing density and pressure of particles
/// </summary>
/// <param name="particleCount">Number of particles in the scene</param>
/// <param name="particles">Pointer to particle array on DEVICE</param>
/// <param name="grid">Spatial grid structure</param>
/// <param name="MASS">Particle mass</param>
/// <param name="GAS_CONST">Gass constant</param>
/// <param name="REST_DENS">Rest denstiy</param>
__global__ void densityPressureKernel(int particleCount, Particle* particles, ParticleGrid grid, float MASS, float GAS_CONST, float REST_DENS);

/// <summary>
/// Kernel for computing forces action on particles
/// </summary>
/// <param name="particleCount">Number of particles in the scene</param>
/// <param name="particles">Pointer to particle array on DEVICE</param>
/// <param name="grid">Spatial grid structure</param>
/// <param name="MASS">Particle mass</param>
/// <param name="VISC">Viscosity of fluid</param>
/// <param name="G">Gravity vector</param>
__global__ void forceKernel(int particleCount, Particle* particles, ParticleGrid grid, float MASS, float VISC, MyVec2 G);

/// <summary>
/// Kernel for computing new positions using Euler method
/// </summary>
/// <param name="particleCount">Number of particles in the scene</param>
/// <param name="particles">Pointer to particle array on DEVICE</param>
/// <param name="timeStep">Simulation timestep duration</param>
/// <param name="BOUND_DAMPING">Damping factor of boundaries</param>
/// <param name="VIEW_WIDTH">Width of view (for collisions)</param>
/// <param name="VIEW_HEIGHT">Height of view (for collisions)</param>
__global__ void integrateKernel(int particleCount, Particle* particles, float timeStep, float BOUND_DAMPING, float VIEW_WIDTH, float VIEW_HEIGHT);



/// <summary>
/// Helper function for handling and checking CUDA errors
/// </summary>
/// <param name="error">Error code</param>
/// <param name="file">File where error happened</param>
/// <param name="line">Line where error happened</param>
static void HandleError(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		//cout << cudaGetErrorString( error ) " in " << file << " at line " << line;
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		//scanf(" ");
		exit(EXIT_FAILURE);
	}
}