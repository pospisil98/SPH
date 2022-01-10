#include "CudaFunctions.cuh"

void MyCudaWrapper::init(Simulation& simulation) {
	this->simulation = &simulation;

	// Alloc particles on GPU
	CHECK_ERROR(cudaMalloc((void**)&simulation.particlesDevice, (simulation.MAX_PARTICLES) * sizeof(Particle)));

	// Page lock host particles
	size_t memsize = ((simulation.MAX_PARTICLES * sizeof(Particle) + 4095) / 4096) * 4096;
	CHECK_ERROR(cudaHostRegister(simulation.particles.data(), memsize, cudaHostRegisterMapped));

	// Move particles from host to device
	cudaMemcpy(simulation.particlesDevice, simulation.particles.data(), simulation.particleCount * sizeof(Particle), cudaMemcpyHostToDevice);

	// Allocate space for spatial grid

	// Page-lock host location of spatial grid

	// Move spatial grid to GPU
}

__global__ void densityKernel(int particleCount, Particle* particles, ParticleGrid grid)
{

}

__global__ void velocityKernel(int particleCount, Particle* particles, ParticleGrid grid)
{

}

__global__ void integrateKernel(int particleCount, Particle* particles, float timeStep)
{

}