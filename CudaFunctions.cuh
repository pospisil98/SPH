// Include CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Particle.h"
#include "ParticleGrid.h"

__global__ void densityKernel(int particleCount, Particle* particles, ParticleGrid grid);

__global__ void velocityKernel(int particleCount, Particle* particles, ParticleGrid grid);

__global__ void integrateKernel(int particleCount, Particle* particles, float timeStep);