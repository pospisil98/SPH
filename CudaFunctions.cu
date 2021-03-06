#include "CudaFunctions.cuh"
#include "Simulation.h"
#include "Constants.h"

void MyCudaWrapper::Init(Simulation& simulation) {
	// Alloc particles on GPU
	CHECK_ERROR(cudaMalloc((void**)&simulation.particlesDevice, (simulation.MAX_PARTICLES) * sizeof(Particle)));

	// Page lock host particles
	size_t memsize = ((simulation.MAX_PARTICLES * sizeof(Particle) + 4095) / 4096) * 4096;
	//CHECK_ERROR(cudaHostRegister(simulation.particles.data(), memsize, cudaHostRegisterMapped));

	// Move particles from host to device
	CopyParticlesHostToDevice(simulation);

	// Allocate space for spatial grid
	CHECK_ERROR(cudaMalloc((void**)&simulation.particleGrid.gridDevice, simulation.particleGrid.grid.size() * sizeof(int)));

	//std::cout << "Allocated " << simulation.particleGrid.grid.size() * sizeof(int) << " for grid" << std::endl;

	// Page-lock host location of spatial grid
	// TODO: fix :(
	//memsize = ((simulation.particleGrid.grid.size() * sizeof(int) + 4095) / 4096) * 4096;
	//CHECK_ERROR(cudaHostRegister(simulation.particleGrid.grid.data(), memsize, cudaHostRegisterMapped));

	// Move spatial grid to GPU
	CopyGridHostToDevice(simulation);

	CHECK_ERROR(cudaDeviceSynchronize());
}

void MyCudaWrapper::Finalize(Simulation& simulation) {
	CHECK_ERROR(cudaFree(simulation.particlesDevice));
	CHECK_ERROR(cudaFree(simulation.particleGrid.gridDevice));
}

void MyCudaWrapper::WindowSizeChange(Simulation& simulation) {
	CHECK_ERROR(cudaFree(simulation.particleGrid.gridDevice));
	CHECK_ERROR(cudaMalloc((void**)&simulation.particleGrid.gridDevice, simulation.particleGrid.grid.size() * sizeof(int)));
}

void MyCudaWrapper::Update(Simulation& simulation, float timeStep) {
	// copy spatial grid to device
	CopyGridHostToDevice(simulation);

	// copy particles to device
	CopyParticlesHostToDevice(simulation);

	CHECK_ERROR(cudaDeviceSynchronize());

	// call kernels
	unsigned int threadCount = 256;
	unsigned int blockCount = std::ceil((float)simulation.particleCount / threadCount);

	densityPressureKernel << <blockCount, threadCount >> > (simulation.particleCount, simulation.particlesDevice, simulation.particleGrid, simulation.MASS, simulation.GAS_CONST, simulation.REST_DENS);
	//CHECK_ERROR(cudaDeviceSynchronize());
	forceKernel << <blockCount, threadCount >> > (simulation.particleCount, simulation.particlesDevice, simulation.particleGrid, simulation.MASS, simulation.VISC, simulation.G);
	//CHECK_ERROR(cudaDeviceSynchronize());
	integrateKernel << <blockCount, threadCount >> > (simulation.particleCount, simulation.particlesDevice, timeStep, simulation.BOUND_DAMPING, simulation.VIEW_WIDTH, simulation.VIEW_HEIGHT);

	//CHECK_ERROR(cudaDeviceSynchronize());

	// copy particles back to host
	CopyParticlesDeviceToHost(simulation);

	CHECK_ERROR(cudaDeviceSynchronize());
}

void MyCudaWrapper::CopyParticlesHostToDevice(Simulation& simulation) {
	CHECK_ERROR(cudaMemcpy(simulation.particlesDevice, simulation.particles, simulation.particleCount * sizeof(Particle), cudaMemcpyHostToDevice));
}

void MyCudaWrapper::CopyParticlesDeviceToHost(Simulation& simulation) {
	CHECK_ERROR(cudaMemcpy(simulation.particles, simulation.particlesDevice, simulation.particleCount * sizeof(Particle), cudaMemcpyDeviceToHost));
}

void MyCudaWrapper::CopyGridHostToDevice(Simulation& simulation) {
	CHECK_ERROR(cudaMemcpy(simulation.particleGrid.gridDevice, simulation.particleGrid.grid.data(), simulation.particleGrid.grid.size() * sizeof(int), cudaMemcpyHostToDevice));
}

__global__ void densityPressureKernel(int particleCount, Particle* particles, ParticleGrid grid, float MASS, float GAS_CONST, float REST_DENS)
{
	int particleID = blockDim.x * blockIdx.x + threadIdx.x;
	if (particleID >= particleCount) {
		return;
	}

	Particle& pi = particles[particleID];
	pi.rho = 0.0f;

	int posX;
	int posY;
	grid.Index1Dto2D(pi.gridCellID, posX, posY);

	// Go over neighbour cells
	for (int x = posX - 1; x <= posX + 1; x++) {
		for (int y = posY - 1; y <= posY + 1; y++) {
			// Check grid boundaries
			if (x < 0 || x >= grid.dimX || y < 0 || y >= grid.dimY) {
				continue;
			}

			int gridIndex = grid.Index2Dto1D(x, y);
			int currentIndex = grid.gridDevice[gridIndex];

			// While there are some neighbours in that grid cell
			while (currentIndex != -1) {
				Particle& pj = particles[currentIndex];

				MyVec2 rij = pj.position - pi.position;
				float r2 = rij.LengthSquared();

				if (r2 < HSQ) {
					pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
				}

				currentIndex = pj.nextParticle;
			}
		}
	}

	pi.p = GAS_CONST * (pi.rho - REST_DENS);
}

__global__ void forceKernel(int particleCount, Particle* particles, ParticleGrid grid, float MASS, float VISC, MyVec2 G)
{
	int particleID = blockDim.x * blockIdx.x + threadIdx.x;
	if (particleID >= particleCount) {
		return;
	}

	Particle& pi = particles[particleID];
	
	MyVec2 fpress(0.f, 0.f);
	MyVec2 fvisc(0.f, 0.f);

	int posX;
	int posY;
	grid.Index1Dto2D(pi.gridCellID, posX, posY);

	// Go over neighbour cells
	for (int x = posX - 1; x <= posX + 1; x++) {
		for (int y = posY - 1; y <= posY + 1; y++) {
			// Check grid boundaries
			if (x < 0 || x >= grid.dimX || y < 0 || y >= grid.dimY) {
				continue;
			}

			int gridIndex = grid.Index2Dto1D(x, y);
			int currentIndex = grid.gridDevice[gridIndex];

			// While there are some neighbours in that grid cell
			while (currentIndex != -1) {
				Particle& pj = particles[currentIndex];

				if (pi.id != pj.id) {
					MyVec2 rij = pj.position - pi.position;
					float r = rij.Length();

					if (r < H) {
						// compute pressure force contribution
						fpress += -rij.Normalized() * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);

						// compute viscosity force contribution
						fvisc += VISC * MASS * (pj.velocity - pi.velocity) / pj.rho * VISC_LAP * (H - r);
					}
				}

				currentIndex = pj.nextParticle;
			}
		}
	}

	MyVec2 fgrav = G * MASS / pi.rho;
	pi.force = fpress + fvisc + fgrav;
}

__global__ void integrateKernel(int particleCount, Particle* particles, float timeStep, float BOUND_DAMPING, float VIEW_WIDTH, float VIEW_HEIGHT)
{
	int particleID = blockDim.x * blockIdx.x + threadIdx.x;
	if (particleID >= particleCount) {
		return;
	}

	Particle& p = particles[particleID];

	// forward Euler integration
	if (p.rho > 0.0f) {
		p.velocity += timeStep * p.force / p.rho;
	}

	p.position += timeStep * p.velocity;

	// enforce boundary conditions
	if (p.position.x - EPS < 0.f) {
		p.velocity.x *= BOUND_DAMPING;
		p.position.x = EPS;
	}

	if (p.position.x + EPS > VIEW_WIDTH) {
		p.velocity.x *= BOUND_DAMPING;
		p.position.x = VIEW_WIDTH - EPS;
	}

	if (p.position.y - EPS < 0.f) {
		p.velocity.y *= BOUND_DAMPING;
		p.position.y = EPS;
	};

	if (p.position.y + EPS > VIEW_HEIGHT) {
		p.velocity.y *= BOUND_DAMPING;
		p.position.y = VIEW_HEIGHT - EPS;
	}
}