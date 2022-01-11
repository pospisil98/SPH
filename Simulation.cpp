#include "Simulation.h"

void Simulation::Initialize() {
	// Do grid initialization everytime to be able to switch acceleration on and off
	int dimX = (VIEW_WIDTH + EPS) / (2.f * H);
	int dimY = (VIEW_HEIGHT + EPS) / (2.f * H);
	particleGrid.particles = particles;
	particleGrid.Initialize(dimX, dimY);

	// Initialize cuda wrapper - memory registrations and allocations
	cudaWrapper.Init(*this);
}

void Simulation::Reset()
{
	delete particles;
	particles = new Particle[MAX_PARTICLES];

	particleCount = 0;

	InitSPH();
}

void Simulation::Update(float timeStep) {
	timeStep = fixedTimestep ? DT : timeStep;

	if (useSpatialGrid) {
		particleGrid.Update();
	}

	if (simulateOnGPU && useSpatialGrid) {
		cudaWrapper.Update(*this, timeStep);
	}
	else {

		ComputeDensityPressure();
		ComputeForces();

		Integrate(timeStep);
	}
}

void Simulation::InitSPH() {
	std::cout << "Init dam break with " << DAM_BREAK_PARTICLES << " particles" << std::endl;

	for (float y = EPS; y < VIEW_HEIGHT - EPS * 2.f; y += H) {
		for (float x = VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H) {
			if (particleCount < DAM_BREAK_PARTICLES) {
				float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				particles[particleCount] = Particle(x + jitter, y, particleCount);
				particleCount++;
			}
			else {
				return;
			}
		}
	}
}

void Simulation::AddParticleRectangle() {
	if (particleCount >= MAX_PARTICLES) {
		std::cout << "Maximum number of particles reached " << MAX_PARTICLES << std::endl;
	}
	else {
		unsigned int placed = 0;
		for (float y = VIEW_HEIGHT / 1.5f - VIEW_HEIGHT / 5.f; y < VIEW_HEIGHT / 1.5f + VIEW_HEIGHT / 5.f; y += H * 0.95f) {
			for (float x = VIEW_WIDTH / 2.f - VIEW_HEIGHT / 5.f; x <= VIEW_WIDTH / 2.f + VIEW_HEIGHT / 5.f; x += H * 0.95f) {
				if (placed++ < BLOCK_PARTICLES && particleCount < MAX_PARTICLES) {
					particles[particleCount] = Particle(x, y, particleCount);
					particleCount++;
				}
			}
		}
	}
}

void Simulation::SetDefaultParameters() {
	REST_DENS = 300.f;
	GAS_CONST = 2000.f;
	MASS = 2.5f;
	VISC = 200.f;

	BOUND_DAMPING = -0.5f;
}

void Simulation::windowRescaleRoutine(int width, int height) {
	int ogWidth = VIEW_WIDTH;
	int ogHeight = VIEW_HEIGHT;

	// Update window properties
	WINDOW_WIDTH = width * 1.5f;
	WINDOW_HEIGHT = height * 1.5f;
	VIEW_WIDTH = 1.5f * WINDOW_WIDTH;
	VIEW_HEIGHT = 1.5f * WINDOW_HEIGHT;

	// rescale positions of particles
	for (int i = 0; i < particleCount; i++) {
		particles[i].position.x = (particles[i].position.x * VIEW_WIDTH) / ogWidth;
		particles[i].position.y = (particles[i].position.y * VIEW_HEIGHT) / ogHeight;
	}

	// Update uniform grid
	int dimX = (VIEW_WIDTH + EPS) / (2.f * H);
	int dimY = (VIEW_HEIGHT + EPS) / (2.f * H);

	particleGrid.Initialize(dimX, dimY);

	if (useSpatialGrid) {
		particleGrid.Update();
	}

	cudaWrapper.WindowSizeChange(*this);
}

void Simulation::ComputeDensityPressure() {
	for (int i = 0; i < particleCount; i++) {
		Particle& pi = particles[i];
		pi.rho = 0.f;

		std::vector<int> potentialNeighbours;
		GetNeighbourParticlesIndices(pi.id, potentialNeighbours);

		for (int index : potentialNeighbours) {
			Particle pj = particles[index];

			MyVec2 rij = pj.position - pi.position;
			float r2 = rij.LengthSquared();

			if (r2 < HSQ) {
				//std::cout << "Collision in Density for id: " << pi.id << " and " << pj.id << std::endl;
				pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.p = GAS_CONST * (pi.rho - REST_DENS);
	}
}

void Simulation::ComputeForces() {
	for (int i = 0; i < particleCount; i++) {
		Particle& pi = particles[i];

		MyVec2 fpress(0.f, 0.f);
		MyVec2 fvisc(0.f, 0.f);

		std::vector<int> potentialNeighbours;
		GetNeighbourParticlesIndices(pi.id, potentialNeighbours);

		for (int index : potentialNeighbours) {
			Particle pj = particles[index];
			if (pi.id == pj.id) {
				continue;
			}

			MyVec2 rij = pj.position - pi.position;
			float r = rij.Length();

			if (r < H) {
				//std::cout << "Collision in forces for id: " << pi.id << " and " << pj.id << std::endl;
				// compute pressure force contribution
				fpress += -rij.Normalized() * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
				// compute viscosity force contribution
				fvisc += VISC * MASS * (pj.velocity - pi.velocity) / pj.rho * VISC_LAP * (H - r);
			}
		}
		MyVec2 fgrav = G * MASS / pi.rho;
		pi.force = fpress + fvisc + fgrav;
	}
}

void Simulation::Integrate(float deltaTime) {
	for (int i = 0; i < particleCount; i++) {
		Particle& p = particles[i];

		// forward Euler integration
		if (p.rho > 0.0f) {
			p.velocity += deltaTime * p.force / p.rho;
		}
		p.position += deltaTime * p.velocity;

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
}

void Simulation::GetNeighbourParticlesIndices(int particleID, std::vector<int>& indices) {
	indices.clear();
	if (useSpatialGrid) {
		particleGrid.GetNeighbourParticlesIndices(particleID, indices);
	}
	else {
		for (int i = 0; i < particleCount; i++) {
			indices.push_back(i);
		}
	}
}