#include "Simulation.h"


void Simulation::Update(float timeStep) {
	timeStep = fixedTimestep ? DT : timeStep;

	if (simulateOnGPU) {
		if (useSpatialGrid) {
			particleGrid.Update();
		}

		// copy grid to GPU

		// copy particles to GPU

		// kernels

		// copy particles back

	}
	else {
		if (useSpatialGrid) {
			particleGrid.Update();
		}

		ComputeDensityPressure();
		ComputeForces();

		Integrate(timeStep);
	}
}

void Simulation::Initialize() {
	// Do grid initialization everytime to be able to switch acceleration on and off
	int dimX = (VIEW_WIDTH + EPS) / (2.f * H);
	int dimY = (VIEW_HEIGHT + EPS) / (2.f * H);

	particleGrid.Initialize(dimX, dimY);
}

void Simulation::Reset()
{
	particles.clear();
	InitSPH();
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
	GRAVITY_VAL = 9.81f;
	G.x = 0.f;
	G.y = -GRAVITY_VAL;
	REST_DENS = 300.f;
	GAS_CONST = 2000.f;
	H = 16.f;
	HSQ = H * H;
	MASS = 2.5f;
	VISC = 200.f;
	DT = 0.0007f;

	EPS = H;
	BOUND_DAMPING = -0.5f;
}

void Simulation::windowRescaleRoutine(int prevWidth, int prevHeight) {
	// rescale positions of particles
	for (int i = 0; i < particleCount; i++) {
		Particle& p = particles[i];

		p.position.x = (p.position.x * VIEW_WIDTH) / prevWidth;
		p.position.y = (p.position.y * VIEW_HEIGHT) / prevHeight;
	}

	// Update uniform grid
	int dimX = (VIEW_WIDTH + EPS) / (2.f * H);
	int dimY = (VIEW_HEIGHT + EPS) / (2.f * H);

	particleGrid.Initialize(dimX, dimY);

	if (useSpatialGrid) {
		particleGrid.Update();
	}
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