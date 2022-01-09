#pragma once

#include <vector>
#include <iostream>

#include "Constants.h"
#include "MyVec2.h"
#include "Particle.h"
#include "ParticleGrid.h"

struct Simulation {
	std::vector<Particle> particles;

	ParticleGrid particleGrid;
	
	bool useSpatialGrid = true;
	bool simulateOnGPU = false;
	bool fixedTimestep = true;

	int MAX_PARTICLES = 5000;
	float REST_DENS = 300.f;		// rest density
	float GAS_CONST = 2000.f;		// const for equation of state
	float MASS = 2.5f;				// assume all particles have the same mass
	float VISC = 200.f;				// viscosity constant
	float BOUND_DAMPING = -0.5f;

	Simulation() : particleGrid(-1, -1, particles) {
		Initialize();
	}

	void Initialize();

	void Update(float timeStep = DT);

	void Reset();

	void SetDefaultParameters();


	void InitSPH();

	void AddParticleRectangle();


	void ComputeDensityPressure();

	void ComputeForces();

	void Integrate(float deltaTime);


	void GetNeighbourParticlesIndices(int particleID, std::vector<int>& indices);

	void windowRescaleRoutine(int prevWidth, int prevHeight);
};