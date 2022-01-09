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