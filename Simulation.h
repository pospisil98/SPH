#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>

#include "Constants.h"
#include "MyVec2.h"
#include "Particle.h"
#include "ParticleGrid.h"

#include "CudaFunctions.cuh"

struct Simulation {
	Particle* particles;
	Particle* particlesDevice;
	
	ParticleGrid particleGrid;

	MyCudaWrapper cudaWrapper;
	
	bool useSpatialGrid = true;
	bool simulateOnGPU = true;
	bool fixedTimestep = true;

	int WINDOW_WIDTH = 1280;
	int WINDOW_HEIGHT = 720;
	double VIEW_WIDTH = 1.5f * WINDOW_WIDTH;
	double VIEW_HEIGHT = 1.5f * WINDOW_HEIGHT;

	int particleCount = 0;

	int MAX_PARTICLES = 15000;
	int DAM_BREAK_PARTICLES = 100;
	int BLOCK_PARTICLES = 400;

	MyVec2 G = MyVec2(0.0f, -GRAVITY_VAL);
	float REST_DENS = 300.f;		// rest density
	float GAS_CONST = 2000.f;		// const for equation of state
	float MASS = 2.5f;				// assume all particles have the same mass
	float VISC = 200.f;				// viscosity constant
	float BOUND_DAMPING = -0.5f;

	Simulation() : particleGrid(-1, -1, particles, particleCount) {
		particles = new Particle[MAX_PARTICLES];

		Initialize();	
	}

	~Simulation() {
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