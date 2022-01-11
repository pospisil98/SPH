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

/// <summary>
/// Struct representing SPH simulation
/// 
/// Including both CPU and GPU implementation using CUDA.
/// 
/// Based on https://lucasschuermann.com/writing/particle-based-fluid-simulation
/// </summary>
struct Simulation {
	/// <summary> Array of simulated particles </summary>
	Particle* particles;
	/// <summary> Pointer to this array on DEVICE </summary>
	Particle* particlesDevice;

	/// <summary> Uniorm spatial grid for neighbour search acceleration </summary>
	ParticleGrid particleGrid;

	/// <summary> Wrapper of CUDA functions </summary>
	MyCudaWrapper cudaWrapper;
	
	/// <summary> Flag controlling usage of acceleration structure </summary>
	bool useSpatialGrid = true;
	/// <summary> Flag controlling CPU/GPU computations </summary>
	bool simulateOnGPU = false;
	/// <summary> Flag controlling usage of fixed timestep (adapive pretty much doesnt work so..) </summary>
	bool fixedTimestep = true;

	/// <summary> Width of simulation window </summary>
	int WINDOW_WIDTH = 1280;
	/// <summary> Height of simulaton window</summary>
	int WINDOW_HEIGHT = 720;
	/// <summary> Width of simulation view </summary>
	double VIEW_WIDTH = 1.5f * WINDOW_WIDTH;
	/// <summary> Heiht of simulation view </summary>
	double VIEW_HEIGHT = 1.5f * WINDOW_HEIGHT;

	/// <summary> Number of particles in simulation </summary>
	int particleCount = 0;

	/// <summary> Maximum number of particles in simulation </summary>
	int MAX_PARTICLES = 20000;
	/// <summary> Number of particles used in initial dam break scenario </summary>
	int DAM_BREAK_PARTICLES = 2048;
	/// <summary> Number of particles being added in block </summary>
	int BLOCK_PARTICLES = 400;

	/// <summary> Gravity in simulation </summary>
	MyVec2 G = MyVec2(0.0f, -GRAVITY_VAL);
	/// <summary> Rest density of particles </summary>
	float REST_DENS = 300.f;
	/// <summary> Gas constant for equation of state </summary>
	float GAS_CONST = 2000.f;
	/// <summary> Mass of each particle </summary>
	float MASS = 2.5f;
	/// <summary> Viscosity constant </summary>
	float VISC = 200.f;
	/// <summary> Damping of bounds</summary>
	float BOUND_DAMPING = -0.5f;

	Simulation() : particleGrid(-1, -1, particles, particleCount) {
		particles = new Particle[MAX_PARTICLES];

		Initialize();	
	}

	~Simulation() {
		delete particles;

		cudaWrapper.Finalize(*this);
	}

	/// <summary>
	/// Initializes simulation - intialization of grid and cuda wrapper
	/// </summary>
	void Initialize();

	/// <summary>
	/// Resets simulation
	/// </summary>
	void Reset();

	/// <summary>
	/// Updates simulation, GPU/CPU according to the flags
	/// <param name="timeStep">Timestep of simulation, defaultly DT from constants</param>
	/// </summary>
	void Update(float timeStep = DT);

	/// <summary>
	/// Sets parameters of simulation to default values
	/// </summary>
	void SetDefaultParameters();

	/// <summary>
	/// Initializes SPH simulation to dam break scenario
	/// </summary>
	void InitSPH();

	/// <summary>
	/// Adds rectangle of particle sinto scene
	/// </summary>
	void AddParticleRectangle();

	/// <summary>
	/// Computes densty and pressure of particles
	/// </summary>
	void ComputeDensityPressure();

	/// <summary>
	/// Computes forces acting on particles
	/// </summary>
	void ComputeForces();

	/// <summary>
	/// Computes new positions of particles using Euler method
	/// <param name="deltaTime">Timestep of simulation</param>
	/// </summary>
	void Integrate(float deltaTime);

	/// <summary>
	/// Gets indices of potential neighbour particles of particle with given ID
	/// <param name="particleID">ID of particle to get neighbours for</param>
	/// <param name="indices">Vector where indices should be stored</param>
	/// </summary>
	void GetNeighbourParticlesIndices(int particleID, std::vector<int>& indices);

	/// <summary>
	/// Routine which should be called after widnow resize, rescales particels and updates everything.
	/// <param name="width">New widow width</param>
	/// <param name="height">New window height</param>
	/// </summary>
	void windowRescaleRoutine(int width, int height);
};