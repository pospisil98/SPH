#pragma once

#include "MyVec2.h"

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

/// <summary>
/// Struct representing particle of SPH simulation
/// </summary>
struct Particle {
	/// <summary> Position in 2D </summary>
	MyVec2 position;
	/// <summary> Velocity of particle </summary>
	MyVec2 velocity;
	/// <summary> Force action on particle </summary>
	MyVec2 force;

	/// <summary> Particle density </summary>
	float rho;
	/// <summary> Pressure </summary>
	float p;

	/// <summary> Index of paricle in particle array </summary>
	int id;
	/// <summary> Index of grid cell in grid cell indices array </summary>
	int gridCellID;

	/// <summary> Next particle in grid (grid is represented as linked list, -1 when none) </summary>
	int nextParticle;
	

	Particle() {
		position = MyVec2();
		velocity = MyVec2();
		force = MyVec2();

		rho = 0.0f;
		p = 0.0f;

		id = -1;
		nextParticle = -1;
		gridCellID = -1;
	}

	Particle(float _x, float _y, int _id) :
		position(_x, _y),
		velocity(0.f, 0.f),
		force(0.f, 0.f),
		rho(0),
		p(0.0f),
		id(_id)
	{
		nextParticle = -1;
		gridCellID = 0;
	}
};