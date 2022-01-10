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

struct Particle {
	MyVec2 position;
	MyVec2 velocity;
	MyVec2 force;

	float rho;
	float p;

	int id;
	int nextParticle;
	int gridCellID;

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