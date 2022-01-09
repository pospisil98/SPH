#pragma once

#include "MyVec2.h"

struct Particle {
	MyVec2 position;
	MyVec2 velocity;
	MyVec2 force;

	float rho;
	float p;

	int id;
	int gridCellID;

	Particle() {}

	Particle(float _x, float _y, int _id) :
		position(_x, _y),
		velocity(0.f, 0.f),
		force(0.f, 0.f),
		rho(0),
		p(0.0f),
		id(_id)
	{
		gridCellID = 0;
	}
};