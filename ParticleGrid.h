#pragma once
#include <vector>
#include <unordered_set>

#include "Particle.h"

struct ParticleGrid {
	std::vector<std::unordered_set<int>> grid;

	std::vector<Particle>& particles;

	int dimX;
	int dimY;

	ParticleGrid(std::vector<Particle>& _particles) :
		particles(_particles) 
	{
		dimX = -1;
		dimY = -1;
	}

	ParticleGrid(int _dimX, int _dimY, std::vector<Particle>& _particles) :
		particles(_particles)
	{
		dimX = _dimX;
		dimY = _dimY;

		grid.resize(dimX * dimY);
	}

	void Initialize(int _dimX, int _dimY, std::vector<Particle>& _particles);

	void Add(int particleID);

	void Update();

	void Clear();

	int GetGridCellIndexFromParticleIndex(int particleID);

	void GetNeighbourParticlesIndices(int particleIndex, std::vector<int>& indices);

	std::vector<int> GetNeighbourCellIndices(int index);

	inline int Index2Dto1D(int x, int y) { 
		return x + y * dimX;
	}

	inline std::pair<int, int> Index1Dto2D(int index) {
		int x = index % dimX;
		int y = index / dimX;

		return std::make_pair(x, y);
	}
};
