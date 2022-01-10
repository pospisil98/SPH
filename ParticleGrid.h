#pragma once
#include <vector>
#include <unordered_set>

#include "Particle.h"

struct ParticleGrid {
	std::vector<int> grid;
	int* gridDevice;

	std::vector<Particle>& particles;
	int& particleCount;

	int dimX;
	int dimY;

	ParticleGrid(std::vector<Particle>& _particles, int& _particleCount) :
		particles(_particles),
		particleCount(_particleCount)
	{
		dimX = -1;
		dimY = -1;

		gridDevice = nullptr;
	}

	ParticleGrid(int _dimX, int _dimY, std::vector<Particle>& _particles, int& _particleCount) :
		particles(_particles),
		particleCount(_particleCount)
	{
		dimX = _dimX;
		dimY = _dimY;

		gridDevice = nullptr;

		Clear();
	}

	void Initialize(int _dimX, int _dimY);

	void Add(int particleID);

	void Update();

	void Clear();

	int GetGridCellIndexFromParticleIndex(int particleID);

	void GetNeighbourParticlesIndices(int particleIndex, std::vector<int>& indices);

	std::vector<int> GetNeighbourCellIndices(int index);

	inline __host__ __device__ void Index1Dto2D(int index, int& x, int& y) {
		x = index % dimX;
		y = index / dimX;
	}

	inline  __host__ __device__ int Index2Dto1D(int x, int y) {
		return x + y * dimX;
	}

	inline std::pair<int, int> Index1Dto2D(int index) {
		int x = index % dimX;
		int y = index / dimX;

		return std::make_pair(x, y);
	}
};
