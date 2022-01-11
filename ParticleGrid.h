#pragma once

#include <vector>

#include "Particle.h"

/// <summary>
/// Representation of uniform spatial grid for faster neighbour search
/// 
/// Grid is represented only as vector of indices of first particle in grid cell. Connection to other is
/// implemented in each particle via nextParticle.
/// 
/// </summary>
struct ParticleGrid {
	/// <summary> Vector of indexes of first particle in grid cell  </summary>
	std::vector<int> grid;
	/// <summary> Pointer to grid vector on DEVICE </summary>
	int* gridDevice;

	/// <summary> Pointer to particle array (kinda wrong, but solves circular dependency) </summary>
	Particle* particles;
	/// <summary> Pointer to particle count in simulation (same as before) </summary>
	int& particleCount;

	/// <summary> X dimension of grid </summary>
	int dimX;
	/// <summary> Y dimension of grid </summary>
	int dimY;

	ParticleGrid(Particle* _particles, int& _particleCount) :
		particles(_particles),
		particleCount(_particleCount)
	{
		dimX = -1;
		dimY = -1;

		gridDevice = nullptr;
	}

	ParticleGrid(int _dimX, int _dimY, Particle* _particles, int& _particleCount) :
		particles(_particles),
		particleCount(_particleCount)
	{
		dimX = _dimX;
		dimY = _dimY;

		gridDevice = nullptr;

		Clear();
	}

	/// <summary>
	/// Initializatin of grid to given dimensions.
	/// </summary>
	/// <param name="_dimX">Size in X dimension</param>
	/// <param name="_dimY">Size in Y dimension</param>
	void Initialize(int _dimX, int _dimY);

	/// <summary>
	/// Adds particle with given ID into spatial grid.
	/// </summary>
	/// <param name="particleID">ID of particle to add</param>
	void Add(int particleID);

	/// <summary>
	/// Clears grid adds all particles into it.
	/// </summary>
	void Update();

	/// <summary>
	/// Clears grid and sets all contents to -1.
	/// </summary>
	void Clear();

	/// <summary>
	/// Gets grid cell index for particle with given ID.
	/// </summary>
	/// <param name="particleID">ID of particle to get grid cell index for</param>
	/// <returns>1D index of cell index in spatial grid</returns>
	int GetGridCellIndexFromParticleIndex(int particleID);

	/// <summary>
	/// Gets indices of neigbour particles for given particle ID.
	/// </summary>
	/// <param name="particleIndex">ID of particle to get particles neighbours indices</param>
	/// <param name="indices">Reference to vector where indices should be stored</param>
	void GetNeighbourParticlesIndices(int particleIndex, std::vector<int>& indices);

	/// <summary>
	/// Get indices of neighbour grid cells for given cell.
	/// </summary>
	/// <param name="index">ID of cell to get neighbours for.</param>
	/// <returns>Vector of 1D indices in grid array</returns>
	std::vector<int> GetNeighbourCellIndices(int index);

	/// <summary>
	/// Converts 1D grid index into 2D
	/// </summary>
	/// <param name="index">Index to convert</param>
	/// <param name="x">Reference to X coordinate storage</param>
	/// <param name="y">Reference to Y coordinate storage</param>
	inline __host__ __device__ void Index1Dto2D(int index, int& x, int& y) {
		x = index % dimX;
		y = index / dimX;
	}

	/// <summary>
	/// Converts 1D grid index into 2D
	/// </summary>
	/// <param name="index">Index to convert</param>
	/// <returns>Pair of (X,Y) 2D index</returns>
	inline std::pair<int, int> Index1Dto2D(int index) {
		int x = index % dimX;
		int y = index / dimX;

		return std::make_pair(x, y);
	}

	/// <summary>
	/// Convert 2D index into 1D
	/// </summary>
	/// <param name="x">X coordinate of index</param>
	/// <param name="y">Y coordinate of index</param>
	/// <returns>Converted 1D index</returns>
	inline  __host__ __device__ int Index2Dto1D(int x, int y) {
		return x + y * dimX;
	}
};
