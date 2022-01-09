#include <iostream>

#include "ParticleGrid.h"
#include "Constants.h"

void ParticleGrid::Initialize(int _dimX, int _dimY, std::vector<Particle>& _particles) {
	dimX = _dimX;
	dimY = _dimY;

	particles = _particles;

	grid.clear();
	grid.resize(dimX * dimY);
}

void ParticleGrid::Add(int particleID) {
	int index = GetGridCellIndexFromParticleIndex(particleID);

	//std::cout << "inserting " << particleID << " in cell " << index << std::endl;

	particles[particleID].gridCellID = index;

	grid[index].insert(particleID);
}

void ParticleGrid::Update() {
	Clear();

	for (Particle& p : particles) {
		Add(p.id);
	}
}

void ParticleGrid::Clear() {
	for (int i = 0; i < dimX * dimY; i++) {
		grid[i].clear();
	}
}

int ParticleGrid::GetGridCellIndexFromParticleIndex(int particleID) {
	int x = particles[particleID].position.x / (2.f * H);
	int y = particles[particleID].position.y / (2.f * H);

	int index = Index2Dto1D(x, y);

	if (index > grid.size()) {
		std::cout << "OJOJ" << std::endl;
	}

	return Index2Dto1D(x, y);
}

void ParticleGrid::GetNeighbourParticlesIndices(int particleIndex, std::vector<int>& indices) {
	int gridIndex = particles[particleIndex].gridCellID;
	std::vector<int> neighbourCells = GetNeighbourCellIndices(gridIndex);

	indices.clear();

	for (int neighbourIndex : neighbourCells) {
		for (int particleID : grid[neighbourIndex]) {
			indices.push_back(particleID);
		}
	}
}

std::vector<int> ParticleGrid::GetNeighbourCellIndices(int index) {
	std::vector<int> neighbours;
	std::pair<int, int> index2D = Index1Dto2D(index);

	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			int newX = index2D.first + x;
			int newY = index2D.second + y;

			if (newX >= 0 && newX < dimX && newY >= 0 && newY < dimY) {
				neighbours.push_back(Index2Dto1D(newX, newY));
			}
		}
	}

	return neighbours;
}