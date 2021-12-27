// Include standard headers
#include <stdio.h>
#include <stdlib.h>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
using namespace glm;

// Include ImGUI
#include "GL/imgui/imgui.h"
#include "GL/imgui/backends/imgui_impl_glfw.h"
#include "GL/imgui/backends/imgui_impl_opengl3.h"


#define _USE_MATH_DEFINES
#include <math.h>

#include <vector>
#include <chrono>
#include <iostream>
#include <unordered_set>

struct MyVec2 {
	float x;
	float y;

	MyVec2(float _x, float _y) : x(_x), y(_y) { }

	inline MyVec2& operator = (const MyVec2& v) { x = v.x; y = v.y; return *this; }
	inline MyVec2& operator = (const float& f) { x = f; y = f; return *this; }
	inline MyVec2& operator - (void) { x = -x; y = -y; return *this; }
	inline bool operator == (const MyVec2& v) const { return (x == v.x) && (y == v.y); }
	inline bool operator != (const MyVec2& v) const { return (x != v.x) || (y != v.y); }

	inline const MyVec2 operator + (const MyVec2& v) const { return MyVec2(x + v.x, y + v.y); }
	inline const MyVec2 operator - (const MyVec2& v) const { return MyVec2(x - v.x, y - v.y); }
	inline const MyVec2 operator * (const MyVec2& v) const { return MyVec2(x * v.x, y * v.y); }
	inline const MyVec2 operator / (const MyVec2& v) const { return MyVec2(x / v.x, y / v.y); }

	inline MyVec2& operator += (const MyVec2& v) { x += v.x; y += v.y; return *this; }
	inline MyVec2& operator -= (const MyVec2& v) { x -= v.x; y -= v.y; return *this; }
	inline MyVec2& operator *= (const MyVec2& v) { x *= v.x; y *= v.y; return *this; }
	inline MyVec2& operator /= (const MyVec2& v) { x /= v.x; y /= v.y; return *this; }

	inline const MyVec2 operator + (float v) const { return MyVec2(x + v, y + v); }
	inline const MyVec2 operator - (float v) const { return MyVec2(x - v, y - v); }
	inline const MyVec2 operator * (float v) const { return MyVec2(x * v, y * v); }
	inline const MyVec2 operator / (float v) const { return MyVec2(x / v, y / v); }

	inline MyVec2& operator += (float v) { x += v; y += v; return *this; }
	inline MyVec2& operator -= (float v) { x -= v; y -= v; return *this; }
	inline MyVec2& operator *= (float v) { x *= v; y *= v; return *this; }
	inline MyVec2& operator /= (float v) { x /= v; y /= v; return *this; }

	float Length() const { return sqrt(x * x + y * y); }
	float LengthSquared() const { return x * x + y * y; }
	float Distance(const MyVec2& v) const { return sqrt(((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y))); }
	float DistanceSquared(const MyVec2& v) const { return ((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y)); }
	float Dot(const MyVec2& v) const { return x * v.x + y * v.y; }
	float Cross(const MyVec2& v) const { return x * v.y + y * v.x; }

	MyVec2 Normalized() {
		float l = Length();
		if (l != 0.0f) {
			return MyVec2(x /= l, y /= l);
		}
		return MyVec2(0, 0);
	}

	MyVec2& Normalize() {
		float l = Length();
		if (l != 0.0f) {
			x /= l;
			y /= l;
		}
		return *this;
	}

};

template <typename T>
MyVec2 operator*(T scalar, MyVec2 const& vec) {
	return vec * scalar;
}

std::ostream& operator<<(std::ostream& os, MyVec2& v) {
	return os << "< " << v.x << ", " << v.y << ">" << std::endl;
}

#define PIXEL_FORMAT GL_RGB

int WINDOW_WIDTH = 1280;
int WINDOW_HEIGHT = 720;
double VIEW_WIDTH = 1.5f * WINDOW_WIDTH;
double VIEW_HEIGHT = 1.5f * WINDOW_HEIGHT;

const int PARTICLES = 100;
const int MAX_PARTICLES = 5000;
const int BLOCK_PARTICLES = 400;

bool accelerateCPU = true;

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
float GRAVITY_VAL = 9.81f;
MyVec2 G(0.f, -GRAVITY_VAL);	// external (gravitational) forces
float REST_DENS = 300.f;		// rest density
float GAS_CONST = 2000.f;		// const for equation of state
float H = 16.f;					// kernel radius
float HSQ = H * H;				// radius^2 for optimization
float MASS = 2.5f;				// assume all particles have the same mass
float VISC = 200.f;				// viscosity constant
float DT = 0.0007f;				// integration timestep

float EPS = H;
float BOUND_DAMPING = -0.5f;

struct Particle {
	MyVec2 position;
	MyVec2 velocity;
	MyVec2 force;

	float rho;
	float p;

	int id;
	int gridCellID;

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

std::vector<Particle> particles;

struct ParticleGrid {
	std::vector<std::unordered_set<int>> grid;

	int dimX;
	int dimY;

	ParticleGrid() {
		dimX = -1;
		dimY = -1;
	}

	ParticleGrid(int _dimX, int _dimY) :
		dimX(_dimX),
		dimY(_dimY)
	{
		grid.resize(dimX * dimY);
	}

	void Add(int particleID) {
		/*
		if (particleID == 0) {
			std::cout << "BREAK" << std::endl;
		}
		*/
		
		int index = GetGridCellIndexFromParticleIndex(particleID);

		//std::cout << "inserting " << particleID << " in cell " << index << std::endl;

		particles[particleID].gridCellID = index;

		grid[index].insert(particleID);
	}

	void Update() {
		Clear();

		for (Particle& p : particles) {
			Add(p.id);
		}
	}

	void Clear() {
		for (int i = 0; i < dimX * dimY; i++) {
			grid[i].clear();
		}
	}

	int GetGridCellIndexFromParticleIndex(int particleID) {
		int x = particles[particleID].position.x / (2.f * H);
		int y = particles[particleID].position.y / (2.f * H);

		int index = Index2Dto1D(x, y);

		if (index > grid.size()) {
			std::cout << "OJOJ" << std::endl;
		}

		return Index2Dto1D(x, y);
	}

	void GetNeighbourParticlesIndices(int particleIndex, std::vector<int>& indices) {
		int gridIndex = particles[particleIndex].gridCellID;
		std::vector<int> neighbourCells = GetNeighbourCellIndices(gridIndex);

		/*
		for (int i = 0; i < neighbourCells.size(); i++) {
			std::pair<int, int> p = Index1Dto2D(neighbourCells[i]);
			std::cout << p.first << ", " << p.second << std::endl;
		}
		*/

		indices.clear();

		for (int neighbourIndex : neighbourCells) {
			for (int particleID : grid[neighbourIndex]) {
				indices.push_back(particleID);
			}
		}
	}

	std::vector<int> GetNeighbourCellIndices(int index) {
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

	int Index2Dto1D(int x, int y) {
		return x + y * dimX;
	}

	std::pair<int, int> Index1Dto2D(int index) {
		int x = index % dimX;
		int y = index / dimX;

		return std::make_pair(x, y);
	}
};

ParticleGrid particleGrid;

// smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
const static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
const static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
const static float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));

static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

double  t;
double  t_old = 0.f;
double  dt;

ImVec4 clearColor = ImVec4(0.45f, 0.55f, 0.60f, 1.0f);
ImVec4 particlesColor = ImVec4(0.f, 0.f, 0.f, 1.0f);

void InitSPH() {
	std::cout << "Init dam break with " << PARTICLES << " particles" << std::endl;

	for (float y = EPS; y < VIEW_HEIGHT - EPS * 2.f; y += H) {
		for (float x = VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H) {
			if (particles.size() < PARTICLES) {
				float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				particles.emplace_back(x + jitter, y, particles.size());
			}
			else {
				return;
			}
		}
	}
}

void GetNeighbourParticlesIndicesTrivial(std::vector<int>& indices) {
	indices.clear();

	for (int i = 0; i < particles.size(); i++) {
		indices.push_back(i);
	}
}

void Integrate(float deltaTime) {
	for (Particle& p : particles) {
		// forward Euler integration
		if (p.rho > 0.0f) {
			p.velocity += deltaTime * p.force / p.rho;
		}
		p.position += deltaTime * p.velocity;

		// enforce boundary conditions
		if (p.position.x - EPS < 0.f) {
			p.velocity.x *= BOUND_DAMPING;
			p.position.x = EPS;
		}

		if (p.position.x + EPS > VIEW_WIDTH) {
			p.velocity.x *= BOUND_DAMPING;
			p.position.x = VIEW_WIDTH - EPS;
		}

		if (p.position.y - EPS < 0.f) {
			p.velocity.y *= BOUND_DAMPING;
			p.position.y = EPS;
		};

		if (p.position.y + EPS > VIEW_HEIGHT) {
			p.velocity.y *= BOUND_DAMPING;
			p.position.y = VIEW_HEIGHT - EPS;
		}
	}
}

void ComputeDensityPressure() {
	for (Particle& pi : particles) {
		pi.rho = 0.f;

		std::vector<int> potentialNeighbours;
		if (accelerateCPU) {
			particleGrid.GetNeighbourParticlesIndices(pi.id, potentialNeighbours);

			/*
			if (potentialNeighbours.size() > 0) {
				std::cout << "WE HAVE SOME NEIGHBOURS"<< std::endl;
			}
			*/
		} else {
			GetNeighbourParticlesIndicesTrivial(potentialNeighbours);
		}

		for (int index : potentialNeighbours) {
			Particle pj = particles[index];

			MyVec2 rij = pj.position - pi.position;
			float r2 = rij.LengthSquared();

			if (r2 < HSQ) {
				//std::cout << "Collision in Density for id: " << pi.id << " and " << pj.id << std::endl;
				pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.p = GAS_CONST * (pi.rho - REST_DENS);
	}
}

void ComputeForces() {
	for (Particle& pi : particles) {
		MyVec2 fpress(0.f, 0.f);
		MyVec2 fvisc(0.f, 0.f);

		std::vector<int> potentialNeighbours;
		if (accelerateCPU) {
			particleGrid.GetNeighbourParticlesIndices(pi.id, potentialNeighbours);
			
			/*
			if (potentialNeighbours.size() > 0) {
				std::cout << "WE HAVE SOME NEIGHBOURS" << std::endl;
			}
			*/
		}
		else {
			GetNeighbourParticlesIndicesTrivial(potentialNeighbours);
		}

		for (int index : potentialNeighbours) {
			Particle pj = particles[index];
			if (pi.id == pj.id) {
				continue;
			}

			MyVec2 rij = pj.position - pi.position;
			float r = rij.Length();

			if (r < H) {
				//std::cout << "Collision in forces for id: " << pi.id << " and " << pj.id << std::endl;
				// compute pressure force contribution
				fpress += -rij.Normalized() * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
				// compute viscosity force contribution
				fvisc += VISC * MASS * (pj.velocity - pi.velocity) / pj.rho * VISC_LAP * (H - r);
			}
		}
		MyVec2 fgrav = G * MASS / pi.rho;
		pi.force = fpress + fvisc + fgrav;
	}
}

void Update(float deltaTime) {
	if (accelerateCPU) {
		particleGrid.Update();

		/*
		for (int i = 0; i < particleGrid.grid.size(); i++) {
			if (particleGrid.grid[i].size() > 0) {
				std::pair<int, int> pos = particleGrid.Index1Dto2D(i);
				std::cout << "Grid cell " << i << "(" << pos.first << ", " << pos.second << ")" << "has " << particleGrid.grid[i].size() << std::endl;
			}
		}
		*/
	}

	ComputeDensityPressure();
	ComputeForces();

	Integrate(deltaTime);
}

void Render(GLFWwindow* window) {
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);

	glColor4f(
		particlesColor.x * particlesColor.w,
		particlesColor.y * particlesColor.w,
		particlesColor.z * particlesColor.w,
		particlesColor.w
	);
	glBegin(GL_POINTS);

	for (Particle& p : particles) {
		glVertex2f(p.position.x, p.position.y);
	}

	glEnd();
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		if (particles.size() >= MAX_PARTICLES) {
			std::cout << "maximum number of particles reached" << std::endl;
		}
		else {
			unsigned int placed = 0;
			for (float y = VIEW_HEIGHT / 1.5f - VIEW_HEIGHT / 5.f; y < VIEW_HEIGHT / 1.5f + VIEW_HEIGHT / 5.f; y += H * 0.95f) {
				for (float x = VIEW_WIDTH / 2.f - VIEW_HEIGHT / 5.f; x <= VIEW_WIDTH / 2.f + VIEW_HEIGHT / 5.f; x += H * 0.95f) {
					if (placed++ < BLOCK_PARTICLES && particles.size() < MAX_PARTICLES) {
						particles.push_back(Particle(x, y, particles.size()));
					}
				}
			}
		}
	}
	else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		particles.clear();
		InitSPH();
	}
	else if (key == GLFW_KEY_W && action == GLFW_PRESS) {
		G.x = 0.0f;
		G.y = GRAVITY_VAL;
	}
	else if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		G.x = 0.0f;
		G.y = -GRAVITY_VAL;
	}
	else if (key == GLFW_KEY_A && action == GLFW_PRESS) {
		G.x = -GRAVITY_VAL;
		G.y = 0.0f;
	}
	else if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		G.x = GRAVITY_VAL;
		G.y = 0.0f;
	} else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
	// Update OpenGL viewport
	int fbWidth, fbHeight;
	glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
	glViewport(0, 0, fbWidth, fbHeight);

	int ogWidth = VIEW_WIDTH;
	int ogHeight = VIEW_HEIGHT;

	// Update window properties
	WINDOW_WIDTH = width * 1.5f;
	WINDOW_HEIGHT = height * 1.5f;
	VIEW_WIDTH = 1.5f * WINDOW_WIDTH;
	VIEW_HEIGHT = 1.5f * WINDOW_HEIGHT;

	// rescale positions of particles
	for (Particle& p : particles) {
		p.position.x = (p.position.x * VIEW_WIDTH) / ogWidth;
		p.position.y = (p.position.y * VIEW_HEIGHT) / ogHeight;
	}

	// Update uniform grid
	int dimX = (VIEW_WIDTH + EPS) / (2.f * H);
	int dimY = (VIEW_HEIGHT + EPS) / (2.f * H);
	particleGrid = ParticleGrid(dimX, dimY);
	if (accelerateCPU) {
		particleGrid.Update();
	}
}

void InitGL() {
	glClearColor(0.9f, 0.9f, 0.9f, 1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(H / 2.f);
	glMatrixMode(GL_PROJECTION);
}

void SetDefaultParameters() {
	GRAVITY_VAL = 9.81f;
	G.x = 0.f;
	G.y = -GRAVITY_VAL;	
	REST_DENS = 300.f;		
	GAS_CONST = 2000.f;		
	H = 16.f;					
	HSQ = H * H;				
	MASS = 2.5f;				
	VISC = 200.f;				
	DT = 0.0007f;

	EPS = H;
	BOUND_DAMPING = -0.5f;
}

int main(void)
{
	glfwSetErrorCallback(error_callback);

	// Initialize the library
	if (!glfwInit())
		return -1;

	// Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
	// GL ES 2.0 + GLSL 100
	const char* glsl_version = "#version 100";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

	// Create a windowed mode window and its OpenGL context 
	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Smoothed Particle Hydrodynamics", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, key_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);

	InitGL();


	// ImGui setup
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Dear ImGui style
		//ImGui::StyleColorsDark();
		ImGui::StyleColorsClassic();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);
	}

	// Our state
	bool fixedTimestep = true;


	InitSPH();

	// Do grid initialization everytime to be able to witch acceleration on and off
	int dimX = (VIEW_WIDTH + EPS) / (2.f * H);
	int dimY = (VIEW_HEIGHT + EPS) / (2.f * H);
	particleGrid = ParticleGrid(dimX, dimY);


	// Loop until the user closes the window 
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		t = glfwGetTime();
		dt = t - t_old;
		t_old = t;
		
		if (fixedTimestep) {
			Update(DT);
		} else {
			Update(dt);
		}
		
		Render(window);

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		static float f = 0.0f;
		static int counter = 0;

		ImGui::Begin("SPH settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

		ImGui::Checkbox("FixedTimestep", &fixedTimestep);
		if (fixedTimestep) {
			ImGui::InputFloat("Delta Time", &DT);
		}

		ImGui::SliderFloat("Bounds damping", &BOUND_DAMPING, -1.5f, 0);
		ImGui::SliderFloat("Viscosity", &VISC, 10.0f, 500.0f);
		ImGui::SliderFloat("Particle mass", &MASS, 0.5f, 20.0f);
		ImGui::SliderFloat("Rest density", &REST_DENS, 10.0f, 500.0f);
		ImGui::SliderFloat("Gas constant", &GAS_CONST, 500.0f, 6000.0f);

		if (ImGui::Button("Reset parameters to default")) {
			SetDefaultParameters();
		}

		ImGui::Checkbox("Accelerate CPU version with uniform grid", &accelerateCPU);

		ImGui::ColorEdit3("Background color", (float*)&clearColor); 
		ImGui::ColorEdit3("Particles color", (float*)&particlesColor); 

		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::End();


		// Rendering
		ImGui::Render();
		glClearColor(clearColor.x * clearColor.w, clearColor.y * clearColor.w, clearColor.z * clearColor.w, clearColor.w);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	// ImGUI Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();

	return 0;
}
