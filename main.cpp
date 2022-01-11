// Include CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"
#include "CudaFunctions.cuh"

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

#include "Constants.h"
#include "MyVec2.h"
#include "Particle.h"
#include "ParticleGrid.h"
#include "Simulation.h"


#define PIXEL_FORMAT GL_RGB

static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

Simulation simulation;

double  t;
double  t_old = 0.f;
double  dt;

ImVec4 clearColor = ImVec4(0.45f, 0.55f, 0.60f, 1.0f);
ImVec4 particlesColor = ImVec4(0.f, 0.f, 0.f, 1.0f);

void Render(GLFWwindow* window) {
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glOrtho(0, simulation.VIEW_WIDTH, 0, simulation.VIEW_HEIGHT, 0, 1);

	glColor4f(
		particlesColor.x * particlesColor.w,
		particlesColor.y * particlesColor.w,
		particlesColor.z * particlesColor.w,
		particlesColor.w
	);

	glBegin(GL_POINTS);

	for (int i = 0; i < simulation.particleCount; i++) {
		glVertex2f(simulation.particles[i].position.x, simulation.particles[i].position.y);
	}

	glEnd();
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		simulation.AddParticleRectangle();
	}
	else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		simulation.Reset();
	}
	else if (key == GLFW_KEY_W && action == GLFW_PRESS) {
		simulation.G.x = 0.0f;
		simulation.G.y = GRAVITY_VAL;
	}
	else if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		simulation.G.x = 0.0f;
		simulation.G.y = -GRAVITY_VAL;
	}
	else if (key == GLFW_KEY_A && action == GLFW_PRESS) {
		simulation.G.x = -GRAVITY_VAL;
		simulation.G.y = 0.0f;
	}
	else if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		simulation.G.x = GRAVITY_VAL;
		simulation.G.y = 0.0f;
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

	simulation.windowRescaleRoutine(width, height);
}

void InitGL() {
	glClearColor(0.9f, 0.9f, 0.9f, 1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(H / 2.f);
	glMatrixMode(GL_PROJECTION);
}

void RenderGUIControlPanel() {
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("SPH settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

	ImGui::Checkbox("FixedTimestep", &simulation.fixedTimestep);
	if (simulation.fixedTimestep) {
		//ImGui::InputFloat("Delta Time", &DT);
	}

	ImGui::Spacing();
	ImGui::SliderInt("Maximum particles", &simulation.MAX_PARTICLES, 0, 50000);
	ImGui::SliderFloat("Bounds damping", &simulation.BOUND_DAMPING, -1.5f, 0);
	ImGui::SliderFloat("Viscosity", &simulation.VISC, 10.0f, 500.0f);
	ImGui::SliderFloat("Particle mass", &simulation.MASS, 0.5f, 20.0f);
	ImGui::SliderFloat("Rest density", &simulation.REST_DENS, 10.0f, 500.0f);
	ImGui::SliderFloat("Gas constant", &simulation.GAS_CONST, 500.0f, 6000.0f);

	ImGui::Spacing();
	ImGui::Text("Gravity: (%f, %f)", simulation.G.x, simulation.G.y);
	if (ImGui::Button("Toggle Gravity")) {
		if (simulation.G.x != 0.0f || simulation.G.y != 0.0f) {
			simulation.G = MyVec2(0.0f, 0.0f);
		} else {
			simulation.G = MyVec2(0.0f, -GRAVITY_VAL);
		}
	}
	if (ImGui::Button("Reset parameters to default")) {
		simulation.SetDefaultParameters();
	}

	ImGui::Spacing();
	ImGui::Checkbox("Accelerate CPU version with uniform grid", &simulation.useSpatialGrid);
	ImGui::Checkbox("Use GPU for computations", &simulation.simulateOnGPU);

	ImGui::Spacing();
	ImGui::ColorEdit3("Background color", (float*)&clearColor);
	ImGui::ColorEdit3("Particles color", (float*)&particlesColor);

	ImGui::Spacing();
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Text("Particle count: %d", simulation.particleCount);
	ImGui::End();


	// Rendering
	ImGui::Render();
	glClearColor(clearColor.x * clearColor.w, clearColor.y * clearColor.w, clearColor.z * clearColor.w, clearColor.w);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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
	GLFWwindow* window = glfwCreateWindow(simulation.WINDOW_WIDTH, simulation.WINDOW_HEIGHT, "Smoothed Particle Hydrodynamics", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, key_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);

	// TODO: unlimited FPS / cap to screen refresh rate
	glfwSwapInterval(0);

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

	// Initialize SPH simulation
	//simulation.Initialize();
	simulation.InitSPH();

	// Test ability to map host memory (from some page-lock tutorial)
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	if (!deviceProp.canMapHostMemory) {
		fprintf(stderr, "Device %d cannot map host memory!\n", 0);
		exit(EXIT_FAILURE);
	}

	// Loop until the user closes the window 
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		t = glfwGetTime();
		dt = t - t_old;
		t_old = t;
		
		simulation.Update(dt);
		
		Render(window);
		RenderGUIControlPanel();

		glfwSwapBuffers(window);
	}

	// ImGUI Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();

	return 0;
}
