#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include "MyVec2.h"

static int WINDOW_WIDTH = 1280;
static int WINDOW_HEIGHT = 720;
static double VIEW_WIDTH = 1.5f * WINDOW_WIDTH;
static double VIEW_HEIGHT = 1.5f * WINDOW_HEIGHT;

static const int PARTICLES = 100;
static const int MAX_PARTICLES = 5000;
static const int BLOCK_PARTICLES = 400;

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
static float GRAVITY_VAL = 9.81f;
static MyVec2 G(0.f, -GRAVITY_VAL);	// external (gravitational) forces
static float REST_DENS = 300.f;		// rest density
static float GAS_CONST = 2000.f;		// const for equation of state
static float H = 16.f;					// kernel radius
static float HSQ = H * H;				// radius^2 for optimization
static float MASS = 2.5f;				// assume all particles have the same mass
static float VISC = 200.f;				// viscosity constant
static float DT = 0.0007f;				// integration timestep

static float EPS = H;
static float BOUND_DAMPING = -0.5f;

// smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
static 
float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));