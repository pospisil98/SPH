#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include "MyVec2.h"

static int WINDOW_WIDTH = 1280;
static int WINDOW_HEIGHT = 720;
static double VIEW_WIDTH = 1.5f * WINDOW_WIDTH;
static double VIEW_HEIGHT = 1.5f * WINDOW_HEIGHT;

static int DAM_BREAK_PARTICLES = 100;
static int MAX_PARTICLES = 5000;
static int BLOCK_PARTICLES = 400;

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
static float GRAVITY_VAL = 9.81f;
static MyVec2 G(0.f, -GRAVITY_VAL);	// external (gravitational) forces
static float H = 16.f;					// kernel radius
static float HSQ = H * H;				// radius^2 for optimization
static float DT = 0.0007f;				// integration timestep

static float EPS = H;

// smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
static float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));