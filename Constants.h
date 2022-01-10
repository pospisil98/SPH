#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include "MyVec2.h"

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
#define GRAVITY_VAL 9.81f
#define H 16.f					// kernel radius
#define HSQ H * H			// radius^2 for optimization
#define DT  0.0007f				// integration timestep

#define EPS H

// smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
#define POLY6 (4.f / (M_PI * pow(H, 8.f)))
#define SPIKY_GRAD (-10.f / (M_PI * pow(H, 5.f)))
#define VISC_LAP  (40.f / (M_PI * pow(H, 5.f)))