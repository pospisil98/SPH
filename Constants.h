#pragma once

#define _USE_MATH_DEFINES
#include <math.h>


/// <summary> Value of gravity force </summary>
#define GRAVITY_VAL		9.81f

/// <summary> Kernel radius in which we check interactions of particles </summary>
#define H				16.f

/// <summary> Kernel radius squared - optimization reasons </summary>
#define HSQ				H * H			

/// <summary> Integration timestep </summary>
#define DT				0.0007f

/// <summary> Epsilon for usage in boundary collision checks </summary>
#define EPS				H


// Smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.

#define POLY6			(4.f / (M_PI * pow(H, 8.f)))

#define SPIKY_GRAD		(-10.f / (M_PI * pow(H, 5.f)))

#define VISC_LAP		(40.f / (M_PI * pow(H, 5.f)))