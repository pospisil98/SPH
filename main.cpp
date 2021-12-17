#include <GL/glut.h>

#define _USE_MATH_DEFINES
#include <math.h>
	
#include <vector>
#include <chrono>
#include <iostream>


struct Particle;

struct Vec2 {
	float x;
	float y;

	Vec2(float _x, float _y) :  x(_x), y(_y) { }

	inline Vec2& operator = (const Vec2& v) { x = v.x; y = v.y; return *this; }
	inline Vec2& operator = (const float& f) { x = f; y = f; return *this; }
	inline Vec2& operator - (void) { x = -x; y = -y; return *this; }
	inline bool operator == (const Vec2& v) const { return (x == v.x) && (y == v.y); }
	inline bool operator != (const Vec2& v) const { return (x != v.x) || (y != v.y); }

	inline const Vec2 operator + (const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
	inline const Vec2 operator - (const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
	inline const Vec2 operator * (const Vec2& v) const { return Vec2(x * v.x, y * v.y); }
	inline const Vec2 operator / (const Vec2& v) const { return Vec2(x / v.x, y / v.y); }

	inline Vec2& operator += (const Vec2& v) { x += v.x; y += v.y; return *this; }
	inline Vec2& operator -= (const Vec2& v) { x -= v.x; y -= v.y; return *this; }
	inline Vec2& operator *= (const Vec2& v) { x *= v.x; y *= v.y; return *this; }
	inline Vec2& operator /= (const Vec2& v) { x /= v.x; y /= v.y; return *this; }

	inline const Vec2 operator + (float v) const { return Vec2(x + v, y + v); }
	inline const Vec2 operator - (float v) const { return Vec2(x - v, y - v); }
	inline const Vec2 operator * (float v) const { return Vec2(x * v, y * v); }
	inline const Vec2 operator / (float v) const { return Vec2(x / v, y / v); }

	inline Vec2& operator += (float v) { x += v; y += v; return *this; }
	inline Vec2& operator -= (float v) { x -= v; y -= v; return *this; }
	inline Vec2& operator *= (float v) { x *= v; y *= v; return *this; }
	inline Vec2& operator /= (float v) { x /= v; y /= v; return *this; }

	float Length() const { return sqrt(x * x + y * y); }
	float LengthSquared() const { return x * x + y * y; }
	float Distance(const Vec2& v) const { return sqrt(((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y))); }
	float DistanceSquared(const Vec2& v) const { return ((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y)); }
	float Dot(const Vec2& v) const { return x * v.x + y * v.y; }
	float Cross(const Vec2& v) const { return x * v.y + y * v.x; }

	Vec2 Normalized() {
		float l = Length();
		if (l != 0.0f) {
			return Vec2(x /= l, y /= l);
		}
		return Vec2(0, 0);
	}

	Vec2& Normalize() {
		float l = Length();
		if (l != 0.0f) {
			x /= l;
			y /= l;
		}
		return *this;
	}

};

template <typename T>
Vec2 operator*(T scalar, Vec2 const& vec) {
	return vec * scalar;
}

std::ostream& operator<<(std::ostream& os, Vec2& v) {
	return os << "< " << v.x << ", " << v.y << ">" << std::endl;
}

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const double VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const double VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;

const int PARTICLES = 1000;
const int MAX_PARTICLES = 5000;
const int BLOCK_PARTICLES = 250;

std::chrono::high_resolution_clock::time_point lastUpdate;

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
const static Vec2 G(0.f, -10.f);   // external (gravitational) forces
const static float REST_DENS = 300.f;  // rest density
const static float GAS_CONST = 2000.f; // const for equation of state
const static float H = 16.f;		   // kernel radius
const static float HSQ = H * H;		   // radius^2 for optimization
const static float MASS = 2.5f;		   // assume all particles have the same mass
const static float VISC = 200.f;	   // viscosity constant
const static float DT = 0.0007f;	   // integration timestep

const float EPS = H;
const static float BOUND_DAMPING = -0.5f;

// smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
const static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
const static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
const static float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));

std::vector<Particle> particles;

struct Particle {
	Vec2 position;
	Vec2 velocity;
	Vec2 force;

	float rho;
	float p;

	Particle(float _x, float _y) :
		position(_x, _y),
		velocity(0.f, 0.f),
		force(0.f, 0.f),
		rho(0),
		p(0.f)
	{ }
};

void InitSPH() {
	std::cout << "Init dam break with " << PARTICLES << " particles" << std::endl;
	for (float y = EPS; y < VIEW_HEIGHT - EPS * 2.f; y += H) {
		for (float x = VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H) {
			if (particles.size() < PARTICLES) {
				float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				particles.emplace_back(x + jitter, y);
			} else {
				return;
			}
		}
	}
}

void Integrate(float deltaTime = DT) {
	for (Particle& p : particles) {
		// forward Euler integration
		p.velocity += deltaTime * p.force / p.rho;
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
		for (Particle& pj : particles) {
			Vec2 rij = pj.position - pi.position;
			float r2 = rij.LengthSquared();

			if (r2 < HSQ) {
				pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.p = GAS_CONST * (pi.rho - REST_DENS);
	}
}

void ComputeForces() {
	for (Particle& pi : particles) {
		Vec2 fpress(0.f, 0.f);
		Vec2 fvisc(0.f, 0.f);
		for (Particle& pj : particles) {
			if (&pi == &pj) {
				continue;
			}

			Vec2 rij = pj.position - pi.position;
			float r = rij.Length();

			if (r < H) {
				// compute pressure force contribution
				fpress += -rij.Normalized() * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
				// compute viscosity force contribution
				fvisc += VISC * MASS * (pj.velocity - pi.velocity) / pj.rho * VISC_LAP * (H - r);
			}
		}
		Vec2 fgrav = G * MASS / pi.rho;
		pi.force = fpress + fvisc + fgrav;
	}
}

void Update() {
	ComputeDensityPressure();
	ComputeForces();

	auto now = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> deltaTime = std::chrono::duration_cast<std::chrono::duration<float>>(now - lastUpdate);
	lastUpdate = now;

	Integrate();

	glutPostRedisplay();
}

void InitGL() {
	glClearColor(0.9f, 0.9f, 0.9f, 1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(H / 2.f);
	glMatrixMode(GL_PROJECTION);
}

void Render() {
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);

	glColor4f(0.2f, 0.6f, 1.f, 1);
	glBegin(GL_POINTS);

	for (Particle& p : particles) {
		glVertex2f(p.position.x, p.position.y);
	}
	
	glEnd();

	glutSwapBuffers();
}

void Keyboard(unsigned char c, int x, int y) {
	switch (c)
	{
		case ' ':
			if (particles.size() >= MAX_PARTICLES) {
				std::cout << "maximum number of particles reached" << std::endl;
			} else {
				unsigned int placed = 0;
				for (float y = VIEW_HEIGHT / 1.5f - VIEW_HEIGHT / 5.f; y < VIEW_HEIGHT / 1.5f + VIEW_HEIGHT / 5.f; y += H * 0.95f) {
					for (float x = VIEW_WIDTH / 2.f - VIEW_HEIGHT / 5.f; x <= VIEW_WIDTH / 2.f + VIEW_HEIGHT / 5.f; x += H * 0.95f) {
						if (placed++ < BLOCK_PARTICLES && particles.size() < MAX_PARTICLES) {
							particles.push_back(Particle(x, y));
						}
					}
				}
			}
			break;
		case 'r':
		case 'R':
			particles.clear();
			InitSPH();
			break;
	}
}

int main(int argc, char** argv)
{
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInit(&argc, argv);

	glutCreateWindow("Smoothed Particle Hydrodynamics");
	
	glutDisplayFunc(Render);
	glutIdleFunc(Update);
	glutKeyboardFunc(Keyboard);

	InitGL();
	InitSPH();

	glutMainLoop();
	return 0;
}