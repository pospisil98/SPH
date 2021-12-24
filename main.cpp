/*
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
		p(0.0f)
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
		glColor4f(0.2f, 0.6f, p.p, 1);
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
*/

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

#include <iostream>

#define PIXEL_FORMAT GL_RGB

// reference from 
// https://gist.github.com/victusfate/9214902
// https://nervous.io/ffmpeg/opengl/2017/01/31/ffmpeg-opengl/

static const GLchar* v_shader_source =
"attribute vec2 position;\n"
"varying vec2 texCoord;\n"
"void main(void) {\n"
"  gl_Position = vec4(position, 0, 1);\n"
"  texCoord = position;\n"
"}\n";

static const GLchar* f_shader_source =
"uniform sampler2D tex;\n"
"varying vec2 texCoord;\n"
"void main() {\n"
"  gl_FragColor = texture2D(tex, texCoord * 0.5 + 0.5);\n"
"}\n";
/*
typedef struct {
	const AVClass *class;
	GLuint        program;
	GLuint        frame_tex;
	GLFWwindow    *window;
	GLuint        pos_buf;
} GenericShaderContext
*/
static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

int main(void)
{
	GLFWwindow* window;

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
	window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, key_callback);


	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Our state
	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


	// Loop until the user closes the window 
	while (!glfwWindowShouldClose(window))
	{
		float ratio;
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		ratio = width / (float)height;
		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glRotatef((float)glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
		glBegin(GL_TRIANGLES);
		glColor3f(1.f, 0.f, 0.f);
		glVertex3f(-0.6f, -0.4f, 0.f);
		glColor3f(0.f, 1.f, 0.f);
		glVertex3f(0.6f, -0.4f, 0.f);
		glColor3f(0.f, 0.f, 1.f);
		glVertex3f(0.f, 0.6f, 0.f);
		glEnd();



		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		static float f = 0.0f;
		static int counter = 0;

		ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

		ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
		ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
		ImGui::Checkbox("Another Window", &show_another_window);

		ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
		ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

		if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
			counter++;
		ImGui::SameLine();
		ImGui::Text("counter = %d", counter);

		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::End();


		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// ImGUI Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();

	return 0;
}
