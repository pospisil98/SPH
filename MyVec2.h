#pragma once

// Include CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ostream>
#include <math.h>

struct MyVec2 {
	float x;
	float y;

	__host__ __device__ MyVec2() : x(0.0f), y(0.0f) { }
	__host__ __device__ MyVec2(float _x, float _y) : x(_x), y(_y) { }

	inline __host__ __device__ MyVec2& operator = (const MyVec2& v) { x = v.x; y = v.y; return *this; }
	inline __host__ __device__ MyVec2& operator = (const float& f) { x = f; y = f; return *this; }
	inline __host__ __device__ MyVec2& operator - (void) { x = -x; y = -y; return *this; }
	inline __host__ __device__ bool operator == (const MyVec2& v) const { return (x == v.x) && (y == v.y); }
	inline __host__ __device__ bool operator != (const MyVec2& v) const { return (x != v.x) || (y != v.y); }

	inline __host__ __device__ const MyVec2 operator + (const MyVec2& v) const { return MyVec2(x + v.x, y + v.y); }
	inline __host__ __device__ const MyVec2 operator - (const MyVec2& v) const { return MyVec2(x - v.x, y - v.y); }
	inline __host__ __device__ const MyVec2 operator * (const MyVec2& v) const { return MyVec2(x * v.x, y * v.y); }
	inline __host__ __device__ const MyVec2 operator / (const MyVec2& v) const { return MyVec2(x / v.x, y / v.y); }

	inline __host__ __device__ MyVec2& operator += (const MyVec2& v) { x += v.x; y += v.y; return *this; }
	inline __host__ __device__ MyVec2& operator -= (const MyVec2& v) { x -= v.x; y -= v.y; return *this; }
	inline __host__ __device__ MyVec2& operator *= (const MyVec2& v) { x *= v.x; y *= v.y; return *this; }
	inline __host__ __device__ MyVec2& operator /= (const MyVec2& v) { x /= v.x; y /= v.y; return *this; }

	inline __host__ __device__ const MyVec2 operator + (float v) const { return MyVec2(x + v, y + v); }
	inline __host__ __device__ const MyVec2 operator - (float v) const { return MyVec2(x - v, y - v); }
	inline __host__ __device__ const MyVec2 operator * (float v) const { return MyVec2(x * v, y * v); }
	inline __host__ __device__ const MyVec2 operator / (float v) const { return MyVec2(x / v, y / v); }

	inline __host__ __device__ MyVec2& operator += (float v) { x += v; y += v; return *this; }
	inline __host__ __device__ MyVec2& operator -= (float v) { x -= v; y -= v; return *this; }
	inline __host__ __device__ MyVec2& operator *= (float v) { x *= v; y *= v; return *this; }
	inline __host__ __device__ MyVec2& operator /= (float v) { x /= v; y /= v; return *this; }

	inline __host__ __device__ float Length() const { return sqrt(x * x + y * y); }
	inline __host__ __device__ float LengthSquared() const { return x * x + y * y; }
	inline __host__ __device__ float Distance(const MyVec2& v) const { return sqrt(((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y))); }
	inline __host__ __device__ float DistanceSquared(const MyVec2& v) const { return ((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y)); }
	inline __host__ __device__ float Dot(const MyVec2& v) const { return x * v.x + y * v.y; }
	inline __host__ __device__ float Cross(const MyVec2& v) const { return x * v.y + y * v.x; }

	inline __host__ __device__ MyVec2 Normalized() {
		float l = Length();
		if (l != 0.0f) {
			return MyVec2(x /= l, y /= l);
		}
		return MyVec2(0, 0);
	}

	inline __host__ __device__ MyVec2& Normalize() {
		float l = Length();
		if (l != 0.0f) {
			x /= l;
			y /= l;
		}
		return *this;
	}
};

template <typename T>
inline  __host__ __device__ MyVec2 operator*(T scalar, MyVec2 const& vec) {
	return vec * scalar;
}

std::ostream& operator<<(std::ostream& os, MyVec2& v);
