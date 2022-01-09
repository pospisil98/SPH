#pragma once

#include <ostream>
#include <math.h>

struct MyVec2 {
	float x;
	float y;

	MyVec2() : x(0.0f), y(0.0f) { }
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

	inline float Length() const { return sqrt(x * x + y * y); }
	inline float LengthSquared() const { return x * x + y * y; }
	inline float Distance(const MyVec2& v) const { return sqrt(((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y))); }
	inline float DistanceSquared(const MyVec2& v) const { return ((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y)); }
	inline float Dot(const MyVec2& v) const { return x * v.x + y * v.y; }
	inline float Cross(const MyVec2& v) const { return x * v.y + y * v.x; }

	inline MyVec2 Normalized() {
		float l = Length();
		if (l != 0.0f) {
			return MyVec2(x /= l, y /= l);
		}
		return MyVec2(0, 0);
	}

	inline MyVec2& Normalize() {
		float l = Length();
		if (l != 0.0f) {
			x /= l;
			y /= l;
		}
		return *this;
	}
};

template <typename T>
inline MyVec2 operator*(T scalar, MyVec2 const& vec) {
	return vec * scalar;
}

std::ostream& operator<<(std::ostream& os, MyVec2& v);
