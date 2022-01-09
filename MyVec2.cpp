#include "MyVec2.h"

std::ostream& operator<<(std::ostream& os, MyVec2& v) {
	return os << "< " << v.x << ", " << v.y << ">" << std::endl;
}