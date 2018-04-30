#include "PathConstraint.h"

std::string PathConstraint::description() const
{
	return "paths must be [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]";
}

std::string PathConstraint::shortID() const
{
	return "objname";
}

bool PathConstraint::check(const int & value) const
{
	return value == 1024 || value == 2048 || value == 4096 || value == 8192 || value == 16384 || value == 32768 || value == 65536 || value == 131072 || value == 262144 || value == 524288;
}
