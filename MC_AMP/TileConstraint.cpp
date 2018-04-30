#include "TileConstraint.h"


std::string TileConstraint::description() const
{
	return "tile must be [16, 32, 64, 128, 256, 512, 1024]";
}

std::string TileConstraint::shortID() const
{
	return "objname";
}

bool TileConstraint::check(const int & value) const
{
	return value == 16 || value == 32 || value == 64 || value == 128 || value == 256 || value == 512 || value == 1024;
}
