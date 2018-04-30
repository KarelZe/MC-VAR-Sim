#include "PercentageConstraint.h"


std::string PercentageConstraint::description() const
{
	return "number must be (0,1)";
}

std::string PercentageConstraint::shortID() const
{
	return "objname";
}

bool PercentageConstraint::check(const float & value) const
{
	return value > 0 && value < 1;
}


