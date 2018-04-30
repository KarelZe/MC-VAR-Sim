#include "IntegerConstraint.h"

std::string PostiveValueConstraint::description() const
{
	return "number must be (0,1000]";
}

std::string PostiveValueConstraint::shortID() const
{
	return "objname";
}

bool PostiveValueConstraint::check(const float & value) const
{
	return value > 0 && value <= 1'000;
}
