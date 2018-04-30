#include "EvenIntegerConstraint.h"

std::string DaysConstraint::description() const
{
	return std::string();
}

std::string DaysConstraint::shortID() const
{
	return std::string();
}

bool DaysConstraint::check(const int & value) const
{
	return value > 0 && value < 1'000 && value % 2 == 0;
}
