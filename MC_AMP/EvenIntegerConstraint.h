#pragma once
#include <string>
#include<tclap/Constraint.h>
class DaysConstraint : public TCLAP::Constraint<int>
{
	virtual std::string description() const override;
	virtual std::string shortID() const override;
	virtual bool check(const int & value) const override;
};

