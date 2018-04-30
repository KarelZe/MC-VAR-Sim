#pragma once
#include <string>
#include<tclap/Constraint.h>
class PostiveValueConstraint : public TCLAP::Constraint<float>
{
	virtual std::string description() const override;
	virtual std::string shortID() const override;
	virtual bool check(const float & value) const override;
};

