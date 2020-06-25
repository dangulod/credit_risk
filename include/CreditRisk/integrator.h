#ifndef INTEGRATORPOINTS_H__
#define INTEGRATORPOINTS_H__

#pragma once

#include <armadillo>
#include <boost/math/constants/constants.hpp>
#include "utils.h"

namespace CreditRisk
{
    namespace Integrator
    {
        struct PointsAndWeigths
        {
            arma::vec points;
            arma::vec weigths;
        };

        PointsAndWeigths ghi(unsigned long n = 7, double eps = 1e-13);
        PointsAndWeigths gki(double start = -4.5, unsigned long steps = 100);
    }
}

#endif // INTEGRATORPOINTS_H__
