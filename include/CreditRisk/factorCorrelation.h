#ifndef FACTORCORRELATION_H__
#define FACTORCORRELATION_H__


#include <armadillo>
#include <vector>
#include <string>
#include <algorithm>
#include "equation.h"
#include <fstream>
#include <boost/algorithm/string/split.hpp>

namespace pt = boost::property_tree;

const static double EPSILON = 1e-9;

using std::string;
using std::vector;

namespace CreditRisk
{
    class CorMatrix
    {
    public:
        arma::mat cor, vs;

        CorMatrix() = delete;
        CorMatrix(arma::mat cor);
        CorMatrix(const CorMatrix & value) = delete;
        CorMatrix(CorMatrix && value) = default;
        ~CorMatrix() = default;

        pt::ptree to_ptree();
        static CorMatrix from_ptree(pt::ptree & value);

        void to_csv(string file);
        static CorMatrix from_csv(string file, size_t n_factors);

        void check_equation(CreditRisk::Equation & value);
    };
}

#endif // !FACTORCORRELATION_H__
