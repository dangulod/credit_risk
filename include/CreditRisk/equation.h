#ifndef EQUATION_H
#define EQUATION_H

#include <armadillo>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "utils.h"

namespace pt = boost::property_tree;

namespace CreditRisk
{
    class Equation
    {
    public:
        unsigned long idio_seed;
        arma::vec weights;
        double R2, idio;

        Equation() = delete;
        Equation(unsigned long idio_seed, arma::vec weigths);
        Equation(const Equation & v) = delete;
        Equation(Equation && v) = default;
        ~Equation() = default;

        pt::ptree to_ptree();
        static Equation from_ptree(pt::ptree & value);

        double CWI(arma::vec f, unsigned long id);
        double CWI(arma::vec f, double i);

        void setIdio(arma::mat cor);

        double systematic(arma::vec f);
    };
}
#endif // EQUATION_H
