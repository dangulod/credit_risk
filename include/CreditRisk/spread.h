#ifndef SPREAD_H
#define SPREAD_H

#include <armadillo>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "utils.h"

namespace pt = boost::property_tree;

namespace CreditRisk
{

class Spread
{
public:
    Spread() = default;
    Spread(std::vector<std::string> states, std::vector<unsigned int> terms, arma::mat matrix);
    Spread(const Spread & value) = delete;
    Spread(Spread && value) = default;
    Spread & operator= (Spread && ) = default;
    ~Spread() = default;

    pt::ptree to_ptree();
    static Spread from_ptree(pt::ptree & value);

    void to_csv(std::string file);
    static Spread from_csv(std::string file);

    static Spread from_ect(std::string file);

    double at(size_t ii, size_t jj);

    size_t n_states();
    size_t n_terms();

    double spread(std::string state, double term);
    size_t state(std::string state);

    arma::vec get_spreads(double term, std::string state,  double rf, double max = 1);

    arma::mat getMatrix();

private:
    std::vector<std::string> m_states;
    std::vector<unsigned int> m_terms;
    arma::mat m_matrix;
};

}

#endif // SPREAD_H
