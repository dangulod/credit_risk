#ifndef TRANSITION_H
#define TRANSITION_H

#include <armadillo>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "utils.h"

namespace pt = boost::property_tree;

namespace CreditRisk
{

class Transition
{
public:
    Transition() = delete;
    Transition(std::vector<std::string> states, arma::mat matrix);
    Transition(const Transition & value) = delete;
    Transition(Transition && value) = default;
    ~Transition() = default;

    pt::ptree to_ptree();
    static Transition from_ptree(pt::ptree & value);

    void to_csv(std::string file);
    static Transition from_csv(std::string file);

    double at(size_t ii, size_t jj);

    size_t n_states();

    arma::vec states_prob(double pd);
    std::string state(double pd);

private:
    std::vector<std::string> m_states;

    arma::mat m_matrix;
};

}

#endif // TRANSITION_H
