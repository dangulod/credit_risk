#ifndef UTILS_HPP
#define UTILS_HPP

#include <boost/math/special_functions/erf.hpp>
#include <boost/algorithm/string/split.hpp>
#include <random>
#include <armadillo>
#include <vector>

namespace CreditRisk
{
    namespace Utils
    {
        size_t number_of_lines(std::string file);

        void isProbability(double p);

        double qnorm(double p);
        arma::vec qnorm(const arma::vec & p);
        double pnorm(double x);
        arma::vec pnorm(const arma::vec & x);

        double randn_s();
        double randn_s(unsigned long seed);

        arma::vec randn_v(size_t n, unsigned long seed);

        template<class T>
        size_t get_position(std::vector<T> v, T x)
        {
            size_t ii(0);
            if (v.size() == 0) return ii;
            while (ii < v.size() & v[ii] != x)
            {
                ii++;
            }
            return ii;
        }
        double mean(const arma::vec & x);

        double quantile(arma::vec x, double q);
        arma::vec contributions(const arma::mat & x, double q, double lower, double upper);

        arma::vec rowSum(const arma::mat & x);
    }

    namespace saddle {
        double p_c(double p, double beta, double idio, double cwi);
        arma::vec p_states_c(arma::vec & p_states, double npd, double beta, double idio, double cwi);

        double num(double s, arma::vec l_states, arma::vec p_states);
        double num2(double s, arma::vec l_states, arma::vec p_states);
        double den(double s, arma::vec l_states, arma::vec p_states);

        double K(double s, unsigned long n, arma::vec l_states, arma::vec p_states);
        double K1(double s, unsigned long n, arma::vec l_states, arma::vec p_states);
        double K2(double s, unsigned long n, arma::vec l_states, arma::vec p_states);

        double num(double s, double _le, double pd_c);
        double den(double s, double _le, double pd_c);

        double K(double s, unsigned long n, double _le, double pd_c);
        double K1(double s, unsigned long n, double _le, double pd_c);
        double K2(double s, unsigned long n, double _le, double pd_c);
    }
}

#endif // UTILS_HPP
