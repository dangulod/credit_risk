#ifndef UTILS_HPP
#define UTILS_HPP

#include <boost/math/special_functions/erf.hpp>
#include <random>
#include <armadillo>
#include <vector>

namespace CreditRisk
{
    namespace Utils
    {
        void isProbability(double p);

        double qnorm(double p);
        double pnorm(double x);

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
        double num(double s, double _le, double pd_c);
        double den(double s, double _le, double pd_c);

        double K(double s, unsigned long n, double _le, double pd_c);
        double K1(double s, unsigned long n, double _le, double pd_c);
        double K2(double s, unsigned long n, double _le, double pd_c);
    }
}

#endif // UTILS_HPP
