#ifndef PORTFOLIO_H
#define PORTFOLIO_H

#include "element.h"
#include "optim.hpp"

namespace CreditRisk
{
    class Portfolio: public std::vector<CreditRisk::Element>
    {
    public:
        string name;
        double CtI, HR, rf, tax;

        double __fit_T_loss(double t, double loss0, arma::vec f, unsigned long idio_id);
        double __fit_T_loss_t(double t, double loss0, arma::vec cwi, arma::vec v_t);

        Portfolio() = default;
        Portfolio(string name);
        Portfolio(string name, double CtI, double HR, double rf, double tax);
        Portfolio(const Portfolio & value) = delete;
        Portfolio(Portfolio && value) = default;
        virtual ~Portfolio();

        virtual pt::ptree to_ptree();
        void to_ptree(pt::ptree & ptree);
        static Portfolio from_ptree(pt::ptree & value);

        void operator+(CreditRisk::Element && v);
        void operator+(CreditRisk::Element & v);

        double getT_EADxLGD();
        double getT_EAD();
        double get_PE();

        double loss(arma::vec f, unsigned long idio_id);
        double loss(double t, arma::vec  f, unsigned long idio_id);

        double loss(double t, arma::vec cwi, arma::vec v_t);

        arma::vec get_cwi(arma::vec f, unsigned long idio_id);
        arma::vec get_t(arma::vec cwi);

        double K (double s, arma::vec pd_c, size_t & id);
        double K1(double s, arma::vec pd_c, size_t & id);
        double K2(double s, arma::vec pd_c, size_t & id);

        std::tuple<double, double, double> K012(double s, arma::vec pd_c, size_t & id);
        std::tuple<double, double>         K12(double  s, arma::vec pd_c, size_t & id);
    };
}

#endif // PORTFOLIO_H
