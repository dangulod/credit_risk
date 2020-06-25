#ifndef PORTFOLIO_OPTIM_H
#define PORTFOLIO_OPTIM_H

#include "credit_portfolio.h"
#include "../LBFGSB.h"

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

Vector arma_to_eigen(arma::vec &x, size_t i = 0);

namespace CreditRisk
{
    class Portfolio_optim
    {
    private:
        Credit_portfolio * m_credit_portfolio;

        // constraints

        double m_total_ead_var;

        Integrator::PointsAndWeigths * m_points;

        arma::vec m_ns;
        arma::mat m_pd_c;
        arma::vec m_eadxlgd;

        arma::vec m_EAD_p;
        double m_total_ead;

    public:
        arma::vec m_values0;
        arma::vec m_lower_constraints;
        arma::vec m_upper_constraints;

        Portfolio_optim() = delete;
        Portfolio_optim(Credit_portfolio * portfolio, CreditRisk::Integrator::PointsAndWeigths * points);

        Portfolio_optim(const Portfolio_optim & value) = delete;
        Portfolio_optim(Portfolio_optim && value) = default;
        ~Portfolio_optim() = default;

        void lower_constraints(arma::vec lower);
        void upper_constraints(arma::vec upper);

        void lower_constraints(size_t pos, double value);
        void upper_constraints(size_t pos, double value);

        void total_ead_var(double var);

        void setAttributtes();

        double check_ead(arma::vec x);
        double evaluate(arma::vec x);

        // t_ead portfolio_growth

        arma::vec get_x_type1(Vector x);
        double eva_portfolio_growth(Vector x);
        arma::vec optim_portfolio_growth(arma::vec x);

    };
}

#endif // Portfolio_optim_H
