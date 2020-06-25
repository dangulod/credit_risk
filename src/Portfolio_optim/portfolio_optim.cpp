#include "portfolio_optim.h"

Vector arma_to_eigen(arma::vec &x, size_t i)
{
    Vector x0(x.size() - i);

    for (long ii = 0; ii < x0.size(); ii++)
    {
        x0[ii] = x.at(ii);
    }

    return x0;
}

namespace CreditRisk
{
    Portfolio_optim::Portfolio_optim(Credit_portfolio * portfolio, CreditRisk::Integrator::PointsAndWeigths * points) :
        m_credit_portfolio(portfolio), m_total_ead_var(0), m_points(points)
    {
        this->setAttributtes();
    }

    void Portfolio_optim::lower_constraints(arma::vec lower)
    {
        if (this->m_credit_portfolio->size() == lower.size())
        {
            this->m_lower_constraints = lower;
        }
    }

    void Portfolio_optim::upper_constraints(arma::vec upper)
    {
        if (this->m_credit_portfolio->size() == upper.size())
        {
            this->m_upper_constraints = upper;
        }
    }

    void Portfolio_optim::lower_constraints(size_t pos, double value)
    {
        if (pos > 0 & pos < this->m_lower_constraints.size())
        {
            this->m_lower_constraints(pos) = value;
        }
    }

    void Portfolio_optim::upper_constraints(size_t pos, double value)
    {
        if (pos > 0 & pos < this->m_upper_constraints.size())
        {
            this->m_upper_constraints(pos) = value;
        }
    }

    void Portfolio_optim::total_ead_var(double var)
    {
        this->m_total_ead_var = var;
    }

    void Portfolio_optim::setAttributtes()
    {
        this->m_ns = this->m_credit_portfolio->get_Ns();
        this->m_pd_c = this->m_credit_portfolio->pd_c(*this->m_points);
        this->m_eadxlgd = this->m_credit_portfolio->get_std_EADxLGDs();
        this->m_EAD_p = arma::vec(this->m_credit_portfolio->get_portfolios_EADs());
        this->m_total_ead = arma::accu(this->m_EAD_p);
    }

    double Portfolio_optim::check_ead(arma::vec x)
    {
        double t_ead = 0;

        for (size_t ii = 0; ii < x.size(); ii++)
        {
            t_ead += (1 + x[ii]) * this->m_EAD_p.at(ii);
        }

        return t_ead;
    }

    double Portfolio_optim::evaluate(arma::vec x)
    {
        /*
        std::cout << "upper: " << arma::accu(x > this->m_upper_constraints) << std::endl;
        std::cout << "lower: " << arma::accu(x < this->m_lower_constraints) << std::endl;
        if (arma::accu(x > this->m_upper_constraints) | arma::accu(x < this->m_lower_constraints))
        {
            return 1e9;
        }
        */

        x.t().print();
        //printf("EAD: %.20f\n", this->check_ead(x));

        arma::vec eadxldg(this->m_credit_portfolio->getN());

        double T_EADxLGD = 0;
        size_t jj = 0;
        size_t kk = 0;

        for (auto & ii: *this->m_credit_portfolio)
        {
            for (size_t hh = 0; hh < ii->size(); hh++)
            {
                eadxldg[jj] = this->m_eadxlgd.at(jj) * (1 + x[kk]);
                jj++;
            }
            T_EADxLGD += this->m_EAD_p.at(kk) * (1 + x.at(kk));
            kk++;
        }

        double loss = m_credit_portfolio->quantile(0.9995, this->m_ns, eadxldg,
                                                   this->m_pd_c, this->m_points, 1e-13, 1e-7, 1);

        arma::vec contrib = this->m_credit_portfolio->getContrib_without_secur(loss, this->m_ns,
                                                                               eadxldg, this->m_pd_c,
                                                                               this->m_points);

        return  -this->m_credit_portfolio->EVA(eadxldg * m_credit_portfolio->T_EADxLGD,
                                               contrib * m_credit_portfolio->T_EADxLGD);
    }

    arma::vec Portfolio_optim::get_x_type1(Vector x)
    {
        double xn = this->m_total_ead * (1 + this->m_total_ead_var);
        arma::vec growths(x.size() + 1);

        for (long ii = 0; ii < x.size(); ii++)
        {
            xn -= (1 + x(ii)) * this->m_EAD_p.at(ii);
        }

        xn /= this->m_EAD_p.at(this->m_EAD_p.size() - 1);

        for (long ii = 0; ii < x.size(); ii++)
        {
            growths.at(ii) = x(ii);
        }

        growths.at(x.size()) = (xn - 1);

        return growths;

    }

    double Portfolio_optim::eva_portfolio_growth(Vector x)
    {
        arma::vec growths = this->get_x_type1(x);

        return this->evaluate(growths);
    }

    arma::vec Portfolio_optim::optim_portfolio_growth(arma::vec x)
    {
        Vector x0 = arma_to_eigen(x, 1);
        Vector lb = arma_to_eigen(this->m_lower_constraints, 1);
        Vector up = arma_to_eigen(this->m_upper_constraints, 1);
        double fx;

        LBFGSpp::L_BFGS_B(&Portfolio_optim::eva_portfolio_growth, *this, x0, fx, lb, up);

        return this->get_x_type1(x0);

    }
}
