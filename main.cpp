#include <iostream>
#include <armadillo>
#include <credit_portfolio.h>
#include <chrono>
#include "ThreadPool/threadPool.hpp"
#include <spread.h>
#include <transition.h>

class Fitness_parameters
{
public:
    CreditRisk::Credit_portfolio * m_cp;
    CreditRisk::Integrator::PointsAndWeigths m_p;
    TP::ThreadPool * m_pool;
    std::shared_ptr<std::vector<Scenario>> m_pd_c;
    arma::vec m_ns;
    std::vector<double> m_EAD_p;
    double m_total_ead_var, m_ead_var, m_new_T_EAD;

    Fitness_parameters() = delete;
    Fitness_parameters(
            CreditRisk::Credit_portfolio * credit_portfolio,
            CreditRisk::Integrator::PointsAndWeigths points,
            double total_ead_var,
            double ead_var,
            TP::ThreadPool * pool):
        m_cp(credit_portfolio),
        m_p(points),
        m_pool(pool),
        m_pd_c(credit_portfolio->pd_c(points, pool)),
        m_ns(credit_portfolio->get_Ns()),
        m_EAD_p(credit_portfolio->get_portfolios_EADs()),
        m_total_ead_var(total_ead_var),
        m_ead_var(ead_var),
        m_new_T_EAD(credit_portfolio->T_EAD * (1 + total_ead_var)) {}
    Fitness_parameters(const Fitness_parameters & value) = delete;
    Fitness_parameters(Fitness_parameters && value) = default;
    ~Fitness_parameters() = default;

    size_t get_n()
    {
        return this->m_cp->size() - 1;
    }

    std::vector<double> get_xn(std::vector<double> x)
    {
        double xn = m_new_T_EAD;
        std::vector<double> sol(x.size() + 1);

        for (size_t ii = 0; ii < x.size(); ii++)
        {
            xn -= (1 + x[ii]) * this->m_EAD_p[ii];
        }

        xn /= this->m_EAD_p[this->m_EAD_p.size() - 1];

        for (size_t ii = 0; ii < x.size(); ii++)
        {
            sol[ii] = x[ii];
        }

        sol[x.size()] = (xn - 1);

        return sol;
    }

    double check_ead(std::vector<double> x)
    {
        double t_ead = 0;

        for (size_t ii = 0; ii < x.size(); ii++)
        {
            t_ead += (1 + x[ii]) * this->m_EAD_p[ii];
        }

        return t_ead;
    }

    std::shared_ptr<LStates> std_eadxlgds(std::vector<double> x)
    {
        std::shared_ptr<LStates> std_eadxlgds(new LStates(this->m_cp->getN()));
        double T_EAD = 0;

        auto jj = std_eadxlgds->begin();
        auto hh = x.begin();

        for (auto & ii: *this->m_cp)
        {
            for (auto & kk: *ii)
            {
                *jj = (kk.l_states(true) / kk.n) * (1 + (*hh));
                T_EAD += jj->back();
                jj++;
            }
            hh++;
        }

        for (auto & ii: *std_eadxlgds)
        {
            ii /= T_EAD;
        }

        return std_eadxlgds;
    }


    double evaluate(std::vector<double> x)
    {
        std::shared_ptr<LStates> std_eadxlgds(new LStates(this->m_cp->getN()));
        double T_EAD = 0;

        auto jj = std_eadxlgds->begin();
        auto hh = x.begin();

        for (auto & ii: *this->m_cp)
        {
            for (auto & kk: *ii)
            {
                *jj = (kk.l_states(true) / kk.n) * (1 + (*hh));
                T_EAD += jj->back();
                jj++;
            }
            hh++;
        }

        for (auto & ii: *std_eadxlgds)
        {
            ii /= T_EAD;
        }

        double loss = this->m_cp->quantile(0.9995, &this->m_ns, std_eadxlgds.get(), this->m_pd_c.get(), &this->m_p, this->m_pool);
        arma::vec contrib = this->m_cp->getContrib_without_secur(loss, &this->m_ns, std_eadxlgds.get(), this->m_pd_c.get(), &this->m_p, this->m_pool);

        for (auto & ii: *std_eadxlgds)
        {
            ii.back() *= T_EAD;
        }

        return  this->m_cp->EVA(std_eadxlgds.get(), contrib * T_EAD);
    }

};

static int iter = 0;

double fitness(const std::vector<double> &x, std::vector<double> &grad, void *args)
{
    Q_UNUSED(grad);
    Fitness_parameters * parameters = static_cast<Fitness_parameters *>(args);

    std::vector<double> sol = parameters->get_xn(x);

    if (fabs(sol[sol.size() - 1]) > parameters->m_ead_var) return 1e10;
    double eva = parameters->evaluate(sol);
    iter++;

    //printf("iter %i\r", iter);
    std::cout << "Iter: " << iter << " f(x)= " << eva << " EAD: " << std::setprecision(16) << parameters->check_ead(sol);
    for (auto & ii: sol) std::cout << " " << ii << " ";
    std::cout << std::endl;

    return -eva;
}


int main()
{
    TP::ThreadPool pool(8);
    pool.init();

    pt::ptree pt;
    pt::read_json("/opt/share/data/optim/optim.json", pt);

    CreditRisk::Credit_portfolio p = CreditRisk::Credit_portfolio::from_ptree(pt);

    p.arrange();

    CreditRisk::Integrator::PointsAndWeigths points = CreditRisk::Integrator::gki();

    Fitness_parameters args = Fitness_parameters(&p, points, 0, 0.1, &pool);

    vector<double> lower(args.get_n());
    std::fill(lower.begin(), lower.end(), -0.1);

    vector<double> upper(args.get_n());
    std::fill(upper.begin(), upper.end(), 0.1);

    nlopt::opt optimizer(nlopt::GN_ISRES, args.get_n()); // GN_ISRES GN_ESCH GN_MLSL LN_COBYLA GN_CRS2_LM LN_AUGLAG_EQ NLOPT_LN_BOBYQA
    optimizer.set_lower_bounds(lower);
    optimizer.set_upper_bounds(upper);

    optimizer.set_min_objective(fitness, (void*)&args);

    optimizer.set_xtol_rel(1e-9);
    optimizer.set_maxeval(10000);

    std::vector<double> x0(args.get_n());
    std::fill(x0.begin(), x0.end(), 0);

    nlopt_srand(987654321);

    auto dx = std::chrono::high_resolution_clock::now();

    double minf;

    try{
        nlopt::result result = optimizer.optimize(x0, minf);
        std::cout << "found minimum at f(" << ") = "
                  << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    for (auto & ii: x0) printf("%.20f\n", ii);

    auto dy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dif = dy - dx;
    std::cout << dif.count() << " seconds" << std::endl;

    pool.shutdown();

    return 0;
}


