#ifndef CREDIT_PORTFOLIO_H
#define CREDIT_PORTFOLIO_H

#include <memory>
#include <thread>
#include <mutex>
#include <algorithm>
#include "../ThreadPool/threadPool.hpp"
#include <nlopt.hpp>
#include "portfolio.h"
#include "fund.h"
#include "factorCorrelation.h"
#include "utils.h"
#include "integrator.h"
#include "transition.h"
#include "spread.h"
#ifdef USE_OPENXLSX
#include <openxlsx/OpenXLSX.hpp>
#endif
#include <regex>

#  define Q_UNUSED(x) (void)x;

namespace CreditRisk
{
    class Credit_portfolio: public std::vector<std::unique_ptr<CreditRisk::Portfolio>>
    {
    private:
        size_t n;
        std::shared_ptr<Transition> m_transition;
        std::shared_ptr<Spread> m_spread;

        // Idiosynchratic
        arma::vec v_idio(size_t n);
        void pmIdio(arma::mat *l, size_t n, size_t id, size_t p);

        // Credit Worthiness Index
        arma::vec getCWI(arma::vec f, unsigned long idio_id);
        arma::vec getCWI(unsigned long seed, unsigned long idio_id);
        void pmCWI(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p);

        // Loss distribution at counterparty level
        arma::vec marginal(arma::vec f, unsigned long idio_id, bool migration = true);
        arma::vec marginal(unsigned long seed, unsigned long idio_id, bool migration = true);
        void pmloss(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p, bool migration = true);
        arma::vec marginal_without_secur(arma::vec f, unsigned long idio_id, bool migration = true);
        arma::vec marginal_without_secur(unsigned long seed, unsigned long idio_id, bool migration = true);
        void pmloss_without_secur(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p, bool migration = true);

        // Loss distribution at ru level
        arma::vec sLoss_ru(arma::vec  f, unsigned long idio_id, bool migration = true);
        arma::vec sLoss_ru(unsigned long seed, unsigned long idio_id, bool migration = true);
        void ploss_ru(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration = true);
        arma::vec sLoss_ru_without_secur(arma::vec  f, unsigned long idio_id, bool migration = true);
        arma::vec sLoss_ru_without_secur(unsigned long seed, unsigned long idio_id, bool migration = true);
        void ploss_ru_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration = true);

        // Loss distribution at portfolio level
        arma::vec sLoss_portfolio(arma::vec  f, unsigned long idio_id, bool migration = true);
        arma::vec sLoss_portfolio(unsigned long seed, unsigned long idio_id, bool migration = true);
        void ploss_portfolio(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration = true);
        arma::vec sLoss_portfolio_without_secur(arma::vec  f, unsigned long idio_id, bool migration = true);
        arma::vec sLoss_portfolio_without_secur(unsigned long seed, unsigned long idio_id, bool migration = true);
        void ploss_portfolio_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration = true);

        // Loss distribution
        double sLoss(arma::vec  f, unsigned long idio_id, bool migration = true);
        void ploss(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration = true);
        double sLoss_without_secur(arma::vec  f, unsigned long idio_id, bool migration = true);
        void ploss_without_secur(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration = true);

        // Factors generation
        arma::vec v_rand(unsigned long seed);
        void pv_rand(arma::mat *r, size_t n, unsigned long seed, size_t id, size_t p);

        // Saddle Point

        void pd_c_fill(std::vector<Scenario> * pd_c_mig, size_t * ii, CreditRisk::Integrator::PointsAndWeigths * points, bool migration = true);

        void saddle_point(double loss, arma::vec * n, LStates *eadxlgd, std::vector<Scenario> * pd_c,
                             CreditRisk::Integrator::PointsAndWeigths * points, arma::vec * saddle_points, size_t id, size_t p);
        void contrib_without_secur(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c, arma::vec * con,
                                   arma::vec * c_contrib, CreditRisk::Integrator::PointsAndWeigths * points, size_t id, size_t p);
        void contrib(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c, arma::vec * con, arma::vec * c_contrib,
                     CreditRisk::Integrator::PointsAndWeigths * points, size_t id, size_t p);

        double fitQuantile(double loss, double prob, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                           CreditRisk::Integrator::PointsAndWeigths & points, TP::ThreadPool * pool);

    public:
        double fitSaddle_n(double s, double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);
        double fitSaddle_n_secur(double s, double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);

        double T_EAD, T_EADxLGD;
        CreditRisk::CorMatrix cf;
        vector<unsigned long> rus;
        vector<size_t> rus_pos;

        Credit_portfolio() = delete;
        Credit_portfolio(arma::mat cor);
        Credit_portfolio(arma::mat cor, Transition & transition, Spread & spread);
        Credit_portfolio(const Credit_portfolio & value) = delete;
        Credit_portfolio(Credit_portfolio && value) = default;
        ~Credit_portfolio() = default;

        void operator+(CreditRisk::Portfolio &  value);
        void operator+(CreditRisk::Portfolio && value);
        void operator+(CreditRisk::Fund &  value);
        void operator+(CreditRisk::Fund && value);

        pt::ptree to_ptree();
        static Credit_portfolio from_ptree(pt::ptree & value);

        static Credit_portfolio from_csv(string Portfolios, string Funds, string Elements, string CorMatrix, size_t n_factors,
                                         string transition = "", string spread = "");

        static Credit_portfolio from_ect(string wholesale, string retail, string CorMatrix,
                                         string Funds = "", string transition = "", string spread = "");

#ifdef USE_OPENXLSX
        static Credit_portfolio from_xlsx_ps(string file, string transition = "", string spread = "");
#endif
        size_t getN();
        void setT_EADxLGD();
        void setT_EAD();

        void setRUs();

        double getPE();
        size_t n_factors();

        Element & get_element(size_t column);
        long which_fund(size_t column);
        size_t which_portfolio(size_t column);

        Transition * get_transition();
        Spread * get_spread();

        // Get vectors

        arma::vec get_std_EADxLGDs();
        std::shared_ptr<LStates> get_std_states(bool migration = true);
        arma::vec get_EADxLGDs();
        arma::vec get_Ns();
        std::vector<double> get_portfolios_EADs();
        std::vector<Scenario_data> get_scenario_data(unsigned long n);

        arma::vec get_ru_el();
        arma::vec get_ru_allocation(const arma::vec & contrib);

        // Monte Carlo
        // Factors generation
        arma::mat m_rand(size_t n, unsigned long seed, TP::ThreadPool * pool);
        double d_rand(size_t row, size_t column, size_t n, unsigned long seed);

        // Credit Worthiness Index
        arma::mat getCWIs(size_t n, unsigned long seed, TP::ThreadPool * pool);
        double d_CWI(size_t row, size_t column, size_t n, unsigned long seed);

        // Idiosynchratic
        arma::mat getIdio(size_t n, TP::ThreadPool * pool);
        double d_Idio(size_t row, size_t column, size_t n);

        // Loss distribution at counterparty level
        double smargin_loss(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario, bool migration = true);
        double smargin_loss_without_secur(size_t row, size_t column, size_t n, unsigned long seed, bool migration = true);
        arma::mat margin_loss(size_t n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);
        arma::mat margin_loss_without_secur(size_t n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);

        // Loss distribution at ru level

        double sLoss_ru(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario, bool migration = true);
        double sLoss_ru_without_secur(size_t row, size_t column, size_t n, unsigned long seed, bool migration = true);

        arma::mat loss_ru(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);
        arma::mat loss_ru_without_secur(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);

        // Loss distribution at portfolio level

        double sLoss_portfolio(size_t row, size_t column, size_t n, unsigned long seed, bool migration = true);
        double sLoss_portfolio_without_secur(size_t row, size_t column, size_t n, unsigned long seed, bool migration = true);
        arma::mat loss_portfolio(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);
        arma::mat loss_portfolio_without_secur(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);

        // Loss distribution
        double sLoss(unsigned long seed, unsigned long idio_id, bool migration = true);
        double sLoss_without_secur(unsigned long seed, unsigned long idio_id, bool migration = true);
        arma::vec loss(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);
        arma::vec loss_without_secur(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration = true);

        // Conditional probabiliy

        Scenario pd_c(arma::vec t, double scenarios, bool migration = true);
        Scenario pd_c(double scenario, bool migration = true);

        std::shared_ptr<std::vector<Scenario>> pd_c(CreditRisk::Integrator::PointsAndWeigths points, TP::ThreadPool * pool, bool migration = true);

        // Saddle Point
        // with vectors

        arma::vec get_t_secur(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, arma::vec k1s, double scenario);

        double K (double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);
        double K1(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);
        double K1_secur(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);
        double K2(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);
        arma::vec K1_secur_vec(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);

        std::tuple<double, double, double> K012(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);
        std::tuple<double, double>         K12(double  s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);

        std::tuple<double, double>         K12_secur(double  s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c);

        double getSaddle(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c,
                         double s0 = 0, double a = -1e12, double b = 1e12, double tol = 1e-7);
        double getSaddleBrent(double loss, arma::vec * n, LStates *eadxlgd, Scenario * pd_c,
                              double a = -1e3, double b = 1e3, double xtol = 1e-12, double rtol = 1e-6);
        std::tuple<double, double> getSaddleNewton(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c,
                                                   double s0 = 0, double tol = 1e-9);

        double getSaddle_secur(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double s0 = 0, double a = -1e10, double b = 1e10, double tol = 1e-7);
        double getSaddleBrent_secur(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double a = -1e3, double b = 1e3, double xtol = 1e-12, double rtol = 1e-6);
        std::tuple<double, double> getSaddleNewton_secur(double loss, arma::vec * n,  LStates * eadxlgd, Scenario * pd_c, double s0 = 0, double tol = 1e-9);

        // Optimizer

        double cdf(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                   CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool);

        double quantile(double prob, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                        CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool,
                        double xtol = 1e-12, double rtol = 1e-6);

        arma::vec getContrib_without_secur(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                                           CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool);

        arma::vec getContrib(double loss,  arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                             CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool);

        double EVA(LStates * eadxlgd, arma::vec contrib);

        // Optimizer

        arma::vec minimize_EAD_constant(arma::vec * n, std::vector<Scenario> * pd_c, CreditRisk::Integrator::PointsAndWeigths * points,
                                        TP::ThreadPool * pool, double total_ead_var, std::vector<double> x0, std::vector<double> lower, std::vector<double> upper);

    };
}

#endif // CREDIT_PORTFOLIO_H
