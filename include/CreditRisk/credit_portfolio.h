#ifndef CREDIT_PORTFOLIO_H
#define CREDIT_PORTFOLIO_H

#include "portfolio.h"
#include "fund.h"
#include <memory>
#include "factorCorrelation.h"
#include <thread>
#include "utils.h"
#include "integrator.h"
#include <mutex>

namespace CreditRisk
{
    namespace saddle
    {
        double num(double s, double _le, double pd_c);
        double den(double s, double _le, double pd_c);
        double K(double s, unsigned long n, double _le, double pd_c);
        double K1(double s, unsigned long n, double _le, double pd_c);
        double K2(double s, unsigned long n, double _le, double pd_c);
    }

    class Credit_portfolio: public std::vector<std::unique_ptr<CreditRisk::Portfolio>>
    {
    private:
        size_t n;

        // Idiosynchratic
        arma::vec v_idio(size_t n);
        void pmIdio(arma::mat *l, size_t n, size_t id, size_t p);

        // Credit Worthiness Index
        arma::vec getCWI(arma::vec f, unsigned long idio_id);
        arma::vec getCWI(unsigned long seed, unsigned long idio_id);
        void pmCWI(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p);

        // Loss distribution at counterparty level
        arma::vec marginal(arma::vec f, unsigned long idio_id);
        arma::vec marginal(unsigned long seed, unsigned long idio_id);
        void pmloss(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p);
        arma::vec marginal_without_secur(arma::vec f, unsigned long idio_id);
        arma::vec marginal_without_secur(unsigned long seed, unsigned long idio_id);
        void pmloss_without_secur(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p);

        // Loss distribution at ru level
        arma::vec sLoss_ru(arma::vec  f, unsigned long idio_id);
        arma::vec sLoss_ru(unsigned long seed, unsigned long idio_id);
        void ploss_ru(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p);
        arma::vec sLoss_ru_without_secur(arma::vec  f, unsigned long idio_id);
        arma::vec sLoss_ru_without_secur(unsigned long seed, unsigned long idio_id);
        void ploss_ru_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p);

        // Loss distribution at portfolio level
        arma::vec sLoss_portfolio(arma::vec  f, unsigned long idio_id);
        arma::vec sLoss_portfolio(unsigned long seed, unsigned long idio_id);
        void ploss_portfolio(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p);
        arma::vec sLoss_portfolio_without_secur(arma::vec  f, unsigned long idio_id);
        arma::vec sLoss_portfolio_without_secur(unsigned long seed, unsigned long idio_id);
        void ploss_portfolio_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p);

        // Loss distribution
        double sLoss(arma::vec  f, unsigned long idio_id);
        void ploss(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p);
        double sLoss_without_secur(arma::vec  f, unsigned long idio_id);
        void ploss_without_secur(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p);

        // Factors generation
        arma::vec v_rand(unsigned long seed);
        void pv_rand(arma::mat *r, size_t n, unsigned long seed, size_t id, size_t p);

        // Saddle Point
        double fitSaddle(double s, double loss, arma::vec pd_c);

        void pd_c_fill(arma::mat * pd_c, size_t * ii, CreditRisk::Integrator::PointsAndWeigths * points);

        void saddle_point_pd(double loss, arma::vec * n, arma::vec * eadxlgd, arma::mat * pd_c, CreditRisk::Integrator::PointsAndWeigths * points, arma::vec * saddle_points, size_t id, size_t p);
        void contrib_without_secur(double loss, arma::vec * n, arma::vec * eadxlgd, arma::mat * pd_c, arma::vec * con, arma::vec * c_contrib, CreditRisk::Integrator::PointsAndWeigths * points, size_t id, size_t p);
        void contrib(double loss, arma::vec * n, arma::vec * eadxlgd, arma::mat * pd_c, arma::vec * con, arma::vec * c_contrib, CreditRisk::Integrator::PointsAndWeigths * points, size_t id, size_t p);

        double fitQuantile_pd(double loss, double prob, arma::vec n, arma::vec eadxlgd, arma::mat pd_c, CreditRisk::Integrator::PointsAndWeigths & points, size_t p);

    public:
        double fitSaddle_n(double s, double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);
        double fitSaddle_n_secur(double s, double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);

        double T_EAD, T_EADxLGD;
        CreditRisk::CorMatrix cf;
        vector<unsigned long> rus;
        vector<size_t> rus_pos;

        Credit_portfolio() = delete;
        Credit_portfolio(arma::mat cor);
        Credit_portfolio(const Credit_portfolio & value) = delete;
        Credit_portfolio(Credit_portfolio && value) = default;
        ~Credit_portfolio() = default;

        void operator+(CreditRisk::Portfolio &  value);
        void operator+(CreditRisk::Portfolio && value);
        void operator+(CreditRisk::Fund &  value);
        void operator+(CreditRisk::Fund && value);

        pt::ptree to_ptree();
        static Credit_portfolio from_ptree(pt::ptree & value);

        static Credit_portfolio from_csv(string Portfolios, string Funds, string Elements, string CorMatrix, size_t n_factors);

        size_t getN();
        void setT_EADxLGD();
        void setT_EAD();

        void setRUs();

        double getPE();
        size_t n_factors();

        Element & get_element(size_t column);
        long which_fund(size_t column);
        size_t which_portfolio(size_t column);

        // Get vectors

        arma::vec get_std_EADxLGDs();
        arma::vec get_EADxLGDs();
        arma::vec get_Ns();
        std::vector<double> get_portfolios_EADs();
        std::vector<Scenario_data> get_scenario_data(unsigned long n);


        // Monte Carlo
        // Factors generation
        arma::mat m_rand(size_t n, unsigned long seed, size_t p = std::thread::hardware_concurrency());
        double d_rand(size_t row, size_t column, size_t n, unsigned long seed);

        // Credit Worthiness Index
        arma::mat getCWIs(size_t n, unsigned long seed, size_t p = std::thread::hardware_concurrency());
        double d_CWI(size_t row, size_t column, size_t n, unsigned long seed);

        // Idiosynchratic
        arma::mat getIdio(size_t n, size_t p = std::thread::hardware_concurrency());
        double d_Idio(size_t row, size_t column, size_t n);

        // Loss distribution at counterparty level
        double smargin_loss(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario);
        double smargin_loss_without_secur(size_t row, size_t column, size_t n, unsigned long seed);
        arma::mat margin_loss(size_t n, unsigned long seed, size_t p = std::thread::hardware_concurrency());
        arma::mat margin_loss_without_secur(size_t n, unsigned long seed, size_t p = std::thread::hardware_concurrency());

        // Loss distribution at ru level

        double sLoss_ru(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario);
        double sLoss_ru_without_secur(size_t row, size_t column, size_t n, unsigned long seed);

        arma::mat loss_ru(unsigned long n, unsigned long seed, unsigned long p = std::thread::hardware_concurrency());
        arma::mat loss_ru_without_secur(unsigned long n, unsigned long seed, unsigned long p = std::thread::hardware_concurrency());

        // Loss distribution at portfolio level

        double sLoss_portfolio(size_t row, size_t column, size_t n, unsigned long seed);
        double sLoss_portfolio_without_secur(size_t row, size_t column, size_t n, unsigned long seed);
        arma::mat loss_portfolio(unsigned long n, unsigned long seed, unsigned long p = std::thread::hardware_concurrency());
        arma::mat loss_portfolio_without_secur(unsigned long n, unsigned long seed, unsigned long p = std::thread::hardware_concurrency());

        // Loss distribution
        double sLoss(unsigned long seed, unsigned long idio_id);
        double sLoss_without_secur(unsigned long seed, unsigned long idio_id);
        arma::vec loss(unsigned long n, unsigned long seed, unsigned long p = std::thread::hardware_concurrency());
        arma::vec loss_without_secur(unsigned long n, unsigned long seed, unsigned long p = std::thread::hardware_concurrency());

        // Conditional probabiliy

        arma::vec pd_c(double scenario);
        arma::vec pd_c(double t, double scenario);
        arma::vec pd_c(arma::vec t, double scenarios);
        arma::vec pd_c(arma::vec scenarios);
        arma::vec pd_c(double t, arma::vec scenarios);
        arma::vec pd_c(arma::vec t, arma::vec scenarios);

        arma::mat pd_c(CreditRisk::Integrator::PointsAndWeigths points = Integrator::ghi(), unsigned long p = std::thread::hardware_concurrency());

        // Saddle Point
        // with vectors

        arma::vec get_t_secur(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, arma::vec k1s, double scenario);

        double K (double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);
        double K1(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);
        double K1_secur(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);
        double K2(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);
        arma::vec K1_secur_vec(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);

        std::tuple<double, double, double> K012(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);
        std::tuple<double, double>         K12(double  s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);

        std::tuple<double, double>         K12_secur(double  s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c);

        double getSaddle(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0 = 0, double a = -1e12, double b = 1e12, double tol = 1e-7);
        double getSaddleBrent(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double a = -1e3, double b = 1e3, double xtol = 1e-12, double rtol = 1e-6);
        std::tuple<double, double> getSaddleNewton(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0 = 0, double tol = 1e-9);

        double getSaddle_secur(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0 = 0, double a = -1e10, double b = 1e10, double tol = 1e-7);
        double getSaddleBrent_secur(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double a = -1e3, double b = 1e3, double xtol = 1e-12, double rtol = 1e-6);
        std::tuple<double, double> getSaddleNewton_secur(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0 = 0, double tol = 1e-9);

        // with elements
        double K (double s, arma::vec pd_c);
        double K1(double s, arma::vec pd_c);
        double K2(double s, arma::vec pd_c);

        std::tuple<double, double, double> K012(double s, arma::vec pd_c);
        std::tuple<double, double>         K12(double  s, arma::vec pd_c);

        double getSaddle(double loss, arma::vec pd_c, double s0 = 0, double a = -1e12, double b = 1e12, double tol = 1e-7);
        double getSaddleBrent(double loss, arma::vec pd_c, double a = -1e3, double b = 1e3, double tol = 1e-9);
        std::tuple<double, double> getSaddleNewton(double loss, arma::vec pd_c, double s0 = 0, double tol = 1e-9);

        // Optimizer

        double cdf(double loss, arma::vec n, arma::vec eadxlgd, arma::mat pd_c,
                   CreditRisk::Integrator::PointsAndWeigths * points = new CreditRisk::Integrator::PointsAndWeigths(CreditRisk::Integrator::ghi()),
                   size_t p = std::thread::hardware_concurrency());

        double quantile(double prob, arma::vec n, arma::vec eadxlgd, arma::mat pd_c,
                        CreditRisk::Integrator::PointsAndWeigths * points = new CreditRisk::Integrator::PointsAndWeigths(CreditRisk::Integrator::ghi()),
                        double xtol = 1e-12, double rtol = 1e-6, size_t p = std::thread::hardware_concurrency());

        arma::vec getContrib_without_secur(double loss,  arma::vec n, arma::vec eadxlgd, arma::mat pd_c,
                                           CreditRisk::Integrator::PointsAndWeigths * points = new CreditRisk::Integrator::PointsAndWeigths(CreditRisk::Integrator::ghi()),
                                           size_t p = std::thread::hardware_concurrency());

        arma::vec getContrib(double loss,  arma::vec n, arma::vec eadxlgd, arma::mat pd_c,
                             CreditRisk::Integrator::PointsAndWeigths * points = new CreditRisk::Integrator::PointsAndWeigths(CreditRisk::Integrator::ghi()),
                             size_t p = std::thread::hardware_concurrency());

        double EVA(arma::vec eadxlgd, arma::vec contrib);

    };
}

#endif // CREDIT_PORTFOLIO_H