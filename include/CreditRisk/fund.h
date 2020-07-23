#ifndef FUND_H
#define FUND_H

#include "portfolio.h"
#include "integrator.h"

struct FundParam
{
    double purchase, at, de;
};

namespace CreditRisk
{
    struct Tranche_data
    {
        double p, t_at, t_de, l_at, l_de;
        Tranche_data() : p(1), t_at(-1), t_de(-1), l_at(0), l_de(0) {}
        Tranche_data(double p) : p(p), t_at(-1), t_de(-1), l_at(0), l_de(0) {}
    };

    typedef std::vector<Tranche_data> Fund_data;
    typedef std::vector<Fund_data> Scenario_data;

    class Fund : public CreditRisk::Portfolio::Portfolio
    {
    public:
        std::vector<FundParam> fundParam;

        Fund() = delete;
        Fund(string name, double purchase, double at, double de);
        Fund(string name, FundParam value);
        Fund(string name, std::vector<FundParam> value);
        Fund(string name, double CtI, double HR, double rf, double tax, double purchase, double at, double de);
        Fund(string name, double CtI, double HR, double rf, double tax, FundParam value);
        Fund(string name, double CtI, double HR, double rf, double tax, std::vector<FundParam> value);
        Fund(const Fund & value) = delete;
        Fund(Fund && value) = default;
        ~Fund() = default;

        pt::ptree to_ptree();
        static Fund from_ptree(pt::ptree & value);

        double loss_sec(arma::vec  f, unsigned long idio_id);
        double loss_sec(double t, arma::vec  f, unsigned long idio_id);

        arma::vec get_T_at(arma::vec f, unsigned long idio_id, double tol = 1e-12);
        arma::vec get_T_de(arma::vec f, unsigned long idio_id, double tol = 1e-12);

        arma::vec get_T_at(arma::vec cwi, arma::vec v_t, double tol = 1e-12);
        arma::vec get_T_de(arma::vec cwi, arma::vec v_t, double tol = 1e-12);

        double fit_t_sad(double t, double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double k1s, double scenario, size_t id);
        double get_T_saddle(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double k1s, double scenario, size_t id);

        void check_fund_data(Fund_data & data, arma::vec f, unsigned long idio_id);
    };
}

#endif // FUND_H
