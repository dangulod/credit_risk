#include "fund.h"

void checkPurchase(double purchase)
{
    if (purchase < 0 | purchase > 1)
        throw std::invalid_argument("purchase mus be between [0-1]");
}

namespace CreditRisk
{
Fund::Fund(string name, double purchase, double at, double de) : Portfolio(name), fundParam({{purchase, at, de}})
{
    checkPurchase(purchase);
}

Fund::Fund(string name, FundParam value) : Portfolio(name), fundParam({value})
{
    checkPurchase(value.purchase);
}

Fund::Fund(string name, vector<FundParam> value): Portfolio(name), fundParam(value)
{
    for (auto ii: value)
    {
        checkPurchase(ii.purchase);
    }
};

Fund::Fund(string name, double CtI, double HR, double rf, double tax, double purchase, double at, double de) :
    Portfolio(name, CtI, HR, rf, tax), fundParam({{purchase, at, de}})
{
    checkPurchase(purchase);
}

Fund::Fund(string name, double CtI, double HR, double rf, double tax, FundParam value):
    Portfolio(name, CtI, HR, rf, tax), fundParam({value})
{
    checkPurchase(value.purchase);
}

Fund::Fund(string name, double CtI, double HR, double rf, double tax, vector<FundParam> value) :
    Portfolio(name, CtI, HR, rf, tax), fundParam(value)
{
    for (auto ii: value)
    {
        checkPurchase(ii.purchase);
    }
}

pt::ptree Fund::to_ptree()
{
    pt::ptree root;
    pt::ptree array;

    for (auto & ii: this->fundParam)
    {
        pt::ptree param;
        param.put("purchase", ii.purchase);
        param.put("attachment", ii.at);
        param.put("detachment", ii.de);

        array.push_back(std::make_pair("", param));
    }

    root.add_child("ParamFounds", array);

    Portfolio::to_ptree(root);

    return root;
}

Fund Fund::from_ptree(pt::ptree & value)
{
    std::vector<FundParam> fundParam;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("ParamFounds"))
    {
        double purchase = ii.second.get<double>("purchase");
        double at = ii.second.get<double>("attachment");
        double de = ii.second.get<double>("detachment");
        fundParam.push_back({purchase, at, de});
    }

    Fund f(value.get<string>("name"), value.get<double>("CtI"), value.get<double>("HR"), value.get<double>("rf"), value.get<double>("tax"), fundParam);

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("Portfolio"))
    {
        pt::ptree pt = ii.second;
        f + CreditRisk::Element::Element::from_ptree(pt);
    }

    return f;
}

double Fund::loss_sec(arma::vec f, unsigned long idio_id)
{
    double loss_wf = this->loss(f, idio_id, false);
    double loss    = 0;

    for (auto & ii: this->fundParam)
    {
        loss += fmax(fmin(loss_wf - ii.at, ii.de - ii.at), 0) * ii.purchase;
    }

    return loss;
}

double Fund::loss_sec(double t, arma::vec  f, unsigned long idio_id)
{
    double loss_wf = this->loss(t, f, idio_id);
    double loss    = 0;

    for (auto & ii: this->fundParam)
    {
        loss += fmax(fmin(loss_wf - ii.at, ii.de - ii.at), 0) * ii.purchase;
    }

    return loss;
}

arma::vec Fund::get_T_at(arma::vec f, unsigned long idio_id, double tol)
{
    //return root_secant(&Fund::__fit_T_loss, *this, 0, 1, tol, true, this->at, f, idio_id);
    //return root_Brentq(&Fund::__fit_T_loss, *this, 0, 1, 1e-12, 1e-6, true, this->at, f, idio_id);
    arma::vec at(this->fundParam.size());

    for (size_t ii = 0; ii < this->fundParam.size(); ii++)
    {
        at.at(ii) = CreditRisk::Optim::root_bisection(&Fund::__fit_T_loss, *this, 0, 1, tol, true, this->fundParam[ii].at, f, idio_id);
    }

    return at;
}

arma::vec Fund::get_T_de(arma::vec f, unsigned long idio_id, double tol)
{
    arma::vec de(this->fundParam.size());

    for (size_t ii = 0; ii < this->fundParam.size(); ii++)
    {
        // de.at(ii) = root_secant(&Fund::__fit_T_loss, *this, 0, 1, tol, false, this->fundParam[ii].de, f, idio_id);
        // de.at(ii) = root_Brentq(&Fund::__fit_T_loss, *this, 0, 1, 1e-12, 1e-6, false, this->fundParam[ii].de, f, idio_id);
        de.at(ii) = CreditRisk::Optim::root_bisection(&Fund::__fit_T_loss, *this, 0, 1, tol, false, this->fundParam[ii].de, f, idio_id);
    }

    return de;
}

arma::vec Fund::get_T_at(arma::vec cwi, arma::vec v_t, double tol)
{
    arma::vec at(this->fundParam.size());

    for (size_t ii = 0; ii < this->fundParam.size(); ii++)
    {
        // at.at(ii) = root_secant(&Fund::__fit_T_loss, *this, 0, 1, tol, true, this->fundParam[ii].de, cwi, v_t);
        // at.at(ii) = root_Brentq(&Fund::__fit_T_loss_t, *this, 0, 1, 1e-12, 1e-6, 50, true, this->fundParam[ii].at, cwi, v_t);
        at.at(ii) = CreditRisk::Optim::root_bisection(&Fund::__fit_T_loss_t, *this, 0, 1, tol, true, this->fundParam[ii].at, cwi, v_t);
    }

    return at;
}
arma::vec Fund::get_T_de(arma::vec cwi, arma::vec v_t, double tol)
{
    arma::vec de(this->fundParam.size());

    for (size_t ii = 0; ii < this->fundParam.size(); ii++)
    {
        // de.at(ii) = root_secant(&Fund::__fit_T_loss_t, *this, 0, 1, tol, false, this->fundParam[ii].de, cwi, v_t);
        // de.at(ii) = root_Brentq(&Fund::__fit_T_loss_t, *this, 0, 1, 1e-12, 1e-6, 50, false, this->fundParam[ii].de, cwi, v_t);
        de.at(ii) = CreditRisk::Optim::root_bisection(&Fund::__fit_T_loss_t, *this, 0, 1, tol, false, this->fundParam[ii].de, cwi, v_t);
    }

    return de;
}

double Fund::fit_t_sad(double t, double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double k1s, double scenario, size_t id)
{
    double k1 = 0;

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        pd_c->at(ii + id) = this->at(ii).p_states_c(t, scenario);
    }

    for (size_t ii = id; ii < this->size() + id; ii++)
    {
        k1 += saddle::K1(s, n->at(ii), eadxlgd->at(ii), pd_c->at(ii));
    }

    return pow(k1 - k1s, 2);
}

double Fund::get_T_saddle(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double k1s, double scenario, size_t id)
{
    return Optim::root_Brent(&Fund::fit_t_sad, (*this), 1e-9, 1, 1e-9, s, n, eadxlgd, pd_c, k1s, scenario, id);
}

void Fund::check_fund_data(Fund_data & data, arma::vec f, unsigned long idio_id)
{
    if (data.at(0).t_at < 0)
    {
        arma::vec cwi = this->get_cwi(f, idio_id);
        arma::vec v_t = this->get_t(cwi);

        arma::vec t_at = this->get_T_at(cwi, v_t);
        arma::vec t_de = this->get_T_de(cwi, v_t);

        for (size_t ii = 0; ii < data.size(); ii++)
        {
            data.at(ii).t_at = t_at.at(ii);
            data.at(ii).t_de = t_de.at(ii);

            data.at(ii).l_at = this->__fit_T_loss_t(data.at(ii).t_at, this->fundParam[ii].at, cwi, v_t);
            data.at(ii).l_de = this->__fit_T_loss_t(data.at(ii).t_de, this->fundParam[ii].de, cwi, v_t);
        }
    }
}

}
