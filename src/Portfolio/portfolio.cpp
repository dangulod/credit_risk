#include "portfolio.h"

namespace CreditRisk
{
Portfolio::~Portfolio() {}

Portfolio::Portfolio(string name) : name(name), CtI(0), HR(0), rf(0), tax(0) {}

Portfolio::Portfolio(string name, double CtI, double HR, double rf, double tax) :
    name(name), CtI(CtI), HR(HR), rf(rf), tax(tax) {}

pt::ptree Portfolio::to_ptree()
{
    pt::ptree root;

    this->to_ptree(root);

    return root;
}

void Portfolio::to_ptree(pt::ptree & ptree)
{
    pt::ptree array;

    ptree.put("name", this->name);
    ptree.put("CtI", this->CtI);
    ptree.put("HR", this->HR);
    ptree.put("rf", this->rf);
    ptree.put("tax", this->tax);

    for (auto & ii: *this)
    {
        array.push_back(std::make_pair("", ii.to_ptree()));
    }

    ptree.add_child("Portfolio", array);
}

Portfolio Portfolio::from_ptree(pt::ptree & value)
{
    Portfolio p(value.get<string>("name"), value.get<double>("CtI"), value.get<double>("HR"), value.get<double>("rf"), value.get<double>("tax"));

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("Portfolio"))
    {
        pt::ptree pt = ii.second;
        p + CreditRisk::Element::Element::from_ptree(pt);
    }

    return p;
}

void Portfolio::operator+(CreditRisk::Element && v)
{
    this->push_back(std::move(v));
}

void Portfolio::operator+(CreditRisk::Element & v)
{
    this->push_back(std::move(v));
}

double Portfolio::getT_EADxLGD()
{
    double total = 0;

    for (auto & ii: *this)
    {
        total += ii._le;
    }

    return total;
}

double Portfolio::getT_EAD()
{
    double total = 0;

    for (auto & ii: *this)
    {
        total += ii.ead * ii.n;
    }

    return total;
}

double Portfolio::get_PE()
{
    double pe = 0;

    for (auto & ii: (*this))
    {
        pe += ii.pd * ii._le;
    }

    return pe;
}

double Portfolio::loss(arma::vec  f, unsigned long idio_id, bool migration)
{
    double loss(0);
    for (auto & ii: *this)
    {
        loss += ii.loss(f, idio_id, migration);
    }

    return loss;
}

double Portfolio::loss(double t, arma::vec  f, unsigned long idio_id)
{
    double loss(0);
    for (auto & ii: *this)
    {
        loss += ii.loss(t, f, idio_id);;
    }

    return loss;
}

double Portfolio::loss(double t, arma::vec cwi, arma::vec v_t)
{
    double loss(0);

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        loss += (*this)[ii].loss(t, cwi, v_t, ii);
    }

    return loss;
}

arma::vec Portfolio::get_cwi(arma::vec f, unsigned long idio_id)
{
    arma::vec cwi(this->size());
    auto jj = cwi.begin();

    for (auto & ii: *this)
    {
        (*jj) = ii.equ.CWI(f, idio_id);
        jj++;
    }

    return cwi;
}

arma::vec Portfolio::get_t(arma::vec cwi)
{
    arma::vec t(this->size());

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        t[ii] = (*this)[ii].getT(cwi[ii]);
    }

    return t;
}

double Portfolio::__fit_T_loss(double t, double loss0, arma::vec f, unsigned long idio_id)
{
    return this->loss(t, f, idio_id) - loss0;
}

double Portfolio::__fit_T_loss_t(double t, double loss0, arma::vec cwi, arma::vec v_t)
{
    return this->loss(t, cwi, v_t) - loss0;
}
}
