#include "element.h"

namespace CreditRisk
{

Credit_param::Credit_param(unsigned long n, double ead, double pd_b, double pd, double lgd, double lgd_addon, double beta):
    n(n), ead(ead), pd_b(pd_b), pd(pd), lgd(lgd), lgd_addon(lgd_addon), beta(beta), spread_old(0), spread_new(0)
{
    CreditRisk::Utils::isProbability(pd_b);
    CreditRisk::Utils::isProbability(pd);
    CreditRisk::Utils::isProbability(lgd);
    CreditRisk::Utils::isProbability(lgd_addon);
    CreditRisk::Utils::isProbability(fabs(beta));

    this->idio = sqrt(1 - pow(beta, 2));
    this->_npd = CreditRisk::Utils::qnorm(pd_b);
    this->_le  = this->n * this->lgd_addon * this->ead;
}

Credit_param::Credit_param(unsigned long n, double ead, double pd_b, double pd, double lgd, double lgd_addon, double beta, double spread_old, double spread_new):
    n(n), ead(ead), pd_b(pd_b), pd(pd), lgd(lgd), lgd_addon(lgd_addon), beta(beta), spread_old(spread_old), spread_new(spread_new)
{
    CreditRisk::Utils::isProbability(pd_b);
    CreditRisk::Utils::isProbability(pd);
    CreditRisk::Utils::isProbability(lgd);
    CreditRisk::Utils::isProbability(lgd_addon);
    CreditRisk::Utils::isProbability(fabs(beta));

    this->idio = sqrt(1 - pow(beta, 2));
    this->_npd = CreditRisk::Utils::qnorm(pd_b);
    this->_le  = this->n * this->lgd_addon * this->ead;
}

Element::Element(unsigned long ru, unsigned long n, double ead, double pd_b, double pd, double lgd, double lgd_addon, double beta, Treatment mr, CreditRisk::Equation equ):
    Credit_param(n, ead, pd_b, pd, lgd, lgd_addon, beta), ru(ru), mr(mr), equ(std::move(equ)) {}

Element::Element(unsigned long ru, unsigned long n, double ead, double pd_b, double pd, double lgd, double lgd_addon, double beta, double spread_old, double spread_new,
        Treatment mr, CreditRisk::Equation equ) :
    Credit_param(n, ead, pd_b, pd, lgd, lgd_addon, beta, spread_old, spread_new), ru(ru), mr(mr), equ(std::move(equ)) {}

std::string Element::getTreatment()
{
    return this->mr == CreditRisk::Element::Treatment::Retail ? "retail" : "wholesale";
}

void Element::setTreatment(std::string value)
{
    if (value == "retail") this->mr = CreditRisk::Element::Treatment::Retail;
    if (value == "wholesale") this->mr = CreditRisk::Element::Treatment::Wholesale;
}

void Element::setN(unsigned long value)
{
    if (this->n != value)
    {
        this->n = value;
        this->_le  = this->n * this->lgd_addon * this->ead;
    }
}

void Element::setLgd(double value)
{
    if (this->lgd != value)
    {
        CreditRisk::Utils::isProbability(value);
        this->lgd = value;
    }
}

void Element::setLgdAddon(double value)
{
    if (this->lgd_addon != value)
    {
        CreditRisk::Utils::isProbability(lgd_addon);
        this->lgd_addon = value;
        this->_le  = this->n * this->lgd_addon * this->ead;
    }
}

void Element::setEad(double value)
{
    if (this->ead != value)
    {
        this->ead = value;
        this->_le  = this->n * this->lgd_addon * this->ead;
    }
}

void Element::setPD(double value)
{
    if (this->pd != value)
    {
        CreditRisk::Utils::isProbability(this->pd);
        this->pd = value;
    }
}

void Element::setPD_B(double value)
{
    if (this->pd_b != value)
    {
        CreditRisk::Utils::isProbability(pd_b);
        this->pd_b = value;
        this->_npd = CreditRisk::Utils::qnorm(pd_b);
    }
}

void Element::setBeta(double value)
{
    if (this->beta != value)
    {
        CreditRisk::Utils::isProbability(value);
        this->beta = value;
        this->idio = sqrt(1 - value);
    }
}

pt::ptree Element::to_ptree()
{
    pt::ptree root;
    root.put("RU", this->ru);
    root.put("n", this->n);
    root.put("ead", this->ead);
    root.put("pd_b", this->pd_b);
    root.put("pd", this->pd);
    root.put("lgd", this->lgd);
    root.put("lgd_addon", this->lgd_addon);
    root.put("beta", this->beta);
    root.put("spread_old", spread_old);
    root.put("spread_new", spread_new);
    root.put("Treatment", (this->mr == Treatment::Wholesale) ? "wholesale" : "retail");
    root.add_child("equation", this->equ.to_ptree());

    return root;
}

Element Element::from_ptree(pt::ptree & value)
{
    return Element(
                value.find("RU")->second.get_value<unsigned long>(),
                value.find("n")->second.get_value<unsigned long>(),
                value.find("ead")->second.get_value<double>(),
                value.find("pd_b")->second.get_value<double>(),
                value.find("pd")->second.get_value<double>(),
                value.find("lgd")->second.get_value<double>(),
                value.find("lgd_addon")->second.get_value<double>(),
                value.find("beta")->second.get_value<double>(),
                value.find("spread_old")->second.get_value<double>(),
                value.find("spread_new")->second.get_value<double>(),
                value.find("Treatment")->second.get_value<string>() == "wholesale" ? Treatment::Wholesale : Treatment::Retail,
                CreditRisk::Equation::Equation::from_ptree(value.get_child("equation")));
}

double Element::el()
{
    return this->pd * this->_le;
}

double Element::pd_c(double cwi)
{
    return CreditRisk::Utils::pnorm((this->_npd - (this->beta * cwi)) / this->idio);
}

double Element::pd_c(double t, double cwi)
{
    if (t == 1) return CreditRisk::Utils::pnorm((this->_npd - (this->beta * cwi)) / this->idio);
    return CreditRisk::Utils::pnorm((CreditRisk::Utils::qnorm(1 - pow(1 - this->pd, t)) - (this->beta * cwi)) / this->idio);
}

double Element::loss(double cwi)
{
    switch (this->mr)
    {
    case Treatment::Retail :
        return pd_c(cwi) * this->_le;
        break;
    case Treatment::Wholesale:
        return (this->_npd > cwi) * _le;
        break;
    }

    return 0;
}

double Element::loss(arma::vec f, unsigned long id)
{
    double cwi = this->equ.CWI(f, id);

    return this->loss(cwi);
}

double Element::loss(arma::vec f, double idio)
{
    double cwi = this->equ.CWI(f, idio);

    return this->loss(cwi);
}

double Element::loss(double t, arma::vec f, unsigned long id)
{
    double cwi = this->equ.CWI(f, id);

    return this->loss(t, cwi);
}

double Element::loss(double t, arma::vec f, double idio)
{
    double cwi = this->equ.CWI(f, idio);

    return this->loss(t, cwi);
}

double Element::loss(double t, double cwi)
{
    switch (this->mr)
    {
    case Treatment::Retail :
        return pd_c(t, cwi) * this->_le;
        break;
    case Treatment::Wholesale:
        return (this->getT(cwi) < t) * _le;
        break;
    }

    return 0;
}

double Element::loss(double t, arma::vec cwi, arma::vec v_t, size_t id_t)
{
    switch (this->mr)
    {
    case Treatment::Retail :
        return pd_c(t, cwi[id_t]) * this->_le;
        break;
    case Treatment::Wholesale:
        return (v_t[id_t] < t) * _le;
        break;
    }
}

double Element::getT(double cwi)
{
    switch (this->mr)
    {
    case Treatment::Retail :
        return 0;
        break;
    case Treatment::Wholesale:
        return log(1 - CreditRisk::Utils::pnorm(cwi)) / log(1 - this->pd_b);
        break;
    }

    return log(1 - CreditRisk::Utils::pnorm(cwi)) / log(1 -this->pd_b);
}

double Element::num(double s, double pd_c)
{
    return (s < 0) ? pd_c * this->_le * exp(s * this->_le) : pd_c * this->_le;
}

double Element::den(double s, double pd_c)
{
    return (s < 0) ? (1 - pd_c) + pd_c * exp(s * this->_le) : (1 - pd_c) * exp(-s * this->_le) + pd_c;
}

double Element::K(double s, double pd_c)
{
    return this->n * log(den(s, pd_c)) + (s < 0 ? 0 : s * this->_le);
}
double Element::K1(double s, double pd_c)
{
    return this->n * num(s, pd_c) / den(s, pd_c);
}
double Element::K2(double s, double pd_c)
{
    double num(this->num(s, pd_c)), den(this->den(s, pd_c));

    return this->n * ((num * this->_le) / den - (num * num) / (den * den));
}

double Element::EVA(double eadxlgd, double CeR, double cti, double rf, double tax, double hr)
{
    double t_ead = eadxlgd / this->lgd_addon;
    double n_ead = fmax(t_ead - this->ead, 0);
    double o_ead = (n_ead > 0) ? this->ead : t_ead;
    double el    = t_ead * this->pd * this->lgd;
    CeR -= el;
    return ((((this->spread_new * 12 * n_ead) + (this->spread_old * 12 * o_ead)) * (1 - cti) - el + rf * CeR) * (1 - tax) - (hr * CeR)) * this->n;
}
}
