#include "CreditRisk/equation.h"

namespace CreditRisk
{
Equation::Equation(unsigned long idio_seed, arma::vec weigths) :
    idio_seed(idio_seed), weights(weigths)
{
    this->R2   = arma::accu(arma::pow(weigths, 2));
    if (this->R2 > 1) throw std::invalid_argument("Invalid equations weights R2 greater than 1");
    this->idio = sqrt(1 - R2);
}

pt::ptree Equation::to_ptree()
{
    pt::ptree root;

    root.put("idio_seed", this->idio_seed);

    pt::ptree weights;

    for (auto & ii: this->weights)
    {
        pt::ptree w;
        w.put("", ii);

        weights.push_back(std::make_pair("", w));
    }

    root.add_child("weights", weights);

    return root;
}


Equation Equation::from_ptree(pt::ptree & value)
{
    arma::vec w(value.get_child("weights").size());
    size_t jj = 0;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("weights"))
    {
        w.at(jj) = ii.second.get_value<double>();
        jj++;
    }

    return Equation(value.find("idio_seed")->second.get_value<unsigned long>(), w);
}

double Equation::CWI(arma::vec f, unsigned long id)
{
    double d = CreditRisk::Utils::randn_s(this->idio_seed + id);

    return CWI(f, d);
}

double Equation::CWI(arma::vec f, double i)
{
    return this->systematic(f) + this->idio * i;
}

double Equation::systematic(arma::vec f)
{
    return arma::accu(f % this->weights);
}

void Equation::setIdio(arma::mat cor)
{
    this->R2   = arma::as_scalar(this->weights.t() * cor * this->weights);
    if (this->R2 > 1) throw std::invalid_argument("Invalid equations weights R2 greater than 1");
    this->idio = sqrt(1 - R2);
}

}
