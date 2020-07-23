#ifndef ELEMENT_H
#define ELEMENT_H

#include <string>
#include "equation.h"
#include "transition.h"
#include "spread.h"

using std::string;

class Mig
{
    bool m_mig;
public:
    arma::vec p_states, l_states;

    Mig() : m_mig(false) {};
    Mig(arma::vec states, arma::vec le) : m_mig(false), p_states(states), l_states(le)
    {
        if (p_states.size() != le.size())
        {
            throw std::invalid_argument("states and losses does not have the same size");
        }

        if (this->size() > 0) this->m_mig= true;
    }

    bool migration()
    {
        return this->m_mig;
    }

    size_t size()
    {
        return this->p_states.size();
    }
};

namespace CreditRisk
{
    class Credit_param
    {
    public:
        unsigned long n;
        double ead, pd_b, pd, lgd, lgd_addon, beta, t, idio, _npd, _le;
        double spread_old, spread_new;

        Credit_param() = delete;
        Credit_param(unsigned long n, double ead, double pd_b, double pd, double lgd, double lgd_addon, double beta, double t);
        Credit_param(unsigned long n, double ead, double pd_b, double pd, double lgd, double lgd_addon, double beta, double t, double spread_old, double spread_new);
        Credit_param(const Credit_param & value) = delete;
        Credit_param(Credit_param && value) = default;
        ~Credit_param() = default;
    };

    class Element : public Credit_param
    {
    public:
        Mig _states;
        enum class Treatment{
            Wholesale = 0,
            Retail    = 1
        };

        unsigned long ru;
        Treatment mr;
        CreditRisk::Equation equ;

        Element() = delete;
        Element(unsigned long ru, unsigned long n, double ead, double pd_b, double pd, double lgd,
                double lgd_addon, double beta,  double t, Treatment mr, CreditRisk::Equation equ);
        Element(unsigned long ru, unsigned long n, double ead, double pd_b, double pd, double lgd,
                double lgd_addon, double beta, double t, double spread_new, double spread_old,
                Treatment mr, CreditRisk::Equation equ);
        Element(const Element & value) = delete;
        Element(Element && value) = default;
        ~Element() = default;

        std::string getTreatment();
        void setTreatment(std::string value);

        void setN(unsigned long value);
        void setLgd(double value);
        void setLgdAddon(double value);
        void setEad(double value);
        void setPD(double value);
        void setPD_B(double value);
        void setBeta(double value);

        pt::ptree to_ptree();
        static Element from_ptree(pt::ptree & value);

        double pd_c(double t, double cwi);
        double pd_c(double cwi);

        // Migration
        void setMigration(Transition * tr, Spread * sp, double rf);
        arma::vec p_states_c(double t, double cwi);
        arma::vec p_states_c(double cwi);
        arma::vec l_states();

        double el();
        double getT(double cwi);

        double loss(double t, arma::vec f, unsigned long id);
        double loss(double t, arma::vec f, double idio);
        double loss(double t, double cwi);

        double loss(double t, arma::vec cwi, arma::vec v_t, size_t id_t);

        double loss(arma::vec f, unsigned long id, bool migration = true);
        double loss(arma::vec f, double idio, bool migration = true);
        double loss(double cwi, bool migration = true);

        double EVA(double eadxlgd, double CeR, double cti, double rf, double tax, double hr);
    };
}
#endif // ELEMENT_H
