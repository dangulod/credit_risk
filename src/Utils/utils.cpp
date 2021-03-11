#include "CreditRisk/utils.h"

namespace CreditRisk
{
namespace Utils
{

size_t number_of_lines(std::string file)
{
    std::ifstream input(file);

    if (input.is_open())
    {
        std::string buffer;
        std::vector<std::string> splitted;

        std::getline(input, buffer);
        boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });
        size_t ll = splitted.size();
        size_t ii = 1;

        while (std::getline(input, buffer))
        {
            boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

            if (splitted.size() == ll)
            {
                ii++;
            }
        }
        return ii;
    } else
    {
        throw std::invalid_argument("File can not be opened");
    }
}

void isProbability(double p)
{
    if (p < 0 | p > 1 | !std::isfinite(p))
    {
        throw std::invalid_argument("Invalid parameter not contain in [0-1]");
    }
}

double qnorm(double p)
{
    isProbability(p);
    double result;

    if (fabs(p - 0) < DBL_MIN) return static_cast<double>(-INFINITY);
    if (fabs(p - 1) < DBL_MIN) return static_cast<double>(INFINITY);

    result = boost::math::erfc_inv(2 * p);

    return -result * M_SQRT2;

}

arma::vec qnorm(const arma::vec & p)
{
    arma::vec r = p;

    for (auto & ii: r)
    {
        ii = qnorm(ii);
    }

    return r;
}

double pnorm(double x)
{
    double result;
    double diff = x / M_SQRT2;
    result = boost::math::erfc(-diff) / 2;
    return result;
}

arma::vec pnorm(const arma::vec & x)
{
    arma::vec r = x;

    for (auto & ii: r)
    {
        ii = pnorm(ii);
    }

    return r;
}

double randn_s()
{
    std::mt19937_64 generator;

    double numerator = generator();
    double divisor = static_cast<double>(generator.max());
    double p = numerator / divisor;

    return qnorm(p);
}


double randn_s(unsigned long seed)
{
    std::mt19937_64 generator;
    generator.seed(seed);

    double numerator = generator();
    double divisor = static_cast<double>(generator.max());
    double p = numerator / divisor;

    return qnorm(p);
}

arma::vec randn_v(size_t n, unsigned long seed)
{
    arma::vec ale(n);
    double numerator, divisor, p;

    std::mt19937 generator;
    generator.seed(seed);

    for (size_t ii = 0; ii < n; ii++)
    {
        numerator = generator();
        divisor = static_cast<double>(generator.max());
        p = numerator / divisor;

        ale.at(ii) = qnorm(p);
    }

    return ale;
}

double mean(const arma::vec & x)
{
    double mean = 0;

    for (auto & ii: x)
    {
        mean += ii;
    }

    return mean / x.size();
}

double quantile(arma::vec x, double q)
{
    if (q < 0 | q > 1) throw std::invalid_argument("Invalid percentile value");
    x = arma::sort(x);

    int n = q * x.size();

    double y1 = x.at(static_cast<size_t>(q * x.size()));

    if (static_cast<size_t>(q * x.size() + 1) >= x.size())
    {
        return y1;
    }

    double y2 = x.at(static_cast<size_t>(q * x.size() + 1));

    double x1 = (static_cast<double>(n) - 0.5) / x.size();
    double x2 = (static_cast<double>(n) + 0.5) / x.size();

    return (y1 * (x2 - q) + y2 * (q - x1)) / (x2 - x1);
}

arma::vec contributions(const arma::mat & x, double q, double lower, double upper)
{
    arma::vec total(x.n_rows);

    for (size_t ii = 0; ii < total.size(); ii++)
    {
        total.at(ii) = arma::accu(x.row(ii));
    }

    arma::uvec rank = arma::sort_index(total);

    arma::vec contrib(x.n_cols, arma::fill::zeros);

    size_t l = (x.n_rows * lower);
    size_t u = (x.n_rows * upper);

    for (size_t ii = 0; ii < rank.size(); ii++)
    {
        if ((rank.at(ii) > l) & (rank.at(ii) < u))
        {
            for (size_t jj = 0; jj < contrib.size(); jj++)
            {
                contrib.at(jj) += x.at(ii, jj);
            }
        }
    }

    std::cout << std::endl;

    contrib /= (u - l);

    std::cout << std::endl;

    contrib /= arma::accu(contrib);
    contrib *= quantile(total, q);

    return contrib;
}

arma::vec rowSum(const arma::mat & x)
{
    arma::vec total(x.n_rows);

    for (size_t ii = 0; ii < x.n_rows; ii++)
    {
        total.at(ii) = arma::accu(x.row(ii));
    }

    return total;
}

}

namespace saddle
{
    double p_c(double p, double beta, double idio, double cwi)
    {
        return CreditRisk::Utils::pnorm((p - (beta * cwi)) / idio);
    }

    double p_c(double t, double p, double beta, double idio, double cwi)
    {
        return CreditRisk::saddle::p_c(CreditRisk::Utils::qnorm(1 - pow(1 - p, t)), beta, idio, cwi);
    }

    arma::vec p_states_c(arma::vec & p_states, double npd, double beta, double idio, double cwi, bool migration)
    {
        double pb = p_c(npd, beta, idio, cwi);

        if (!migration) return {1 - pb, pb};

        arma::vec pp(p_states.size() + 2);

        pp.back() = pb;
        double pa = pb;

        for (size_t ii =  p_states.size(); ii > 0; ii--)
        {
            pa = CreditRisk::Utils::pnorm((p_states.at(ii - 1) - (beta * cwi)) / idio);
            pp.at(ii) = pa - pb;
            pb = pa;
        }
        pp.front() = 1 - pa;

        return pp;
    }

    arma::vec p_states_c(double t, arma::vec & p_states, double pd, double beta, double idio, double cwi, bool migration)
    {
        double pb = p_c(t, pd, beta, idio, cwi);

        if (!migration) return {1 - pb, pb};

        arma::vec pp(p_states.size() + 2);

        pp.back() = pb;
        double pa = pb;

        for (size_t ii =  p_states.size(); ii > 0; ii--)
        {
            pa = CreditRisk::Utils::pnorm((p_states.at(ii - 1) - (beta * cwi)) / idio);
            pp.at(ii) = pa - pb;
            pb = pa;
        }
        pp.front() = 1 - pa;

        return pp;
    }

    double num(double s, arma::vec & l_states, arma::vec & p_states)
    {
        double num = 0;

        for (size_t ii = 1; ii < l_states.size(); ii++)
        {
            num += p_states.at(ii) * l_states.at(ii) * exp(s * (l_states.at(ii) - (s > 0) * l_states.back()));
        }

        return num;
    }

    double num2(double s, arma::vec & l_states, arma::vec & p_states)
    {
        double num2 = 0;

        for (size_t ii = 1; ii < l_states.size(); ii++)
        {
            num2 += p_states.at(ii) * pow(l_states.at(ii), 2) * exp(s * (l_states.at(ii) - (s > 0) * l_states.back()));
        }

        return num2;
    }

    double den(double s, arma::vec & l_states, arma::vec & p_states)
    {
        double den = 0;

        for (size_t ii = 0; ii < l_states.size(); ii++)
        {
            den += p_states.at(ii) * exp(s * (l_states.at(ii) - (s > 0) * l_states.back()));
        }

        return den;
    }

    double K(double s, unsigned long n, arma::vec & l_states, arma::vec & p_states)
    {
        return n * log(den(s, l_states, p_states)) + (s < 0 ? 0 : s * l_states.back());
    }

    double K1(double s, unsigned long n, arma::vec & l_states, arma::vec & p_states)
    {
        return n * num(s, l_states, p_states) / den(s, l_states, p_states);
    }

    double K2(double s, unsigned long n, arma::vec & l_states, arma::vec & p_states)
    {
        double dnum(num(s, l_states, p_states)), dden(den(s, l_states, p_states)),
                dnum2(num2(s, l_states, p_states));

        return n * (dnum2 / dden - (dnum * dnum) / (dden * dden));
    }

    /*
    double num(double s, double _le, double pd_c)
    {
        return (s < 0) ? pd_c * _le * exp(s * _le) : pd_c * _le;
    }

    double den(double s, double _le, double pd_c)
    {
        return (s < 0) ? (1 - pd_c) + pd_c * exp(s * _le) : (1 - pd_c) * exp(-s * _le) + pd_c;
    }

    double K(double s, unsigned long n, double _le, double pd_c)
    {
        return n * log(den(s, _le, pd_c)) + (s < 0 ? 0 : s * _le);
    }
    double K1(double s, unsigned long n, double _le, double pd_c)
    {
        return n * num(s, _le, pd_c) / den(s, _le, pd_c);
    }
    double K2(double s, unsigned long n, double _le, double pd_c)
    {
        double dnum(num(s, _le, pd_c)), dden(den(s, _le, pd_c));

        return n * ((dnum * _le) / dden - (dnum * dnum) / (dden * dden));
    }
    */

    /*
    std::tuple<double, double, double> K012(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
    {
        std::tuple<double, double, double> k012(0, 0, 0);

        double dnum, dnum2, dden, k1;

        for (size_t ii = 0; ii < n->size(); ii++)
        {
            dden = saddle::den(s, eadxlgd->at(ii), pd_c->at(ii));
            dnum = saddle::num(s, eadxlgd->at(ii), pd_c->at(ii));
            dnum2 = saddle::num2(s, eadxlgd->at(ii), pd_c->at(ii));
            k1 = dnum / dden;

            std::get<0>(k012) += n->at(ii) * (log(dden) + (s < 0 ? 0 : s * eadxlgd->at(ii).back()));
            std::get<1>(k012) += n->at(ii) * k1;
            std::get<2>(k012) += n->at(ii) * ((dnum2 / dden) - pow(k1, 2));
        }

        return k012;
    }

    std::tuple<double, double>         K12(double  s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
    {
        std::tuple<double, double> k12(0, 0);

        double dnum, dnum2, dden, k1;

        for (size_t ii = 0; ii < n->size(); ii++)
        {
            dnum = saddle::num(s, eadxlgd->at(ii), pd_c->at(ii));
            dden = saddle::den(s, eadxlgd->at(ii), pd_c->at(ii));
            dnum2 = saddle::num2(s, eadxlgd->at(ii), pd_c->at(ii));
            k1 = dnum / dden;

            std::get<0>(k12) += n->at(ii) * k1;
            std::get<1>(k12) += n->at(ii) * ((dnum2 / dden) - pow(k1, 2));
        }

        return k12;
    }
    */

    std::tuple<double, double, double> K012(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
    {
        std::tuple<double, double, double> k012(0, 0, 0);

        double dnum, dden, dnum2, den, num;

        auto ii = n->begin();
        auto ll = eadxlgd->begin();
        auto pp = pd_c->begin();

        while (ii != n->end())
        {
            dden = pp->front() * exp(s * (ll->front() - ((s > 0) * ll->back())));
            dnum = 0;
            dnum2 = 0;

            for (auto jj = ll->begin() + 1, kk = pp->begin() + 1; jj != ll->end(); jj++, kk++)
            {
                den = (*kk) * exp(s * ((*jj) - ((s > 0) * ll->back())));
                num = den * (*jj);

                dden += den;
                dnum += num;
                dnum2 += num * (*jj);
            }

            std::get<0>(k012) += (*ii) * (log(dden) + ((s > 0) * s * ll->back()));
            std::get<1>(k012) += (*ii) * (dnum / dden);
            std::get<2>(k012) += (*ii) * (dnum2 / dden - (dnum * dnum) / (dden * dden));

            ii++;
            ll++;
            pp++;
        }

        return k012;
    }

    std::tuple<double, double>         K12(double  s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
    {
        std::tuple<double, double> k12(0, 0);

        double dnum, dden, dnum2, den, num;

        auto ii = n->begin();
        auto ll = eadxlgd->begin();
        auto pp = pd_c->begin();

        while (ii != n->end())
        {
            dden = pp->front() * exp(s * (ll->front() - ((s > 0) * ll->back())));
            dnum = 0;
            dnum2 = 0;

            for (auto jj = ll->begin() + 1, kk = pp->begin() + 1; jj != ll->end(); jj++, kk++)
            {
                den = (*kk) * exp(s * ((*jj) - ((s > 0) * ll->back())));
                num = den * (*jj);

                dden += den;
                dnum += num;
                dnum2 += num * (*jj);
            }

            std::get<0>(k12) += (*ii) * (dnum / dden);
            std::get<1>(k12) += (*ii) * (dnum2 / dden - (dnum * dnum) / (dden * dden));

            ii++;
            ll++;
            pp++;
        }

        return k12;
    }

}

}

