#include <utils.h>

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
    std::mt19937_64 generator;;

    double numerator = generator();
    double divisor = generator.max();
    double p = numerator / divisor;

    return qnorm(p);
}


double randn_s(unsigned long seed)
{
    std::mt19937_64 generator;;
    generator.seed(seed);

    double numerator = generator();
    double divisor = generator.max();
    double p = numerator / divisor;

    return qnorm(p);
}

arma::vec randn_v(size_t n, unsigned long seed)
{
    arma::vec ale(n);
    double numerator, divisor, p;

    std::mt19937_64 generator;;
    generator.seed(seed);

    for (size_t ii = 0; ii < n; ii++)
    {
        numerator = generator();
        divisor = generator.max();
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

    arma::vec p_states_c(arma::vec & p_states, double npd, double beta, double idio, double cwi)
    {
        arma::vec pp(p_states.size() + 2);

        double pb = p_c(npd, beta, idio, cwi);
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

    double num(double s, arma::vec l_states, arma::vec p_states)
    {
        return (s < 0) ?
                    arma::accu(p_states * l_states * exp(s * l_states)) :
                    arma::accu(p_states * l_states * exp(s * (l_states - l_states.back())));
    }

    double num2(double s, arma::vec l_states, arma::vec p_states)
    {
        return (s < 0) ?
                    arma::accu(p_states * l_states * exp(s * l_states)) :
                    arma::accu(p_states * pow(l_states, 2) * exp(s * (l_states - l_states.back())));
    }

    double den(double s, arma::vec l_states, arma::vec p_states)
    {
        return (s < 0) ?
                    arma::accu(p_states * exp(s * l_states)) :
                    arma::accu(p_states * exp(s * (l_states - l_states.back())));
    }

    double K(double s, unsigned long n, arma::vec l_states, arma::vec p_states)
    {
        return n * log(den(s, l_states, p_states)) + (s < 0 ? 0 : s * l_states.back());
    }

    double K1(double s, unsigned long n, arma::vec l_states, arma::vec p_states)
    {
        return n * num(s, l_states, p_states) / den(s, l_states, p_states);
    }

    double K2(double s, unsigned long n, arma::vec l_states, arma::vec p_states)
    {
        double dnum(num(s, l_states, p_states)), dden(den(s, l_states, p_states)),
                dnum2(num2(s, l_states, p_states));

        return n * (dnum2 / dden - (dnum * dnum) / (dden * dden));
    }

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

    std::tuple<double, double, double> K012(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
    {
        std::tuple<double, double, double> k012(0, 0, 0);

        double dnum, dden, k1;

        for (size_t ii = 0; ii < n.size(); ii++)
        {
            dnum = saddle::num(s, eadxlgd[ii], pd_c[ii]);
            dden = saddle::den(s, eadxlgd[ii], pd_c[ii]);
            k1 = dnum / dden;

            std::get<0>(k012) += n[ii] * (log(dden) + (s < 0 ? 0 : s *  eadxlgd[ii]));
            std::get<1>(k012) += n[ii] * k1;
            std::get<2>(k012) += n[ii] * (k1 * eadxlgd[ii] - pow(k1, 2));
        }

        return k012;
    }

    std::tuple<double, double>         K12(double  s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
    {
        std::tuple<double, double> k12(0, 0);

        double dnum, dden, k1;

        for (size_t ii = 0; ii < n.size(); ii++)
        {
            dnum = saddle::num(s, eadxlgd[ii], pd_c[ii]);
            dden = saddle::den(s, eadxlgd[ii], pd_c[ii]);
            k1 = dnum / dden;

            std::get<0>(k12) += n[ii] * k1;
            std::get<1>(k12) += n[ii] * (k1 * eadxlgd[ii] - pow(k1, 2));
        }

        return k12;
    }
}

}

