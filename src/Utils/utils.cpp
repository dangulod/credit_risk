#include <utils.h>

namespace CreditRisk
{
namespace Utils
{
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

double pnorm(double x)
{
    double result;
    double diff = x / M_SQRT2;
    result = boost::math::erfc(-diff) / 2;
    return result;
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

double quantile(arma::vec x, double q)
{
    if (q < 0 | q > 1) throw std::invalid_argument("Invalid percentile value");
    x = arma::sort(x);

    return x.at(static_cast<size_t>(q * x.size() - 1));
}
}

namespace saddle
{
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
}

}

