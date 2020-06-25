#include "integrator.h"

namespace CreditRisk
{
namespace Integrator
{
double z_dict(int n, double z, arma::vec & x, int i)
{
	switch (i)
	{
	case 0:
		return sqrt(2 * n + 1) - 1.85575 * pow(2 * n + 1, -0.16667);
		break;
	case 1:
		return z - 1.14 * pow(n, 0.426) / z;
		break;
	case 2:
		return 1.86 * z - 0.86 * x(0);
		break;
	case 3:
		return 1.91 * z - 0.91 * x(1);
		break;
	default:
		return 2 * z - x(i - 2);
		break;
	}
}

PointsAndWeigths ghi(unsigned long n, double eps)
{
	if ((n % 2 == 0) | (n < 2)) throw std::invalid_argument("N must be odd and greater than 1");

	double dif = 100;

	double pi = boost::math::constants::pi<double>(), p1, p2, p3, pp, z1, z = 1;

	unsigned long m = static_cast<unsigned long>((n + 1) / 2);

	arma::vec x(n, arma::fill::zeros), w(n, arma::fill::zeros);

	for (unsigned int ii = 0; ii < m; ii++)
	{
		dif = 100;
		z = z_dict(n, z, x, ii);

		while (abs(dif) > eps)
		{
			p1 = 1 / pow(pi, 0.25);
			p2 = 0;

			for (unsigned int jj = 0; jj < n; jj++)
			{
				p3 = p2;
				p2 = p1;
				p1 = z * sqrt(2 / static_cast<double>(jj + 1)) * p2 - sqrt(static_cast<double>(jj) / static_cast<double>(jj + 1)) * p3;
			}

			pp = sqrt(2 * n) * p2;
			z1 = z;
			z = z1 - p1 / pp;
			dif = z - z1;
		}

		x(ii) = z;
		x(n - 1 - ii) = -z;
		w(ii) = 2 / pow(pp, 2);
		w(n - 1 - ii) = 2 / pow(pp, 2);
	}

	return { -x * sqrt(2), w / sqrt(pi) };
}

PointsAndWeigths gki(double start, unsigned long steps)
{
	arma::vec x(steps, arma::fill::zeros), w(steps, arma::fill::zeros);
	double step = 2 * abs(start / static_cast<double>(steps - 1));
    double norm = step / boost::math::constants::root_two_pi<double>();

	for (unsigned int ii = 0; ii < steps; ii++)
	{
		x[ii] = start + step * ii;
        w[ii] = exp(-0.5 * pow(x[ii], 2)) * norm;
	}

	return { x, w };
}
}
}
