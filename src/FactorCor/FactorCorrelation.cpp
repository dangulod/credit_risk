#include "factorCorrelation.h"

namespace CreditRisk
{
void isCor(arma::mat cor)
{
    if (cor.n_cols != cor.n_rows)
    {
        throw std::invalid_argument("Correlation matrix is not squared");
    }

    for (unsigned long i = 0,  l = cor.n_cols; i < l; i++)
    {
        for (unsigned long j = 0; j < l; j++)
        {
            if (i == j)
            {
                if ( fabs(cor(i, j) - 1.0 ) > EPSILON )
                {
                    throw std::invalid_argument("The diagonal of the correlation matrix must be 1");
                }
            }
            else
            {
                if ( fabs(cor(i, j) - cor(j, i)) > EPSILON )
                {
                    throw std::invalid_argument("Correlation matrix is not symetric");
                }
            }
        }
    }
}

CorMatrix::CorMatrix(arma::mat cor)
{
    isCor(cor);

    this->cor = cor;

    arma::mat U, V;
    arma::vec S;

    svd(U, S, V, cor);

    arma::mat A = zeros(size(cor));
    A.diag() = sqrt(S);

    this->vs = V * A;
}


CorMatrix CorMatrix::from_ptree(pt::ptree & value)
{
    arma::mat cor(value.get_child("cor").size(), value.get_child("cor").size());

    size_t c_ii = 0;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("cor"))
    {
        size_t c_jj = 0;
        BOOST_FOREACH(const pt::ptree::value_type & jj, ii.second.get_child(""))
        {
            cor.at(c_ii, c_jj) = jj.second.get_value<double>();
            c_jj++;
        }
        c_ii++;
    }
    return CorMatrix(cor);
}

pt::ptree CorMatrix::to_ptree()
{
    pt::ptree root;

    pt::ptree matrix_node;

    for (size_t && ii = 0; ii < this->cor.n_rows; ii++)
    {
        pt::ptree row;

        for (size_t && jj = 0; jj < this->cor.n_cols; jj++)
        {
            pt::ptree cell;
            cell.put_value(this->cor(ii, jj));
            row.push_back(std::make_pair("", cell));
        }
        matrix_node.push_back(std::make_pair("", row));
    }
    root.add_child("cor", matrix_node);

    return root;
}

void CorMatrix::to_csv(string file)
{
    std::ofstream output(file);

    if (output.is_open())
    {
        for (size_t ii = 0; ii < this->cor.n_rows; ii++)
        {
            for (size_t jj = 0; jj < this->cor.n_cols; jj++)
            {
                if (jj == (this->cor.n_cols - 1))
                {
                    output << std::setprecision(16) << this->cor.at(ii, jj);
                } else
                {
                    output << std::setprecision(16) << this->cor.at(ii, jj) << ",";
                }

            }
            output << std::endl;
        }

        output.close();
    } else
    {
        throw std::invalid_argument("File can not be opened");
    }
}

CorMatrix CorMatrix::from_csv(string file, size_t n_factors)
{
    std::ifstream input(file);

    if (input.is_open())
    {
        std::string buffer;
        std::vector<std::string> splitted;

        arma::mat cor(n_factors, n_factors);

        size_t ii = 0;

        while (std::getline(input, buffer))
        {
            boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

            if (splitted.size() == n_factors)
            {
                for (size_t jj = 0; jj < n_factors; jj++)
                {
                    cor.at(ii, jj) = atof(splitted.at(jj).c_str());
                }
                ii++;
            }
        }
        return CorMatrix(cor);
    } else
    {
        throw std::invalid_argument("File can not be opened");
    }
}

void CorMatrix::check_equation(CreditRisk::Equation & value)
{
    double R2 = arma::as_scalar(value.weights.t() * this->cor * value.weights);

    if ( R2 > 1 ) throw  std::invalid_argument("R2 squared greater than 1");
    if ( this->cor.n_cols != value.weights.size() ) throw std::invalid_argument("Invalid equation, size does not match");
}

}


