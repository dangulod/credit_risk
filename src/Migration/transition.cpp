#include "transition.h"

namespace CreditRisk
{

Transition::Transition(std::vector<std::string> states, arma::mat matrix) :
    m_states(states), m_matrix(matrix)
{
    for (size_t ii = 0; ii < matrix.n_rows; ii++)
    {
        if (fabs(arma::accu(matrix.row(ii)) - 1) > 1e6)
        {
            throw std::invalid_argument("The probability of one rating does not sum 1");
        }
    }
}

Transition Transition::from_ptree(pt::ptree & value)
{
    std::vector<std::string> states(value.get_child("states").size());
    size_t jj = 0;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("states"))
    {
        states.at(jj) = ii.second.get_value<std::string>();
        jj++;
    }

    arma::mat matrix(value.get_child("matrix").size(), value.get_child("matrix").size());
    size_t c_ii = 0;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("matrix"))
    {
        size_t c_jj = 0;
        BOOST_FOREACH(const pt::ptree::value_type & jj, ii.second.get_child(""))
        {
            matrix.at(c_ii, c_jj) = jj.second.get_value<double>();
            c_jj++;
        }
        c_ii++;
    }
    return Transition(states, matrix);
}

pt::ptree Transition::to_ptree()
{
    pt::ptree root;

    pt::ptree states;

    for (auto & ii: this->m_states)
    {
        pt::ptree w;
        w.put("", ii);

        states.push_back(std::make_pair("", w));
    }

    root.add_child("states", states);

    pt::ptree matrix_node;

    for (size_t && ii = 0; ii < this->m_matrix.n_rows; ii++)
    {
        pt::ptree row;

        for (size_t && jj = 0; jj < this->m_matrix.n_cols; jj++)
        {
            pt::ptree cell;
            cell.put_value(this->m_matrix(ii, jj));
            row.push_back(std::make_pair("", cell));
        }
        matrix_node.push_back(std::make_pair("", row));
    }
    root.add_child("matrix", matrix_node);

    return root;
}

double Transition::at(size_t ii, size_t jj)
{
    if ((ii >= 0) & (ii < this->n_states()) & (jj >= 0) & (jj < this->n_states()))
    {
        return this->m_matrix.at(ii, jj);
    }

    return 0;
}

size_t Transition::n_states()
{
    return this->m_states.size();
}

arma::vec Transition::states_prob(double pd)
{
    int d(this->m_matrix.n_cols - 1);
    arma::vec vt(d);

    int kk(0);

    while (pd > this->m_matrix(kk, d)) kk++;

    if (kk > 0)
    {
        double w1 = ( this->m_matrix(kk, d) - pd ) / ( this->m_matrix(kk, d) - this->m_matrix(kk - 1, d) ) ;
        double w2 = ( pd - this->m_matrix(kk - 1, d) ) / ( this->m_matrix(kk, d) - this->m_matrix(kk - 1, d) );

        for (int ii = 0; ii < d; ii++)
        {
            vt(ii) = w1 * this->m_matrix(kk - 1, ii) + w2 * this->m_matrix(kk, ii);
        }
    }
    else
    {
        for (int i = 0; i < d; i++)
        {
            vt(i) = this->m_matrix(0, i) * ( 1 + (this->m_matrix(0, d) - pd) / (1 - this->m_matrix(0, d)));
        }
    }

    return vt;
}

void Transition::to_csv(std::string file)
{
    std::ofstream output(file);

    if (output.is_open())
    {
        output << "Ratings";

        for (auto & ii : this->m_states)
        {
            output << "," << ii;
        }

        output << std::endl;

        for (size_t ii = 0; ii < this->m_matrix.n_rows; ii++)
        {
            output << this->m_states.at(ii);

            for (size_t jj = 0; jj < this->m_matrix.n_cols; jj++)
            {
                output << "," << this->m_matrix.at(ii, jj);
            }
            output << std::endl;
        }

        output.close();
    } else
    {
        throw std::invalid_argument("File can not be opened");
    }
}

Transition Transition::from_csv(std::string file)
{
    std::ifstream input(file);

    if (input.is_open())
    {
        size_t ss = CreditRisk::Utils::number_of_lines(file);

        arma::mat matrix(ss - 1, ss - 1);
        std::string buffer;

        std::vector<std::string> splitted;

        std::getline(input, buffer);
        boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

        std::vector<std::string> states(std::next(splitted.begin()), splitted.end());

        size_t ii = 0;
        while (std::getline(input, buffer))
        {
            boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

            if (splitted.size() == ss)
            {
                for (size_t jj = 0; jj < (ss - 1); jj++)
                {
                    matrix.at(ii, jj) = atof(splitted.at(jj + 1).c_str());
                }
                ii++;
            }
        }
        return Transition(states, matrix);
    } else
    {
        throw std::invalid_argument("File can not be opened");
    }
}

std::string Transition::state(double pd)
{
    if (pd < this->m_matrix.at(0, this->m_matrix.n_cols - 1)) return this->m_states.at(0);
    size_t ii = 1;

    while (pd > this->m_matrix.at(ii, this->m_matrix.n_cols - 1))
    {
        ii++;
    }

    return this->m_states.at(ii - 1);
}

}
