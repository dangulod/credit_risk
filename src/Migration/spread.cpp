#include "spread.h"

namespace CreditRisk
{

Spread::Spread(std::vector<std::string> states, std::vector<unsigned int> terms, arma::mat matrix) :
    m_states(states), m_terms(terms), m_matrix(matrix) {}

Spread Spread::from_ptree(pt::ptree & value)
{
    std::vector<std::string> states(value.get_child("states").size());
    size_t jj = 0;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("states"))
    {
        states.at(jj) = ii.second.get_value<std::string>();
        jj++;
    }

    std::vector<unsigned int> terms(value.get_child("terms").size());
    size_t hh = 0;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("terms"))
    {
        terms.at(hh) = ii.second.get_value<unsigned int>();
        hh++;
    }

    arma::mat matrix(value.get_child("matrix").size(), terms.size());
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
    return Spread(states, terms, matrix);
}

pt::ptree Spread::to_ptree()
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

    pt::ptree terms;

    for (auto & ii: this->m_terms)
    {
        pt::ptree w;
        w.put("", ii);

        terms.push_back(std::make_pair("", w));
    }

    root.add_child("terms", terms);

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

double Spread::at(size_t ii, size_t jj)
{
    if ((ii >= 0) & (ii < this->n_states()) & (jj >= 0) & (jj < this->n_terms()))
    {
        return this->m_matrix.at(ii, jj);
    }

    return 0;
}

size_t Spread::n_states()
{
    return this->m_states.size();
}

size_t Spread::n_terms()
{
    return this->m_terms.size();
}

void Spread::to_csv(std::string file)
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

Spread Spread::from_csv(std::string file)
{
    std::ifstream input(file);

    if (input.is_open())
    {
        size_t ss = CreditRisk::Utils::number_of_lines(file);

        std::string buffer;

        std::vector<std::string> splitted;

        std::getline(input, buffer);
        boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

        size_t ll = splitted.size();

        std::vector<unsigned int> terms(ll - 1);

        for (size_t ii = 0; ii < (ll - 1); ii++)
        {
            terms.at(ii) = atoi(splitted.at(ii + 1).c_str());
        }

        std::vector<std::string> states(ss - 1);
        arma::mat matrix(ss - 1, ll - 1);

        size_t ii = 0;
        while (std::getline(input, buffer))
        {
            boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });
            states.at(ii) = splitted.at(0);

            if (splitted.size() == ll)
            {
                for (size_t jj = 0; jj < (ll - 1); jj++)
                {
                    matrix.at(ii, jj) = atof(splitted.at(jj + 1).c_str());
                }
                ii++;
            }
        }
        return Spread(states, terms, matrix);
    } else
    {
        throw std::invalid_argument("File can not be opened");
    }
}

size_t Spread::state(std::string state)
{
    for (size_t ii = 0; ii < this->m_states.size(); ii++)
    {
        if (this->m_states.at(ii) == state)
        {
            return ii;
        }
    }
    throw std::invalid_argument("State not in the spread matrix");
}

double Spread::spread(std::string state, double term)
{
    if (term <= 1)
    {
        throw std::invalid_argument("term can not be greater than 1");
    }

    size_t row = this->state(state);

    if (this->m_terms.front() > term )
    {
        return this->m_matrix.at(row, this->m_terms.front());
    }


    if (this->m_terms.back() < term )
    {
        return this->m_matrix.at(row, this->m_matrix.n_cols - 1);
    }

    size_t column = 0;

    while (this->m_terms.at(column) < term)
    {
        column++;
    }

    double st1 = this->m_matrix.at(row, column - 1);
    double st2 = this->m_matrix.at(row, column);

    double t1 = this->m_terms.at(column - 1);
    double t2 = this->m_terms.at(column);

    return (st2 * (term - t1) + st1 *(t2 - term)) / (t2 - t1);
}

arma::vec Spread::get_spreads(double term, std::string state,  double rf, double max)
{
    size_t ss = this->state(state);
    arma::vec result(this->m_states.size() - ss - 1, arma::fill::zeros);
    double s0 = this->spread(state, term);
    double a, b, sf, desc;

    for (size_t ii = ss + 1; ii < this->m_states.size(); ii++)
    {
        sf = this->spread(this->m_states.at(ii), term);
        a = rf + s0;
        b = rf + sf;
        desc = (a * pow(1 + b, term) + b - a) / (b * pow(1 + b, term));
        result.at(ii - (ss + 1)) = std::min(1 - desc, max);
    }

    return result;
}

}

