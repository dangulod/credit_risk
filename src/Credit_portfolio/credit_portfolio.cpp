#include "credit_portfolio.h"

std::mutex mu_p;

namespace CreditRisk
{

Credit_portfolio::Credit_portfolio(arma::mat cor):  n(0), T_EAD(0), T_EADxLGD(0), cf(cor) {}

void Credit_portfolio::operator+(CreditRisk::Portfolio & value)
{
    for (auto & ii: *this)
    {
        if (ii->name == value.name) std::invalid_argument("Name of portfolio duplicated");
    }

    for (auto & ii: value) this->cf.check_equation(ii.equ);
    this->n += value.size();

    for (auto & ii: value)
    {
        size_t pos = CreditRisk::Utils::get_position(this->rus, ii.ru);

        if (pos == this->rus.size())
            this->rus.push_back(ii.ru);

        this->rus_pos.push_back(pos);
    }

    this->T_EAD += value.getT_EAD();
    this->T_EADxLGD += value.getT_EADxLGD();

    this->push_back(std::make_unique<CreditRisk::Portfolio>(std::move(value)));
}

void Credit_portfolio::operator+(CreditRisk::Portfolio && value)
{
    for (auto & ii: *this)
    {
        if (ii->name == value.name) std::invalid_argument("Name of portfolio duplicated");
    }

    for (auto & ii: value) this->cf.check_equation(ii.equ);
    this->n += value.size();

    for (auto & ii: value)
    {
        size_t pos = CreditRisk::Utils::get_position(this->rus, ii.ru);

        if (pos == this->rus.size())
            this->rus.push_back(ii.ru);

        this->rus_pos.push_back(pos);
    }

    this->T_EAD += value.getT_EAD();
    this->T_EADxLGD += value.getT_EADxLGD();

    this->push_back(std::make_unique<CreditRisk::Portfolio>(std::move(value)));
}

void Credit_portfolio::operator+(CreditRisk::Fund & value)
{
    for (auto & ii: *this)
    {
        if (ii->name == value.name) std::invalid_argument("Name of portfolio duplicated");
    }

    for (auto & ii: value) this->cf.check_equation(ii.equ);
    this->n += value.size();

    for (auto & ii: value)
    {
        size_t pos = CreditRisk::Utils::get_position(this->rus, ii.ru);

        if (pos == this->rus.size())
            this->rus.push_back(ii.ru);

        this->rus_pos.push_back(pos);
    }

    this->T_EAD += value.getT_EAD();
    this->T_EADxLGD += value.getT_EADxLGD();

    this->push_back(std::make_unique<CreditRisk::Fund>(std::move(value)));
}

void Credit_portfolio::operator+(CreditRisk::Fund && value)
{
    for (auto & ii: *this)
    {
        if (ii->name == value.name) std::invalid_argument("Name of portfolio duplicated");
    }

    for (auto & ii: value) this->cf.check_equation(ii.equ);
    this->n += value.size();

    for (auto & ii: value)
    {
        size_t pos = CreditRisk::Utils::get_position(this->rus, ii.ru);

        if (pos == this->rus.size())
            this->rus.push_back(ii.ru);

        this->rus_pos.push_back(pos);
    }

    this->T_EAD += value.getT_EAD();
    this->T_EADxLGD += value.getT_EADxLGD();

    this->push_back(std::make_unique<CreditRisk::Fund>(std::move(value)));
}

pt::ptree Credit_portfolio::to_ptree()
{
    pt::ptree root;

    root.add_child("cor_factor", cf.to_ptree());

    pt::ptree port;
    pt::ptree fund;

    for (auto & ii: *this)
    {
        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            fund.push_back(std::make_pair("", dynamic_cast<CreditRisk::Fund*>(ii.get())->to_ptree()));
        } else {
            port.push_back(std::make_pair("", ii->to_ptree()));
        }
    }

    root.add_child("Portfolios", port);
    root.add_child("Funds", fund);
    return root;
}

Credit_portfolio Credit_portfolio::from_ptree(pt::ptree & value)
{
    Credit_portfolio p(CreditRisk::CorMatrix::from_ptree(value.get_child("cor_factor")).cor);

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("Portfolios"))
    {
        pt::ptree pt = ii.second;
        p + CreditRisk::Portfolio::from_ptree(pt);
    }

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("Funds"))
    {
        pt::ptree pt = ii.second;
        p + CreditRisk::Fund::from_ptree(pt);
    }

    return p;
}

Credit_portfolio Credit_portfolio::from_csv(string Portfolios, string Funds, string Elements, string CorMatrix, size_t n_factors)
{
    Credit_portfolio p(CreditRisk::CorMatrix::from_csv(CorMatrix, n_factors).cor);

    std::vector<std::shared_ptr<CreditRisk::Portfolio>> portfolios;

    std::ifstream input(Portfolios);
    string header, buffer;
    std::vector<std::string> splitted;

    if (input.is_open())
    {
        std::getline(input, header);

        if (header == "Portfolio" || header == "Portfolio\r")
        {
            while (std::getline(input, buffer))
            {
                boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

                if (splitted.size() == 5)
                {
                    portfolios.push_back(std::make_shared<CreditRisk::Portfolio>(
                                             CreditRisk::Portfolio(splitted[0],
                                             atof(splitted[1].c_str()),
                                             atof(splitted[2].c_str()),
                                             atof(splitted[3].c_str()),
                                             atof(splitted[4].c_str()))));
                } else if (splitted.size() == 1)
                {
                    portfolios.push_back(std::make_shared<CreditRisk::Portfolio>(CreditRisk::Portfolio(splitted[0])));
                }
            }
        } else
        {
            throw std::invalid_argument("Portfolio is not a valid file");
        }

        input.close();
    } else
    {
        throw std::invalid_argument("Portfolio file can not be opened");
    }

    if (Funds != "")
    {

        input.open(Funds);

        if (input.is_open())
        {
            std::string name;
            std::getline(input, header);

            if (header == "Fund,attachment,detachment" || header == "Fund,attachment,detachment\r")
            {
                while (std::getline(input, buffer))
                {
                    boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

                    if (splitted.size() == 7)
                    {
                        if (name != splitted.at(0))
                        {
                            name = splitted.at(0);
                            portfolios.push_back(std::make_shared<CreditRisk::Fund>(CreditRisk::Fund(splitted.at(0), 1, atof(splitted.at(1).c_str()), atof(splitted.at(2).c_str()))));
                        } else
                        {
                            for (auto & ii: portfolios)
                            {
                                if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
                                {
                                    if (ii->name == splitted.at(0))
                                    {
                                        dynamic_cast<CreditRisk::Fund*>(ii.get())->fundParam.push_back({1, atof(splitted.at(1).c_str()),
                                                                                            atof(splitted.at(2).c_str())});
                                    }
                                }
                            }
                        }

                    }
                }
            } else
            {
                throw std::invalid_argument("Fund is not a valid file");
            }

            input.close();
        } else
        {
            throw std::invalid_argument("Fund file can not be opened");
        }
    }
    input.open(Elements);

    if (input.is_open())
    {
        std::getline(input, header);

        while (std::getline(input, buffer))
        {
            boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

            if (splitted.size() > 1)
            {
                arma::vec wei(n_factors);

                for (size_t jj = 0; jj < n_factors; jj++)
                {
                    wei.at(jj) = atof(splitted.at(13 + jj).c_str());
                }

                CreditRisk::Equation eq(atoi(splitted.at(12).c_str()), wei);

                CreditRisk::Element ele(atoi(splitted.at(1).c_str()),
                            atoi(splitted.at(2).c_str()),
                            atof(splitted.at(3).c_str()),
                            atof(splitted.at(4).c_str()),
                            atof(splitted.at(5).c_str()),
                            atof(splitted.at(6).c_str()),
                            atof(splitted.at(7).c_str()),
                            atof(splitted.at(8).c_str()),
                            atof(splitted.at(9).c_str()),
                            atof(splitted.at(10).c_str()),
                            (splitted.at(11) == "wholesale") ? CreditRisk::Element::Element::Treatment::Wholesale : CreditRisk::Element::Element::Treatment::Retail,
                            std::move(eq));

                for (auto & ii: portfolios)
                {
                    if (ii->name == splitted.at(0)) *ii.get() + ele;
                }
            }
        }
        input.close();
    } else
    {
        throw std::invalid_argument("Element file can not be opened");
    }

    for (auto & ii: portfolios)
    {
        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            p + *dynamic_cast<CreditRisk::Fund*>(ii.get());
        } else {
            p + *ii;
        }
    }

    return p;
}

void Credit_portfolio::setT_EADxLGD()
{
    this->T_EADxLGD = 0;

    for (auto & ii: (*this))
    {
        this->T_EADxLGD += ii->getT_EADxLGD();
    }
}

void Credit_portfolio::setT_EAD()
{
    this->T_EAD = 0;

    for (auto & ii: (*this))
    {
        this->T_EAD += ii->getT_EAD();
    }
}

void Credit_portfolio::setRUs()
{
    std::vector<unsigned long> rus;
    std::vector<size_t> ru_pos;

    for (auto & ii: (*this))
    {
        for (auto &jj: *ii)
        {
            size_t pos = CreditRisk::Utils::get_position(rus, jj.ru);
            if (pos == rus.size())
                rus.push_back(jj.ru);

            ru_pos.push_back(pos); // Corregir la posicionl
        }
    }

    this->rus = rus;
    this->rus_pos = ru_pos;
}

double Credit_portfolio::getPE()
{
    double pe = 0;

    for (auto & ii: (*this))
    {
        pe += ii->get_PE();
    }

    return pe;
}

size_t Credit_portfolio::n_factors()
{
    return this->cf.cor.n_cols;
}

size_t Credit_portfolio::getN()
{
    size_t ll(0);

    for (auto & ii: (*this))
    {
        ll += ii->size();
    }

    return ll;
}

arma::vec Credit_portfolio::get_std_EADxLGDs()
{
    size_t jj(0);
    arma::vec vec(this->getN());

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk]._le / (this->T_EADxLGD * (*ii)[kk].n);
            jj++;
        }
    }

    return vec;
}

arma::vec Credit_portfolio::get_EADxLGDs()
{
    size_t jj(0);
    arma::vec vec(this->getN());

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk]._le / (*ii)[kk].n;
            jj++;
        }
    }

    return vec;
}

std::vector<double> Credit_portfolio::get_portfolios_EADs()
{
    std::vector<double> vec(this->size());

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        vec[ii] = (*this)[ii]->getT_EAD();
    }

    return vec;
}

std::vector<Scenario_data> Credit_portfolio::get_scenario_data(unsigned long n)
{
    std::vector<Scenario_data> data(n);
    size_t jj = 0; // number of funds

    for (auto & ii: *this)
    {
        if (dynamic_cast<Fund*>(ii.get()) != nullptr)
        {
            jj++;
        }
    }

    Scenario_data scenario(jj);
    jj = 0;

    for (auto & ii: *this)
    {
        if (dynamic_cast<Fund*>(ii.get()) != nullptr)
        {
            Fund_data fund(dynamic_cast<Fund*>(ii.get())->fundParam.size());

            for (size_t kk = 0; kk < dynamic_cast<Fund*>(ii.get())->fundParam.size(); kk++)
            {
                fund.at(kk) = Tranche_data(dynamic_cast<Fund*>(ii.get())->fundParam.at(kk).purchase);
            }
            scenario.at(jj) = fund;
            jj++;
        }
    }

    for (size_t ii = 0; ii < data.size(); ii++)
    {
        data.at(ii) = scenario;
    }

    return data;
}

arma::vec Credit_portfolio::get_Ns()
{
    size_t jj(0);
    arma::vec vec(this->getN());

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].n;
            jj++;
        }
    }

    return vec;
}

arma::vec Credit_portfolio::v_rand(unsigned long seed)
{
    return this->cf.vs * CreditRisk::Utils::randn_v(this->n_factors(), seed);
}

void Credit_portfolio::pv_rand(arma::mat *r, size_t n, unsigned long seed, size_t id, size_t p)
{
    while (id < n)
    {
        r->row(id) = v_rand(seed + id).t();
        id += p;
    }
}

arma::mat Credit_portfolio::m_rand(size_t n, unsigned long seed, size_t p)
{
    arma::mat ale   = arma::zeros(n, this->n_factors());

    vector<std::thread> threads(p);

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::pv_rand, this, &ale, n, seed, ii, p);
    }

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii].join();
    }

    return ale;
}

double Credit_portfolio::d_rand(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->n_factors()))
    {
        return this->v_rand(seed + row).at(column);
    }
    return 0;
}

void Credit_portfolio::pmIdio(arma::mat *l, size_t n, size_t id, size_t p)
{
    while (id < n)
    {
        l->row(id) = this->v_idio(id).t();
        id += p;
    }
}

arma::vec Credit_portfolio::v_idio(size_t id)
{
    size_t jj(0);
    arma::vec l(this->getN());

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            l[jj] = CreditRisk::Utils::randn_s((*ii)[kk].equ.idio_seed + id);
            jj++;
        }
    }

    return l;
}

arma::mat Credit_portfolio::getIdio(size_t n, size_t p)
{
    arma::mat m(n, this->getN());

    vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::pmIdio, this, &m, n, ii, p);
    }

    for (auto & ii: threads)
    {
        ii.join();
    }

    return m;
}

Element & Credit_portfolio::get_element(size_t column)
{
    if (column < 0 || column > this->getN())
    {
        throw std::invalid_argument("Invalid element number");
    }

    size_t jj = 0;

    for (auto &ii: *this)
    {
        if (column < ii->size() + jj)
        {
            return ii->at(column - jj);
        } else
        {
            jj += ii->size();
        }
    }
    throw std::invalid_argument("Invalid element number");
}

long Credit_portfolio::which_fund(size_t column)
{
    if (column < 0 || column > this->getN())
    {
        throw std::invalid_argument("Invalid element number");
    }

    size_t jj = 0;
    size_t kk = 0;

    for (unsigned long ii = 0; ii < this->size(); ii++)
    {

        if (column < this->at(ii)->size() + jj)
        {
            if (dynamic_cast<Fund*>(this->at(ii).get()) != nullptr)
            {
                return kk;
            } else
            {
                return -1;
            }
        } else
        {
            jj += this->at(ii)->size();
            if (dynamic_cast<Fund*>(this->at(ii).get()) != nullptr)
            {
                kk++;
            }
        }
    }
    throw std::invalid_argument("Invalid element number");
}

size_t Credit_portfolio::which_portfolio(size_t column)
{
    if (column < 0 || column > this->getN())
    {
        throw std::invalid_argument("Invalid element number");
    }

    size_t jj = 0;
    size_t kk = 0;

    for (unsigned long ii = 0; ii < this->size(); ii++)
    {

        if (column < this->at(ii)->size() + jj)
        {
            if (dynamic_cast<Fund*>(this->at(ii).get()) != nullptr)
            {
                return kk;
            } else
            {
                return -1;
            }
        } else
        {
            jj += this->at(ii)->size();
            kk++;
        }
    }
    throw std::invalid_argument("Invalid element number");
}


double Credit_portfolio::d_Idio(size_t row, size_t column, size_t n)
{
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->getN()))
    {
        return CreditRisk::Utils::randn_s(this->get_element(column).equ.idio_seed + row);
    }
    return 0;
}

arma::vec Credit_portfolio::getCWI(arma::vec f, unsigned long idio_id)
{
    size_t jj(0);
    arma::vec l(this->getN());

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            l[jj] =(*ii)[kk].equ.CWI(f, idio_id);
            jj++;
        }
    }

    return l;
}

arma::vec Credit_portfolio::getCWI(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = this->v_rand(seed);
    return getCWI(f, idio_id);
}


arma::mat Credit_portfolio::getCWIs(size_t n, unsigned long seed, size_t p)
{
    arma::mat l(n, this->getN());
    vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++) threads[ii] = std::thread(&Credit_portfolio::pmCWI, this, &l, n, seed, ii, p);
    for (auto & ii: threads) ii.join();

    return l;
}

double Credit_portfolio::d_CWI(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->getN()))
    {
        arma::vec f = this->v_rand(seed + row);
        return this->get_element(column).equ.CWI(f, row);
    }

    return 0;
}

void Credit_portfolio::pmCWI(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p)
{
   while (id < n)
    {
        l->row(id) = getCWI(seed + id, id).t();
        id += p;
    }
}

arma::vec Credit_portfolio::marginal(arma::vec f, unsigned long idio_id)
{
    size_t jj(0);
    arma::vec l(this->getN(), arma::fill::zeros);

    for (auto & ii: *this)
    {

        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            CreditRisk::Fund * spv = dynamic_cast<CreditRisk::Fund*>(ii.get());
            arma::vec cwi = spv->get_cwi(f, idio_id);
            arma::vec v_t = spv->get_t(cwi);

            arma::vec t_at = spv->get_T_at(cwi, v_t);
            arma::vec t_de = spv->get_T_de(cwi, v_t);

            for (size_t hh = 0; hh < spv->fundParam.size(); hh++)
            {
                if (t_at[hh] < 1)
                {
                    double fita = spv->__fit_T_loss_t(t_at[hh], spv->fundParam[hh].at, cwi, v_t);
                    double fitd = spv->__fit_T_loss_t(t_de[hh], spv->fundParam[hh].de, cwi, v_t);

                    for (size_t kk = 0; kk < ii->size(); kk++)
                    {
                        switch ((*ii)[kk].mr)
                        {
                        case CreditRisk::Element::Element::Treatment::Retail:
                        {
                            if (t_de[hh] - t_at[hh] > 1e-9)
                            {
                                double l_de = (*ii)[kk].loss(t_de[hh], cwi[kk]);
                                double l_at = (*ii)[kk].loss(t_at[hh], cwi[kk]);
                                l[jj] = (l_de - l_at) * spv->fundParam[hh].purchase;
                            }
                            break;
                        }
                        case CreditRisk::Element::Element::Treatment::Wholesale:
                        {
                            if ((*ii)[kk]._npd > cwi[kk])
                            {
                                l[jj] = ((t_at[hh] < v_t[kk]) & (v_t[kk] < t_de[hh])) * (*ii)[kk]._le +
                                        (fabs(v_t[kk] - t_at[hh]) < 1e-9) * fita -
                                        (fabs(v_t[kk] - t_de[hh]) < 1e-9) * fitd;
                                l.at(jj) *= spv->fundParam[hh].purchase;
                            }
                            break;
                        }
                        }
                        jj++;
                    }
                } else {
                    jj += ii->size();
                }
            }
        } else {
            for (size_t kk = 0; kk < ii->size(); kk++)
            {
                l[jj] =(*ii)[kk].loss(f, idio_id);
                jj++;
            }
        }
    }

    return l;
}

arma::vec Credit_portfolio::marginal(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = this->v_rand(seed);
    return marginal(f, idio_id);
}

void Credit_portfolio::pmloss(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p)
{
   while (id < n)
    {
        l->row(id) = marginal(seed + id, id).t();
        id += p;
    }
}


double Credit_portfolio::smargin_loss(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario)
{
    // Implementar
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->getN()))
    {
        Element * element = &this->get_element(column);

        long ff = this->which_fund(column);

        if (ff < 0)
        {
            return element->loss(this->d_CWI(row, column, n, seed));
        } else
        {
            size_t zz = this->which_portfolio(column);

            Fund_data * fund = &scenario.at(ff);

            arma::vec f = this->v_rand(seed + row);
            double cwi = this->d_CWI(row, column, n, seed);
            double l = 0;

            dynamic_cast<Fund*>(this->at(zz).get())->check_fund_data(*fund, f, row);

            switch (element->mr)
            {
            case CreditRisk::Element::Element::Treatment::Retail:
            {
                for (auto & ii: *fund)
                {
                    if (ii.t_de - ii.t_at > 1e-9)
                    {
                        double l_de = element->loss(ii.t_de, cwi);
                        double l_at = element->loss(ii.t_at, cwi);
                        l += (l_de - l_at);
                        l *= ii.p;
                    }
                }
                return l;
            }
            case CreditRisk::Element::Element::Treatment::Wholesale:
            {
                double t = this->get_element(column).getT(cwi);

                if (element->_npd < cwi)
                {
                    for (auto & ii: *fund)
                    {
                        l += ((ii.t_at < t) & (t < ii.t_de)) * element->_le +
                                (fabs(t - ii.t_at) < 1e-9) * ii.l_at -
                                (fabs(t - ii.t_de) < 1e-9) * ii.l_de;
                        l *= ii.p;
                    }
                }

                return l;
            }
            }
        }
    }

    return 0;
}

double Credit_portfolio::smargin_loss_without_secur(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->getN()))
    {
        return this->get_element(column).loss(this->d_CWI(row, column, n, seed));
    }

    return 0;
}

arma::mat Credit_portfolio::margin_loss(size_t n, unsigned long seed, size_t p)
{
    arma::mat l(n, this->getN());

    vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::pmloss, this, &l, n, seed, ii, p);
    }

    for (auto & ii: threads)
    {
        ii.join();
    }

    return l;
}

arma::vec Credit_portfolio::marginal_without_secur(arma::vec f, unsigned long idio_id)
{
    size_t jj(0);
    arma::vec l(this->getN(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            l[jj] =(*ii)[kk].loss(f, idio_id);
            jj++;
        }
    }

    return l;
}

arma::vec Credit_portfolio::marginal_without_secur(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = this->v_rand(seed);
    return marginal_without_secur(f, idio_id);
}


void Credit_portfolio::pmloss_without_secur(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p)
{
   while (id < n)
    {
        l->row(id) = this->marginal_without_secur(seed + id, id).t();
        id += p;
    }
}

arma::mat Credit_portfolio::margin_loss_without_secur(size_t n, unsigned long seed, size_t p)
{
    arma::mat l(n, this->getN());

    vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::pmloss_without_secur, this, &l, n, seed, ii, p);
    }

    for (auto & ii: threads)
    {
        ii.join();
    }

    return l;
}

double Credit_portfolio::sLoss_ru(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario)
{
    // Implementar
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->rus.size()))
    {
        double loss = 0;
        size_t cc = 0;
        arma::vec f = this->v_rand(seed + row);

        for (auto & ii: *this)
        {
            for (auto & jj: *ii)
            {
                if (jj.ru == this->rus.at(column))
                {
                    long kk = this->which_fund(cc);
                    size_t zz = this->which_portfolio(cc);
                    if (kk < 0)
                    {
                        loss += jj.loss(f, row);
                    } else
                    {
                        Fund_data * fund = &scenario.at(kk);
                        arma::vec f = this->v_rand(seed + row);

                        double cwi = this->d_CWI(row, cc, n, seed);
                        double l = 0;

                        dynamic_cast<Fund*>(this->at(zz).get())->check_fund_data(*fund, f, row);

                        switch (jj.mr)
                        {
                        case CreditRisk::Element::Element::Treatment::Retail:
                        {
                            for (auto & ii: *fund)
                            {
                                if (ii.t_de - ii.t_at > 1e-9)
                                {
                                    double l_de = jj.loss(ii.t_de, cwi);
                                    double l_at = jj.loss(ii.t_at, cwi);
                                    l += (l_de - l_at);
                                    l *= ii.p;
                                }
                            }
                        }
                        case CreditRisk::Element::Element::Treatment::Wholesale:
                        {
                            double t = this->get_element(column).getT(cwi);

                            if (jj._npd < cwi)
                            {
                                for (auto & ii: *fund)
                                {
                                    l += ((ii.t_at < t) & (t < ii.t_de)) * jj._le +
                                            (fabs(t - ii.t_at) < 1e-9) * ii.l_at -
                                            (fabs(t - ii.t_de) < 1e-9) * ii.l_de;
                                    l *= ii.p;
                                }
                            }
                        }
                        }
                        loss += l;
                    }
                }
                cc++;
            }
        }
        
        return loss;
    }

    return 0;
}

double Credit_portfolio::sLoss_ru_without_secur(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->rus.size()))
    {
        double loss = 0;

        arma::vec f = this->v_rand(seed + row);

        for (auto & ii: *this)
        {
            for (auto & jj: *ii)
            {
                if (jj.ru == this->rus.at(column))
                {
                    loss += jj.loss(f, row);
                }
            }
        }

        return loss;
    }

    return 0;
}

arma::vec Credit_portfolio::sLoss_ru(arma::vec  f, unsigned long idio_id)
{
    size_t jj(0);
    arma::vec l(this->rus.size(), arma::fill::zeros);

    for (auto & ii: *this)
    {

        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            CreditRisk::Fund * spv = dynamic_cast<CreditRisk::Fund*>(ii.get());
            arma::vec cwi = spv->get_cwi(f, idio_id);
            arma::vec v_t = spv->get_t(cwi);

            arma::vec t_at = spv->get_T_at(cwi, v_t);
            arma::vec t_de = spv->get_T_de(cwi, v_t);

            for (size_t hh = 0; hh < spv->fundParam.size(); hh++)
            {
                if (t_at[hh] < 1)
                {
                    double fita = spv->__fit_T_loss_t(t_at[hh], spv->fundParam[hh].at, cwi, v_t);
                    double fitd = spv->__fit_T_loss_t(t_de[hh], spv->fundParam[hh].de, cwi, v_t);

                    for (size_t kk = 0; kk < ii->size(); kk++)
                    {
                        switch ((*ii)[kk].mr)
                        {
                        case CreditRisk::Element::Element::Treatment::Retail:
                        {
                            if (t_de[hh] - t_at[hh] > 1e-9)
                            {
                                double l_de = (*ii)[kk].loss(t_de[hh], cwi[kk]);
                                double l_at = (*ii)[kk].loss(t_at[hh], cwi[kk]);
                                l[this->rus_pos[jj]] += (l_de - l_at) * spv->fundParam[hh].purchase;
                            }
                            break;
                        }
                        case CreditRisk::Element::Element::Treatment::Wholesale:
                        {
                            if ((*ii)[kk]._npd > cwi[kk])
                            {
                                l[this->rus_pos[jj]] += (((t_at[hh] < v_t[kk]) & (v_t[kk] < t_de[hh])) * (*ii)[kk]._le +
                                        (fabs(v_t[kk] - t_at[hh]) < 1e-9) * fita -
                                        (fabs(v_t[kk] - t_de[hh]) < 1e-9) * fitd) * spv->fundParam[hh].purchase;
                            }
                            break;
                        }
                        }
                        jj++;
                    }
                } else {
                    jj += ii->size();
                }
                }
        } else {
            for (size_t kk = 0; kk < ii->size(); kk++)
            {
                l[this->rus_pos[jj]] += (*ii)[kk].loss(f, idio_id);
                jj++;
            }
        }
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_ru(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = v_rand(seed);
    return sLoss_ru(f, idio_id);
}

void Credit_portfolio::ploss_ru(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_ru(seed + id, id).t();
        id += p;
    }
}

arma::mat Credit_portfolio::loss_ru(unsigned long n, unsigned long seed, unsigned long p)
{
    arma::mat l(n, this->rus.size());
    std::vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::ploss_ru, this, &l, n, seed, ii, p);
    }

    for (auto &ii: threads)
    {
        ii.join();
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_ru_without_secur(arma::vec  f, unsigned long idio_id)
{
    size_t jj(0);
    arma::vec l(this->rus.size(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            l[this->rus_pos[jj]] += (*ii)[kk].loss(f, idio_id);
            jj++;
        }
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_ru_without_secur(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = v_rand(seed);
    return sLoss_ru_without_secur(f, idio_id);
}

void Credit_portfolio::ploss_ru_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_ru_without_secur(seed + id, id).t();
        id += p;
    }
}

arma::mat Credit_portfolio::loss_ru_without_secur(unsigned long n, unsigned long seed, unsigned long p)
{
    arma::mat l(n, this->rus.size());
    std::vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::ploss_ru_without_secur, this, &l, n, seed, ii, p);
    }

    for (auto &ii: threads)
    {
        ii.join();
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio(arma::vec  f, unsigned long idio_id)
{
    size_t hh(0);
    arma::vec l(this->size(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            l[hh] = dynamic_cast<CreditRisk::Fund*>(ii.get())->loss_sec(f, idio_id);
        } else {
            l[hh] = ii->loss(f, idio_id);
        }
        hh++;
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = v_rand(seed);
    return sLoss_portfolio(f, idio_id);
}

void Credit_portfolio::ploss_portfolio(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_portfolio(seed + id, id).t();
        id += p;
    }
}

double Credit_portfolio::sLoss_portfolio(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->size()))
    {
        arma::vec f = this->v_rand(seed + row);
        if (dynamic_cast<Fund*>(this->at(column).get()) != nullptr)
        {
            return dynamic_cast<Fund*>(this->at(column).get())->loss_sec(f, row);
        } else
        {
            return this->at(column)->loss(f, row);
        }
    }

    return 0;
}

double Credit_portfolio::sLoss_portfolio_without_secur(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) & (column < this->size()))
    {
        arma::vec f = this->v_rand(seed + row);
        return this->at(column)->loss(f, row);
    }

    return 0;
}

arma::mat Credit_portfolio::loss_portfolio(unsigned long n, unsigned long seed, unsigned long p)
{
    arma::mat l(n, this->size());
    std::vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::ploss_portfolio, this, &l, n, seed, ii, p);
    }

    for (auto &ii: threads)
    {
        ii.join();
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio_without_secur(arma::vec  f, unsigned long idio_id)
{
    size_t hh(0);
    arma::vec l(this->size(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        l[hh] = ii->loss(f, idio_id);
        hh++;
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio_without_secur(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = v_rand(seed);
    return sLoss_portfolio_without_secur(f, idio_id);
}

void Credit_portfolio::ploss_portfolio_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_portfolio_without_secur(seed + id, id).t();
        id += p;
    }
}

arma::mat Credit_portfolio::loss_portfolio_without_secur(unsigned long n, unsigned long seed, unsigned long p)
{
    arma::mat l(n, this->size());
    std::vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::ploss_portfolio_without_secur, this, &l, n, seed, ii, p);
    }

    for (auto &ii: threads)
    {
        ii.join();
    }

    return l;
}

double Credit_portfolio::sLoss(arma::vec f, unsigned long idio_id)
{
    double loss(0);

    for (auto & ii: *this)
    {
        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            loss += dynamic_cast<CreditRisk::Fund*>(ii.get())->loss_sec(f, idio_id);
        } else {
            loss += ii->loss(f, idio_id);
        }

    }

    return loss;
}

double Credit_portfolio::sLoss(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = v_rand(seed);
    return sLoss(f, idio_id);
}

void Credit_portfolio::ploss(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p)
{
    while (id < n)
    {
        l->row(id) = sLoss(seed + id, id);
        id += p;
    }
}

arma::vec Credit_portfolio::loss(unsigned long n, unsigned long seed, unsigned long p)
{
    arma::vec l(n);
    vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::ploss, this, &l, n, seed, ii, p);
    }

    for (auto & ii: threads)
    {
        ii.join();
    }

    return l;
}

double Credit_portfolio::sLoss_without_secur(arma::vec f, unsigned long idio_id)
{
    double loss(0);

    for (auto & ii: *this)
    {
        loss += ii->loss(f, idio_id);
    }

    return loss;
}

double Credit_portfolio::sLoss_without_secur(unsigned long seed, unsigned long idio_id)
{
    arma::vec f = v_rand(seed);
    return sLoss_without_secur(f, idio_id);
}

void Credit_portfolio::ploss_without_secur(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p)
{
    while (id < n)
    {
        l->row(id) = sLoss_without_secur(seed + id, id);
        id += p;
    }
}

arma::vec Credit_portfolio::loss_without_secur(unsigned long n, unsigned long seed, unsigned long p)
{
    arma::vec l(n);
    vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::ploss_without_secur, this, &l, n, seed, ii, p);
    }

    for (auto & ii: threads)
    {
        ii.join();
    }

    return l;
}

arma::vec Credit_portfolio::pd_c(double scenario)
{
    arma::vec vec(this->getN());
    size_t jj = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].pd_c(scenario);
            jj++;
        };
    }

    return vec;
}

arma::vec Credit_portfolio::pd_c(double t, double scenario)
{
    arma::vec vec(this->getN());
    size_t jj = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].pd_c(t, scenario);
            jj++;
        };
    }

    return vec;
}

arma::vec Credit_portfolio::pd_c(arma::vec t, double scenario)
{
    arma::vec vec(this->getN());
    size_t jj = 0;
    size_t hh = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].pd_c(t[hh], scenario);
            jj++;
        };
        hh++;
    }

    return vec;
}

arma::vec Credit_portfolio::pd_c(arma::vec scenarios)
{
    arma::vec vec(this->getN());
    size_t jj = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].pd_c((*ii)[kk].equ.systematic(scenarios));
            jj++;
        };
    }

    return vec;
}

arma::vec Credit_portfolio::pd_c(double t, arma::vec scenarios)
{
    arma::vec vec(this->getN());
    size_t jj = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].pd_c(t, (*ii)[kk].equ.systematic(scenarios));
            jj++;
        };
    }

    return vec;
}

arma::vec Credit_portfolio::pd_c(arma::vec t, arma::vec scenarios)
{
    arma::vec vec(this->getN());
    size_t jj = 0;
    size_t hh = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].pd_c(t[hh], (*ii)[kk].equ.systematic(scenarios));
            jj++;
        };
        hh++;
    }

    return vec;
}

void Credit_portfolio::pd_c_fill(arma::mat * pd_c, size_t * ii, CreditRisk::Integrator::PointsAndWeigths * points)
{
    size_t jj;
    while (*ii < pd_c->n_cols)
    {
        mu_p.lock();
        jj = *ii;
        *ii = *ii + 1;
        mu_p.unlock();
        if (jj < pd_c->n_cols)
        {
            pd_c->col(jj) = this->pd_c(points->points(jj));
        }
    }
}

arma::mat Credit_portfolio::pd_c(CreditRisk::Integrator::PointsAndWeigths points, unsigned long p)
{
    arma::mat pd_c(this->getN(), points.points.size(), arma::fill::zeros);

    std::vector<std::thread> threads(p);

    size_t jj = 0;
    // this->pd_c_fill(&pd_c, &jj, &points);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::pd_c_fill, this, &pd_c, &jj, &points);
    }

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii].join();
    }

    return pd_c;
}

/*
 FUNCTIONS
*/

arma::vec Credit_portfolio::get_t_secur(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, arma::vec k1s, double scenario)
{
    arma::vec ts(this->size(), arma::fill::ones);
    size_t jj = 0;
    size_t kk = 0;

    for (auto &ii: (*this))
    {
        if (dynamic_cast<Fund*>(ii.get()) != nullptr)
        {
            ts.at(kk) = dynamic_cast<Fund*>(ii.get())->get_T_saddle(s, n, eadxlgd, pd_c, k1s.at(kk), scenario, jj);
            jj += ii->size();
            kk++;
        } else
        {
            jj += ii->size();
            kk++;
        }
    }
    return ts;
}


double Credit_portfolio::K (double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    double k = 0;

    for (size_t ii = 0; ii < n.size(); ii++)
    {
        k += saddle::K(s, n[ii], eadxlgd[ii], pd_c[ii]);
    }

    return k;
}

double Credit_portfolio::K1(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    double k1 = 0;

    for (size_t ii = 0; ii < n.size(); ii++)
    {
        k1 += saddle::K1(s, n[ii], eadxlgd[ii], pd_c[ii]);
    }

    return k1;
}

double Credit_portfolio::K1_secur(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    double k1 = 0;
    size_t jj = 0;

    for (auto &ii: *(this))
    {
        double k1p = 0;
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            k1p += saddle::K1(s, n[jj], eadxlgd[jj], pd_c[jj]);
            jj++;
        }

        Fund * fund = dynamic_cast<Fund *>(ii.get());

        if (fund != nullptr)
        {
            double m_loss = 0;

            for (size_t hh = 0; hh < fund->fundParam.size(); hh++)
            {
                m_loss += fmax(fmin(k1p - fund->fundParam.at(hh).at, fund->fundParam.at(hh).de - fund->fundParam.at(hh).at), 0);
            }
            k1p = m_loss;
        }
        k1 += k1p;
    }

    return k1;
}

arma::vec Credit_portfolio::K1_secur_vec(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    arma::vec k1(this->size(), arma::fill::zeros);
    size_t jj = 0;
    size_t pp = 0;

    for (auto &ii: *(this))
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            k1.at(pp) += saddle::K1(s, n[jj], eadxlgd[jj], pd_c[jj]);
            jj++;
        }

        Fund * fund = dynamic_cast<Fund *>(ii.get());

        if (fund != nullptr)
        {
            double m_loss = 0;

            for (size_t hh = 0; hh < fund->fundParam.size(); hh++)
            {
                m_loss += fmax(fmin(k1.at(pp), fund->fundParam.at(hh).de) - fund->fundParam.at(hh).at, 0) * fund->fundParam.at(hh).purchase;
            }

            k1.at(pp) = m_loss;
        }
        pp++;
    }

    return k1;
}

double Credit_portfolio::K2(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    double k2 = 0;

    for (size_t ii = 0; ii < n.size(); ii++)
    {
        k2 += saddle::K2(s, n[ii], eadxlgd[ii], pd_c[ii]);
    }

    return k2;
}

std::tuple<double, double, double> Credit_portfolio::K012(double s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
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

std::tuple<double, double>         Credit_portfolio::K12(double  s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
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


std::tuple<double, double>         Credit_portfolio::K12_secur(double  s, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    std::tuple<double, double> k12(0, 0);

    double dnum, dden, k1;
    size_t jj = 0;

    for (auto &ii: *(this))
    {
        double k1p = 0;
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            dnum = saddle::num(s, eadxlgd.at(jj), pd_c.at(jj));
            dden = saddle::den(s, eadxlgd.at(jj), pd_c.at(jj));
            k1 = n.at(jj) * dnum / dden;

            k1p = n.at(jj) * k1;
            std::get<1>(k12) += n.at(jj) * (k1p * eadxlgd.at(jj) - pow(k1, 2));
            jj++;
        }

        Fund * fund = dynamic_cast<Fund *>(ii.get());

        if (fund != nullptr)
        {
            double m_loss = 0;

            for (size_t hh = 0; hh < fund->fundParam.size(); hh++)
            {
                m_loss += fmax(fmin(k1p, fund->fundParam.at(hh).de) - fund->fundParam.at(hh).at, 0) * fund->fundParam.at(hh).purchase;
            }

            k1p = m_loss;
        }

        std::get<0>(k12) += k1p;
    }

    return k12;
}

double Credit_portfolio::fitSaddle_n(double s, double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    return this->K1(s, n, eadxlgd, pd_c) - loss;
}

double Credit_portfolio::getSaddle(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0, double a, double b, double tol)
{
    std::tuple<double, double> sfx = getSaddleNewton(loss, n, eadxlgd, pd_c, s0, tol);

    if (abs(std::get<1>(sfx)) < tol) return std::get<0>(sfx);

    std::get<0>(sfx) = getSaddleBrent(loss, n, eadxlgd, pd_c, a, b);
    std::get<1>(sfx) = K1(std::get<0>(sfx), n, eadxlgd, pd_c) - loss;

    if (abs(std::get<1>(sfx)) > tol)
    {
        printf("loss: %.20f, a: %.20f, b: %.20f, s: %.20f, abs(fx): %.20f\n", loss, a, b, std::get<0>(sfx), abs(std::get<1>(sfx)));
        throw std::domain_error("A solution could not be found");
    }

    return std::get<0>(sfx);
}

double Credit_portfolio::getSaddleBrent(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double a, double b, double xtol, double rtol)
{
    return CreditRisk::Optim::root_Brentq(&Credit_portfolio::fitSaddle_n, *this, a, b, xtol, rtol, 100, loss, n, eadxlgd, pd_c);
}

std::tuple<double, double> Credit_portfolio::getSaddleNewton(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0, double tol)
{
    std::tuple<double, double> k12(this->K12(s0, n, eadxlgd, pd_c));
    double fs(std::get<0>(k12) - loss);

    int ii(0);

    while ((abs(fs) > tol) & (std::get<1>(k12) > 1e-7) & (ii < 200))
    {
        s0 -= fs / std::get<1>(k12);
        k12 = this->K12(s0, n, eadxlgd, pd_c);
        fs = std::get<0>(k12) - loss;
        ii++;
    }

    return std::make_tuple(s0, fs);
}

double Credit_portfolio::fitSaddle_n_secur(double s, double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c)
{
    return this->K1_secur(s, n, eadxlgd, pd_c) - loss;
}

double Credit_portfolio::getSaddle_secur(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0, double a, double b, double tol)
{
    std::tuple<double, double> sfx = getSaddleNewton_secur(loss, n, eadxlgd, pd_c, s0, tol);

    if (abs(std::get<1>(sfx)) < tol) return std::get<0>(sfx);

    std::get<0>(sfx) = getSaddleBrent_secur(loss, n, eadxlgd, pd_c, a, b);
    std::get<1>(sfx) = K1_secur(std::get<0>(sfx), n, eadxlgd, pd_c) - loss;

    if (abs(std::get<1>(sfx)) > tol)
    {
        printf("loss: %.20f, a: %.20f, b: %.20f, s: %.20f, abs(fx): %.20f\n", loss, a, b, std::get<0>(sfx), abs(std::get<1>(sfx)));
        throw std::domain_error("A solution could not be found");
    }

    return std::get<0>(sfx);
}

double Credit_portfolio::getSaddleBrent_secur(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double a, double b, double xtol, double rtol)
{
    return CreditRisk::Optim::root_Brentq(&Credit_portfolio::fitSaddle_n_secur, *this, a, b, xtol, rtol, 100, loss, n, eadxlgd, pd_c);
}

std::tuple<double, double> Credit_portfolio::getSaddleNewton_secur(double loss, arma::vec n, arma::vec eadxlgd, arma::vec pd_c, double s0, double tol)
{
    std::tuple<double, double> k12(this->K12_secur(s0, n, eadxlgd, pd_c));
    double fs(std::get<0>(k12) - loss);

    int ii(0);

    while ((abs(fs) > tol) & (std::get<1>(k12) > 1e-7) & (ii < 200))
    {
        s0 -= fs / std::get<1>(k12);
        k12 = this->K12_secur(s0, n, eadxlgd, pd_c);
        fs = std::get<0>(k12) - loss;
        ii++;
    }

    return std::make_tuple(s0, fs);
}

double Credit_portfolio::K (double s, arma::vec pd_c)
{
    double k = 0;

    for (size_t ii = 0; ii < this->n; ii++) k+= (*this)[ii]->K(s, pd_c, ii);

    return k;
}

double Credit_portfolio::K1(double s, arma::vec pd_c)
{
    double k1 = 0;

    for (size_t ii = 0; ii < this->n; ii++) k1+= (*this)[ii]->K1(s, pd_c, ii);

    return k1;
}

double Credit_portfolio::K2(double s, arma::vec pd_c)
{
    double k2 = 0;

    for (size_t ii = 0; ii < this->n; ii++) k2+= (*this)[ii]->K2(s, pd_c, ii);

    return k2;
}

std::tuple<double, double, double> Credit_portfolio::K012(double s, arma::vec pd_c)
{
    std::tuple<double, double, double> k012(0, 0, 0);

    for (size_t ii = 0; ii < this->n; ii++)
    {
        std::tuple<double, double, double> k = (*this)[ii]->K012(s, pd_c, ii);
        std::get<0>(k012) += std::get<0>(k);
        std::get<1>(k012) += std::get<1>(k);
        std::get<2>(k012) += std::get<2>(k);
    }

    return k012;
}

std::tuple<double, double> Credit_portfolio::K12(double  s, arma::vec pd_c)
{
    std::tuple<double, double> k12(0, 0);

    for (size_t ii = 0; ii < this->n; ii++)
    {
        std::tuple<double, double> k = (*this)[ii]->K12(s, pd_c, ii);
        std::get<0>(k12) += std::get<0>(k);
        std::get<1>(k12) += std::get<1>(k);
    }

    return k12;
}

double Credit_portfolio::fitSaddle(double s, double loss, arma::vec pd_c)
{
    return pow(this->K1(s, pd_c) - loss, 2);
}

double Credit_portfolio::getSaddle(double loss, arma::vec  pd_c, double s0, double a, double b, double tol)
{
    std::tuple<double, double> sfx = getSaddleNewton(loss, pd_c, s0, tol);

    if (abs(std::get<1>(sfx)) < tol) return std::get<0>(sfx);

    std::get<0>(sfx) = getSaddleBrent(loss, pd_c, a, b, tol);
    std::get<1>(sfx) = K1(std::get<0>(sfx), pd_c) - loss;

    if (abs(std::get<1>(sfx)) > tol)
    {
        printf("loss: %.20f, a: %.20f, b: %.20f, s: %.20f, abs(fx): %.20f\n", loss, a, b, std::get<0>(sfx), abs(std::get<0>(sfx)));
        throw std::domain_error("A solution could not be found");
    }

    return std::get<0>(sfx);
}

std::tuple<double, double> Credit_portfolio::getSaddleNewton(double loss, arma::vec pd_c, double s0, double tol)
{
    std::tuple<double, double> k12(K12(s0, pd_c));
    double fs(std::get<0>(k12) - loss);

    int ii(0);

    while ((abs(fs) > tol) & (std::get<1>(k12) > 1e-7) & (ii < 200))
    {
        s0 -= fs / std::get<1>(k12);
        k12 = K12(s0, pd_c);
        fs = std::get<0>(k12) - loss;
        ii++;
    }

    return std::make_tuple(s0, fs);
}

double Credit_portfolio::getSaddleBrent(double loss, arma::vec  pd_c, double a, double b, double tol)
{
    return CreditRisk::Optim::root_Brent(&Credit_portfolio::fitSaddle, *this, a, b, tol, loss, pd_c);
}

/*
 *
 */

void Credit_portfolio::saddle_point_pd(double loss, arma::vec * n, arma::vec * eadxlgd, arma::mat * pd_c, CreditRisk::Integrator::PointsAndWeigths * points, arma::vec * saddle_points, size_t id, size_t p)
{
    double s = -1000, prob;
    std::tuple<double, double, double> k012;

    while (id < points->points.size())
    {
        s = getSaddle(loss, *n, *eadxlgd, pd_c->col(id), s);

        k012 = K012(s, *n, *eadxlgd, pd_c->col(id));

        prob = exp(std::get<0>(k012) - s * std::get<1>(k012) + 0.5 * std::get<2>(k012) * pow(s, 2)) * CreditRisk::Utils::pnorm(-sqrt(std::get<2>(k012) * pow(s, 2)));
        prob = isnan(prob) ? 0 : prob;

        if (s >= 0) prob = 1 - prob;
        (*saddle_points)[id] = prob * points->weigths[id];

        id += p;
    }
}

double Credit_portfolio::cdf(double loss, arma::vec n, arma::vec eadxlgd, arma::mat pd_c, CreditRisk::Integrator::PointsAndWeigths * points, size_t p)
{
    arma::vec prob(points->points.size());

    vector<std::thread> threads(p);

    for (size_t ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::saddle_point_pd, this, loss, &n, &eadxlgd, &pd_c, points, &prob, ii, p);
    }

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii].join();
    }

    return arma::accu(prob);
}

double Credit_portfolio::fitQuantile_pd(double loss, double prob, arma::vec n, arma::vec eadxlgd, arma::mat pd_c, CreditRisk::Integrator::PointsAndWeigths & points, size_t p)
{
    return this->cdf(loss, n, eadxlgd, pd_c, &points, p) - prob;
}

double Credit_portfolio::quantile(double prob, arma::vec n, arma::vec eadxlgd, arma::mat pd_c, CreditRisk::Integrator::PointsAndWeigths * points, double xtol, double rtol, size_t p)
{
    // return root_secant(&Credit_portfolio::fitQuantile_pd, *this, 0.01, 0.99, 1e-9, prob, n, eadxlgd, pd_c, *points, p);
    return CreditRisk::Optim::root_Brentq(&Credit_portfolio::fitQuantile_pd, *this, 0.01, 0.5, xtol, rtol, 100, prob, n, eadxlgd, pd_c, *points, p);
}

void Credit_portfolio::contrib_without_secur(double loss, arma::vec * n, arma::vec * eadxlgd, arma::mat * pd_c, arma::vec * con,
                                             arma::vec * c_contrib, CreditRisk::Integrator::PointsAndWeigths * points, size_t id, size_t p)
{
    double s = -1000, i_contrib;
    std::tuple<double, double, double> k012;

    while (id < points->points.size())
    {
        s = getSaddle(loss, *n, *eadxlgd, pd_c->col(id), s);

        k012 = K012(s, *n, *eadxlgd, pd_c->col(id));

        (*con)[id] = points->weigths[id] * (exp(std::get<0>(k012) - std::get<1>(k012) * s) / (sqrt(std::get<2>(k012))));

        for (size_t ii = 0; ii < c_contrib->size(); ii++)
        {
            i_contrib = (*con)[id] * saddle::K1(s, 1, (*eadxlgd)[ii], pd_c->at(ii, id));
            mu_p.lock();
            (*c_contrib)[ii] += i_contrib;
            mu_p.unlock();
        }

        id += p;
    }
}

arma::vec Credit_portfolio::getContrib_without_secur(double loss,  arma::vec n, arma::vec eadxlgd, arma::mat pd_c, CreditRisk::Integrator::PointsAndWeigths * points, size_t p)
{
    arma::vec con(points->points.size(), arma::fill::zeros);
    arma::vec c_contrib(this->getN(), arma::fill::zeros);

    std::vector<std::thread> threads(p);

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::contrib_without_secur, this, loss, &n, &eadxlgd, &pd_c, &con, &c_contrib, points, ii, p);
    }

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii].join();
    }

    double t_contrib = arma::accu(con);

    c_contrib /= t_contrib;

    return c_contrib;
}

void Credit_portfolio::contrib(double loss, arma::vec * n, arma::vec * eadxlgd, arma::mat * pd_c, arma::vec * con,
                               arma::vec * c_contrib, Integrator::PointsAndWeigths * points, size_t id, size_t p)
{
    double s = -1000, i_contrib;
    std::tuple<double, double, double> k012;
    arma::vec k1s_port(this->size());
    arma::vec ts(this->size());

    while (id < points->points.size())
    {
        s = getSaddle_secur(loss, *n, *eadxlgd, pd_c->col(id), s);

        k1s_port = this->K1_secur_vec(s, *n, *eadxlgd, pd_c->col(id));
        ts = this->get_t_secur(s, *n, *eadxlgd, pd_c->col(id), k1s_port, points->points.at(id));
        // mirar

        pd_c->col(id) = this->pd_c(ts, points->points.at(id));

        k012 = K012(s, *n, *eadxlgd, pd_c->col(id));

        (*con)[id] = points->weigths[id] * (exp(std::get<0>(k012) - std::get<1>(k012) * s) / (sqrt(std::get<2>(k012))));

        for (size_t ii = 0; ii < c_contrib->size(); ii++)
        {
            i_contrib = (*con)[id] * saddle::K1(s, 1, (*eadxlgd)[ii], pd_c->at(ii, id));
            //printf("%.20f\n", saddle::K1(s, 1, (*eadxlgd)[ii], (*pd_c)[ii]));
            mu_p.lock();
            (*c_contrib)[ii] += i_contrib;
            mu_p.unlock();
        }

        id += p;
    }
}

arma::vec Credit_portfolio::getContrib(double loss,  arma::vec n, arma::vec eadxlgd, arma::mat pd_c, CreditRisk::Integrator::PointsAndWeigths * points, size_t p)
{
    arma::vec con(points->points.size(), arma::fill::zeros);
    arma::vec c_contrib(this->getN(), arma::fill::zeros);

    std::vector<std::thread> threads(p);

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii] = std::thread(&Credit_portfolio::contrib, this, loss, &n, &eadxlgd, &pd_c, &con, &c_contrib, points, ii, p);
    }

    for (unsigned long ii = 0; ii < p; ii++)
    {
        threads[ii].join();
    }

    double t_contrib = arma::accu(con);

    c_contrib /= t_contrib;

    return c_contrib;
}

double Credit_portfolio::EVA(arma::vec eadxlgd, arma::vec contrib)
{
    size_t jj(0);
    double eva = 0;

    for (auto & ii: *this)
    {
        for (auto & kk: *ii)
        {
            eva += kk.EVA(eadxlgd[jj], contrib[jj], ii->CtI, ii->rf, ii->tax, ii->HR);
            jj++;
        }
    }

    return eva;
}
}