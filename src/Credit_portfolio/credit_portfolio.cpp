#include "credit_portfolio.h"

std::mutex mu_p;

namespace CreditRisk
{

Credit_portfolio::Credit_portfolio(arma::mat cor):  n(0), T_EAD(0), T_EADxLGD(0), cf(cor) {}

Credit_portfolio::Credit_portfolio(arma::mat cor, Transition &transition, Spread &spread) :
    n(0), m_transition(std::make_shared<Transition>(std::move(transition))),
    m_spread(std::make_shared<Spread>(std::move(spread))), T_EAD(0),
    T_EADxLGD(0), cf(cor)
{}

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

    if ((this->m_spread != nullptr) && (this->m_transition != nullptr))
    {
        for (auto & ii: value)
        {
            ii.setMigration(this->m_transition.get(), this->m_spread.get(), value.rf);
        }
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

    if ((this->m_spread != nullptr) && (this->m_transition != nullptr))
    {
        for (auto & ii: value)
        {
            ii.setMigration(this->m_transition.get(), this->m_spread.get(), value.rf);
        }
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

    if ((this->m_spread != nullptr) && (this->m_transition != nullptr))
    {
        root.add_child("transition", m_transition->to_ptree());
        root.add_child("spread", m_spread->to_ptree());
    }

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
    boost::optional<pt::ptree &> transition =  value.get_child_optional( "transition" );
    boost::optional<pt::ptree &> spread =  value.get_child_optional( "spread" );

    std::unique_ptr<Transition> tr;
    std::unique_ptr<Spread> sp;
    if (!(!transition & !spread))
    {
        tr.reset(new Transition(Transition::from_ptree(value.get_child("transition"))));
        sp.reset(new Spread(Spread::from_ptree(value.get_child("spread"))));
    }

    Credit_portfolio p = (!transition & !spread) ?
                Credit_portfolio(CreditRisk::CorMatrix::from_ptree(value.get_child("cor_factor")).cor) :
                Credit_portfolio(CreditRisk::CorMatrix::from_ptree(value.get_child("cor_factor")).cor,
                                 *tr.get(),
                                 *sp.get());

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

Credit_portfolio Credit_portfolio::from_ect(string wholesale, string retail, string CorMatrix,
                                 string Funds, string transition, string spread)
{
    std::shared_ptr<Transition> tr;
    std::shared_ptr<Spread> sp;
    if ((transition != "") && (spread != ""))
    {
        tr.reset(new Transition(Transition::from_ect(transition)));
        sp.reset(new Spread(Spread::from_ect(spread)));
    }

    Credit_portfolio p = ((transition == "") && (spread == "")) ?
                Credit_portfolio(CreditRisk::CorMatrix::from_ect(CorMatrix).cor):
                Credit_portfolio(CreditRisk::CorMatrix::from_ect(CorMatrix).cor,
                                 *tr.get(),
                                 *sp.get());

    std::vector<std::shared_ptr<CreditRisk::Portfolio>> portfolios;

    portfolios.push_back(std::make_shared<Portfolio>(Portfolio("0")));

    std::ifstream input;

    if (Funds != "")
    {

        input.open(Funds);
        string header, buffer;
        std::vector<std::string> splitted;

        if (input.is_open())
        {
            std::getline(input, header);

            while (std::getline(input, buffer))
            {
                boost::algorithm::split(splitted, buffer, [](char c) { return c == ';'; });

                if (splitted.size() == 4)
                {
                    std::vector<std::shared_ptr<CreditRisk::Portfolio>>::iterator pos = std::find_if(portfolios.begin(),
                                                                                                     portfolios.end(),
                                                                                                     [&](const auto& val){ return val->name == splitted.at(0); });
                    if (pos != portfolios.end())
                    {
                        dynamic_cast<Fund*>(pos->get())->fundParam.push_back(FundParam(atof(splitted.at(3).c_str()),
                                                                                       atof(splitted.at(1).c_str()),
                                                                                       atof(splitted.at(2).c_str())));
                    } else
                    {
                        portfolios.push_back(std::make_shared<Fund>(Fund(splitted.at(0),
                                                                         atof(splitted.at(3).c_str()),
                                                                         atof(splitted.at(1).c_str()),
                                                                         atof(splitted.at(2).c_str()))));
                    }

                }
            }
            input.close();
        }
    }

    input.open(wholesale);

    if (input.is_open())
    {
        string header, buffer;
        std::vector<std::string> splitted;
        std::getline(input, header);

        while (std::getline(input, buffer))
        {
            boost::algorithm::split(splitted, buffer, [](char c) { return c == '\t'; });

            if (splitted.size() > 1)
            {
                arma::vec wei(p.n_factors());

                for (size_t jj = 0; jj < 17; jj++)
                {
                    wei.at(jj) = atof(splitted.at(10 + jj).c_str());
                }

                for (size_t jj = 17; jj < wei.size(); jj++)
                {
                    wei.at(jj) = 0;
                }

                CreditRisk::Equation eq(atoi(splitted.at(0).c_str()), wei);

                double pd = atof(splitted.at(7).c_str());
                double lgd = atof(splitted.at(5).c_str());
                double k = atof(splitted.at(9).c_str());
                double lgd_addon = lgd * (1 + ((lgd * (1 - lgd)) /(k * pow(lgd, 2) * (1 - pd))));
                lgd_addon = std::fmax(std::fmin(lgd_addon, 1), 0);

                CreditRisk::Element ele(atoi(splitted.at(2).c_str()), // ru
                                        atoi(splitted.at(3).c_str()), // n
                                        atof(splitted.at(4).c_str()), // ead
                                        (atof(splitted.at(8).c_str()) < 1) ? pow(1 + atof(splitted.at(7).c_str()), atof(splitted.at(8).c_str())) - 1 :
                                                                             atof(splitted.at(7).c_str()), // bonificada
                                        atof(splitted.at(7).c_str()), // pd
                                        atof(splitted.at(5).c_str()), // lgd
                                        lgd_addon,
                                        atof(splitted.at(10).c_str()), // beta
                                        atof(splitted.at(8).c_str()), // term
                                        CreditRisk::Element::Element::Treatment::Wholesale,
                                        std::move(eq));

                for (auto & ii: portfolios)
                {
                    if (ii->name == splitted.at(1)) *ii.get() + ele;
                }
            }
        }
        input.close();
    } else
    {
        throw std::invalid_argument("Wholesale file can not be opened");
    }

    input.open(retail);

    if (input.is_open())
    {
        string header, buffer;
        std::vector<std::string> splitted;
        std::getline(input, header);

        while (std::getline(input, buffer))
        {
            boost::algorithm::split(splitted, buffer, [](char c) { return c == '\t'; });

            if (splitted.size() > 1)
            {
                arma::vec wei(p.n_factors(), arma::fill::zeros);

                for (size_t jj = 0; jj < wei.size(); jj++)
                {
                    wei.at(jj) = atof(splitted.at(jj + 11).c_str());
                }

                CreditRisk::Equation eq(atoi(splitted.at(0).c_str()), wei);

                CreditRisk::Element ele(atoi(splitted.at(2).c_str()), // ru
                            atoi(splitted.at(4).c_str()), // n
                            atof(splitted.at(5).c_str()), // ead
                            atof(splitted.at(8).c_str()), // Bonificada
                            atof(splitted.at(8).c_str()), // pd
                            atof(splitted.at(6).c_str()), // lgd
                            std::fmax(std::fmin(atof(splitted.at(6).c_str()) * atof(splitted.at(10).c_str()), 1), 0), // lgd_addon
                            atof(splitted.at(9).c_str()), // beta
                            1, // term
                            CreditRisk::Element::Element::Treatment::Retail,
                            std::move(eq));

                for (auto & ii: portfolios)
                {
                    if (ii->name == splitted.at(1)) *ii.get() + ele;
                }
            }
        }
        input.close();
    } else
    {
        throw std::invalid_argument("Retail file can not be opened");
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

Credit_portfolio Credit_portfolio::from_csv(string Portfolios, string Funds, string Elements, string CorMatrix, size_t n_factors,
                                            string transition, string spread)
{
    std::shared_ptr<Transition> tr;
    std::shared_ptr<Spread> sp;
    if ((transition != "") && (spread != ""))
    {
        tr.reset(new Transition(Transition::from_csv(transition)));
        sp.reset(new Spread(Spread::from_csv(spread)));
    }

    Credit_portfolio p = ((transition == "") && (spread == "")) ?
                Credit_portfolio(CreditRisk::CorMatrix::from_csv(CorMatrix, n_factors).cor):
                Credit_portfolio(CreditRisk::CorMatrix::from_csv(CorMatrix, n_factors).cor,
                                 *tr.get(),
                                 *sp.get());

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

            if (header == "Fund,attachment,detachment,purchase" || header == "Fund,attachment,detachment,purchase\r")
            {
                while (std::getline(input, buffer))
                {
                    boost::algorithm::split(splitted, buffer, [](char c) { return c == ','; });

                    if (splitted.size() == 8)
                    {
                        if (name != splitted.at(0))
                        {
                            name = splitted.at(0);
                            portfolios.push_back(std::make_shared<CreditRisk::Fund>(CreditRisk::Fund(splitted.at(0),
                                                                                                     atof(splitted.at(3).c_str()),
                                                                                                     atof(splitted.at(1).c_str()),
                                                                                                     atof(splitted.at(2).c_str()))));
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
                    wei.at(jj) = atof(splitted.at(14 + jj).c_str());
                }

                CreditRisk::Equation eq(atoi(splitted.at(13).c_str()), wei);

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
                            atof(splitted.at(11).c_str()),
                            (splitted.at(12) == "wholesale") ? CreditRisk::Element::Element::Treatment::Wholesale : CreditRisk::Element::Element::Treatment::Retail,
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

Credit_portfolio Credit_portfolio::from_xlsx_ps(string file, string transition, string spread)
{
    std::shared_ptr<Transition> tr;
    std::shared_ptr<Spread> sp;
    if ((transition != "") && (spread != ""))
    {
        tr.reset(new Transition(Transition::from_csv(transition)));
        sp.reset(new Spread(Spread::from_csv(spread)));
    }

    Credit_portfolio p = ((transition == "") && (spread == "")) ?
                Credit_portfolio({{1, 0}, {0, 1}}):
                Credit_portfolio({{1, 0}, {0, 1}},
                                 *tr.get(),
                                 *sp.get());


    OpenXLSX::XLDocument doc(file);
    auto wkb = doc.workbook();
    auto sheets = wkb.sheetNames();

    std::regex reg("Informe_Reparto");

    auto ii = sheets.begin();

    while (ii != sheets.end())
    {
        if (std::regex_search(ii->c_str(), reg))
        {
            break;
        }
        ii++;
    }

    if (ii == sheets.end()) throw std::invalid_argument("No sheets called Informe_Reparto has been found");

    auto sheet = wkb.worksheet(*ii);

    auto instrument = sheet.range(OpenXLSX::XLCellReference(6, 23), OpenXLSX::XLCellReference(5e5, 23));

    std::vector<std::shared_ptr<CreditRisk::Portfolio>> portfolios;

    int counterparties = 0;

    for (auto & jj: instrument)
    {
        if (jj.valueType() != OpenXLSX::XLValueType::Empty)
        {
            auto kk = portfolios.begin();

            while (kk != portfolios.end())
            {
                if (jj.value().get<std::string>() == (*kk)->name) break;
                kk++;
            }

            if (kk == portfolios.end()) portfolios.push_back(std::make_shared<Portfolio>(Portfolio(jj.value().get<std::string>())));
            counterparties++;
        }
    }

    instrument = sheet.range(OpenXLSX::XLCellReference(6, 23), OpenXLSX::XLCellReference(counterparties + 5, 23));

    auto counter   = instrument.begin();
    auto idio_id   = sheet.range(OpenXLSX::XLCellReference(6, 2), OpenXLSX::XLCellReference(counterparties + 5, 2)).begin();
    auto position  = sheet.range(OpenXLSX::XLCellReference(6, 3), OpenXLSX::XLCellReference(counterparties + 5, 3)).begin();
    auto ru        = sheet.range(OpenXLSX::XLCellReference(6, 4), OpenXLSX::XLCellReference(counterparties + 5, 4)).begin();
    auto ead       = sheet.range(OpenXLSX::XLCellReference(6, 6), OpenXLSX::XLCellReference(counterparties + 5, 6)).begin();
    auto lgd       = sheet.range(OpenXLSX::XLCellReference(6, 7), OpenXLSX::XLCellReference(counterparties + 5, 7)).begin();
    auto lgd_addon = sheet.range(OpenXLSX::XLCellReference(6, 8), OpenXLSX::XLCellReference(counterparties + 5, 8)).begin();
    auto pd_sb     = sheet.range(OpenXLSX::XLCellReference(6, 10), OpenXLSX::XLCellReference(counterparties + 5, 10)).begin();
    auto pd        = sheet.range(OpenXLSX::XLCellReference(6, 11), OpenXLSX::XLCellReference(counterparties + 5, 11)).begin();
    auto rho       = sheet.range(OpenXLSX::XLCellReference(6, 12), OpenXLSX::XLCellReference(counterparties + 5, 12)).begin();
    auto plazo     = sheet.range(OpenXLSX::XLCellReference(6, 18), OpenXLSX::XLCellReference(counterparties + 5, 18)).begin();
    auto beta1     = sheet.range(OpenXLSX::XLCellReference(6, 19), OpenXLSX::XLCellReference(counterparties + 5, 19)).begin();
    auto beta2     = sheet.range(OpenXLSX::XLCellReference(6, 20), OpenXLSX::XLCellReference(counterparties + 5, 20)).begin();
    auto tramo     = sheet.range(OpenXLSX::XLCellReference(6, 21), OpenXLSX::XLCellReference(counterparties + 5, 21)).begin();

    while (counter != instrument.end())
    {
        CreditRisk::Equation eq(idio_id->value().get<int>(),
                                {(beta1->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(beta1->value().get<int>()) : beta1->value().get<double>(),
                                 (beta2->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(beta2->value().get<int>()) : beta1->value().get<double>()});

        CreditRisk::Element ele(ru->value().get<unsigned long>(),
                                position->value().get<unsigned long>(),
                                ((ead->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(ead->value().get<int>()) : ead->value().get<double>()) / static_cast<double>(position->value().get<unsigned long>()),
                                (pd->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(pd->value().get<int>()) : pd->value().get<double>(),
                                (pd_sb->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(pd_sb->value().get<int>()) : pd_sb->value().get<double>(),
                                (lgd->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(lgd->value().get<int>()) : lgd->value().get<double>(),
                                (lgd_addon->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(lgd_addon->value().get<int>()) : lgd_addon->value().get<double>(),
                                (rho->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(rho->value().get<int>()) : rho->value().get<double>(),
                                (plazo->valueType() != OpenXLSX::XLValueType::Empty) ? (plazo->valueType() == OpenXLSX::XLValueType::Integer) ? static_cast<double>(plazo->value().get<int>()) : plazo->value().get<double>() : 1,
                                (tramo->valueType() != OpenXLSX::XLValueType::Empty) ? CreditRisk::Element::Element::Treatment::Retail : CreditRisk::Element::Element::Treatment::Wholesale,
                                std::move(eq));


        for (auto & kk: portfolios)
        {
            if (kk->name == counter->value().get<std::string>()) *kk.get() + ele;
        }

        idio_id++;
        position++;
        ru++;
        ead++;
        lgd++;
        lgd_addon++;
        pd_sb++;
        pd++;
        rho++;
        plazo++;
        beta1++;
        beta2++;
        counter++;
        tramo++;
    }

    for (auto & ii: portfolios)
    {
        p + *ii;
    }

    doc.close();

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
    arma::vec vec(this->getN());
    auto jj = vec.begin();

    for (auto & ii: *this)
    {
        for (auto & kk: *ii)
        {
            *jj = kk._le / (this->T_EADxLGD * kk.n);
            jj++;
        }
    }

    return vec;
}

std::shared_ptr<LStates> Credit_portfolio::get_std_states(bool migration)
{
    std::shared_ptr<LStates> vec(new LStates(this->getN()));
    auto jj = vec->begin();

    for (auto & ii: *this)
    {
        for (auto & kk: *ii)
        {
            *jj = kk.l_states(migration) / (this->T_EADxLGD * kk.n);
            jj++;
        }
    }

    return vec;
}

arma::vec Credit_portfolio::get_EADxLGDs()
{
    arma::vec vec(this->getN());
    auto jj = vec.begin();


    for (auto & ii: *this)
    {
        for (auto & kk: *ii)
        {
            *jj = kk._le / kk.n;
            jj++;
        }
    }

    return vec;
}

std::vector<double> Credit_portfolio::get_portfolios_EADs()
{
    std::vector<double> vec(this->size());
    auto jj = vec.begin();

    for (auto &ii: *this)
    {
        *jj = ii->getT_EAD();
        jj++;
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

arma::vec Credit_portfolio::get_ru_el()
{
    arma::vec el(this->rus.size(), arma::fill::zeros);

    size_t ii = 0;

    for (auto & jj: *this)
    {
        for (auto & kk: *jj)
        {
            el[this->rus_pos.at(ii)] += kk.el();
            ii++;
        }
    }

    return el;
}

arma::vec Credit_portfolio::get_ru_allocation(const arma::vec & contrib)
{
    arma::vec ru_alloc(rus.size(), arma::fill::zeros);
    size_t hh = 0;

    for (auto & ii: contrib)
    {
        ru_alloc.at(this->rus_pos.at(hh)) += ii;
        hh++;
    }

    return ru_alloc;
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

arma::mat Credit_portfolio::m_rand(size_t n, unsigned long seed, TP::ThreadPool * pool)
{
    arma::mat ale   = arma::zeros(n, this->n_factors());

    vector<std::future<void>> futures(pool->size());

    for (unsigned long ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::pv_rand, this, &ale, n, seed, ii, pool->size());
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return ale;
}

double Credit_portfolio::d_rand(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->n_factors()))
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
    arma::vec l(this->getN());
    auto jj = l.begin();

    for (auto & ii: *this)
    {
        for (auto & kk: *ii)
        {
            *jj = CreditRisk::Utils::randn_s(kk.equ.idio_seed + id);
            jj++;
        }
    }

    return l;
}

arma::mat Credit_portfolio::getIdio(size_t n, TP::ThreadPool * pool)
{
    arma::mat m(n, this->getN());

    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::pmIdio, this, &m, n, ii, pool->size());
    }

    for (auto & ii: futures)
    {
        ii.get();
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
            return kk;
        } else
        {
            jj += this->at(ii)->size();
            kk++;
        }
    }
    throw std::invalid_argument("Invalid element number");
}

Transition * Credit_portfolio::get_transition()
{
    return this->m_transition.get();
}

Spread * Credit_portfolio::get_spread()
{
    return this->m_spread.get();
}


double Credit_portfolio::d_Idio(size_t row, size_t column, size_t n)
{
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->getN()))
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


arma::mat Credit_portfolio::getCWIs(size_t n, unsigned long seed, TP::ThreadPool * pool)
{
    arma::mat l(n, this->getN());
    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::pmCWI, this, &l, n, seed, ii, pool->size());
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

double Credit_portfolio::d_CWI(size_t row, size_t column, size_t n, unsigned long seed)
{
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->getN()))
    {
        arma::vec f = this->v_rand(seed + row);
        return this->get_element(column).equ.CWI(f, static_cast<unsigned long>(row));
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

arma::vec Credit_portfolio::marginal(arma::vec f, unsigned long idio_id, bool migration)
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
                                l[jj] = ((t_at[hh] < v_t[kk]) && (v_t[kk] < t_de[hh])) * (*ii)[kk]._le +
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
                l[jj] =(*ii)[kk].loss(f, idio_id, migration);
                jj++;
            }
        }
    }

    return l;
}

arma::vec Credit_portfolio::marginal(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = this->v_rand(seed);
    return marginal(f, idio_id, migration);
}

void Credit_portfolio::pmloss(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p, bool migration)
{
   while (id < n)
    {
        l->row(id) = this->marginal(seed + id, id, migration).t();
        id += p;
    }
}


double Credit_portfolio::smargin_loss(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario, bool migration)
{
    // Implementar
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->getN()))
    {
        Element * element = &this->get_element(column);

        long ff = this->which_fund(column);

        if (ff < 0)
        {
            return element->loss(this->d_CWI(row, column, n, seed), migration);
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
                        l += ((ii.t_at < t) && (t < ii.t_de)) * element->_le +
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

double Credit_portfolio::smargin_loss_without_secur(size_t row, size_t column, size_t n, unsigned long seed, bool migration)
{
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->getN()))
    {
        return this->get_element(column).loss(this->d_CWI(row, column, n, seed), migration);
    }

    return 0;
}

arma::mat Credit_portfolio::margin_loss(size_t n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::mat l(n, this->getN());

    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::pmloss, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

arma::vec Credit_portfolio::marginal_without_secur(arma::vec f, unsigned long idio_id, bool migration)
{
    size_t jj(0);
    arma::vec l(this->getN(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            l[jj] =(*ii)[kk].loss(f, idio_id, migration);
            jj++;
        }
    }

    return l;
}

arma::vec Credit_portfolio::marginal_without_secur(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = this->v_rand(seed);
    return marginal_without_secur(f, idio_id, migration);
}


void Credit_portfolio::pmloss_without_secur(arma::mat *l, size_t n, unsigned long seed, size_t id, size_t p, bool migration)
{
   while (id < n)
    {
        l->row(id) = this->marginal_without_secur(seed + id, id, migration).t();
        id += p;
    }
}

arma::mat Credit_portfolio::margin_loss_without_secur(size_t n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::mat l(n, this->getN());

    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::pmloss_without_secur, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

double Credit_portfolio::sLoss_ru(size_t row, size_t column, size_t n, unsigned long seed, Scenario_data & scenario, bool migration)
{
    // Implementar
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->rus.size()))
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
                        loss += jj.loss(f, static_cast<unsigned long>(row), migration);
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
                                    l += ((ii.t_at < t) && (t < ii.t_de)) * jj._le +
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

double Credit_portfolio::sLoss_ru_without_secur(size_t row, size_t column, size_t n, unsigned long seed, bool migration)
{
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->rus.size()))
    {
        double loss = 0;

        arma::vec f = this->v_rand(seed + row);

        for (auto & ii: *this)
        {
            for (auto & jj: *ii)
            {
                if (jj.ru == this->rus.at(column))
                {
                    loss += jj.loss(f, static_cast<unsigned long>(row), migration);
                }
            }
        }

        return loss;
    }

    return 0;
}

arma::vec Credit_portfolio::sLoss_ru(arma::vec  f, unsigned long idio_id, bool migration)
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
                                l[this->rus_pos[jj]] += (((t_at[hh] < v_t[kk]) && (v_t[kk] < t_de[hh])) * (*ii)[kk]._le +
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
                l[this->rus_pos[jj]] += (*ii)[kk].loss(f, idio_id, migration);
                jj++;
            }
        }
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_ru(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = v_rand(seed);
    return sLoss_ru(f, idio_id, migration);
}

void Credit_portfolio::ploss_ru(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_ru(seed + id, id, migration).t();
        id += p;
    }
}

arma::mat Credit_portfolio::loss_ru(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::mat l(n, this->rus.size());
    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::ploss_ru, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_ru_without_secur(arma::vec  f, unsigned long idio_id, bool migration)
{
    size_t jj(0);
    arma::vec l(this->rus.size(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            l[this->rus_pos[jj]] += (*ii)[kk].loss(f, idio_id, migration);
            jj++;
        }
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_ru_without_secur(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = v_rand(seed);
    return sLoss_ru_without_secur(f, idio_id, migration);
}

void Credit_portfolio::ploss_ru_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_ru_without_secur(seed + id, id, migration).t();
        id += p;
    }
}

arma::mat Credit_portfolio::loss_ru_without_secur(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::mat l(n, this->rus.size());
    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::ploss_ru_without_secur, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }


    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio(arma::vec  f, unsigned long idio_id, bool migration)
{
    size_t hh(0);
    arma::vec l(this->size(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            l[hh] = dynamic_cast<CreditRisk::Fund*>(ii.get())->loss_sec(f, idio_id);
        } else {
            l[hh] = ii->loss(f, idio_id, migration);
        }
        hh++;
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = v_rand(seed);
    return sLoss_portfolio(f, idio_id, migration);
}

void Credit_portfolio::ploss_portfolio(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_portfolio(seed + id, id, migration).t();
        id += p;
    }
}

double Credit_portfolio::sLoss_portfolio(size_t row, size_t column, size_t n, unsigned long seed, bool migration)
{
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->size()))
    {
        arma::vec f = this->v_rand(seed + row);
        if (dynamic_cast<Fund*>(this->at(column).get()) != nullptr)
        {
            return dynamic_cast<Fund*>(this->at(column).get())->loss_sec(f, row);
        } else
        {
            return this->at(column)->loss(f, row, migration);
        }
    }

    return 0;
}

double Credit_portfolio::sLoss_portfolio_without_secur(size_t row, size_t column, size_t n, unsigned long seed, bool migration)
{
    if ((row < n) & !(row < 0) & !(column < 0) && (column < this->size()))
    {
        arma::vec f = this->v_rand(seed + row);
        return this->at(column)->loss(f, row, migration);
    }

    return 0;
}

arma::mat Credit_portfolio::loss_portfolio(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::mat l(n, this->size());
    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::ploss_portfolio, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio_without_secur(arma::vec  f, unsigned long idio_id, bool migration)
{
    size_t hh(0);
    arma::vec l(this->size(), arma::fill::zeros);

    for (auto & ii: *this)
    {
        l[hh] = ii->loss(f, idio_id, migration);
        hh++;
    }

    return l;
}

arma::vec Credit_portfolio::sLoss_portfolio_without_secur(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = v_rand(seed);
    return sLoss_portfolio_without_secur(f, idio_id, migration);
}

void Credit_portfolio::ploss_portfolio_without_secur(arma::mat *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration)
{
    while (id < n)
    {
        l->row(id) = this->sLoss_portfolio_without_secur(seed + id, id, migration).t();
        id += p;
    }
}

arma::mat Credit_portfolio::loss_portfolio_without_secur(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::mat l(n, this->size());
    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::ploss_portfolio_without_secur, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

double Credit_portfolio::sLoss(arma::vec f, unsigned long idio_id, bool migration)
{
    double loss(0);

    for (auto & ii: *this)
    {
        if (dynamic_cast<CreditRisk::Fund*>(ii.get()) != nullptr)
        {
            loss += dynamic_cast<CreditRisk::Fund*>(ii.get())->loss_sec(f, idio_id);
        } else {
            loss += ii->loss(f, idio_id, migration);
        }

    }

    return loss;
}

double Credit_portfolio::sLoss(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = v_rand(seed);
    return sLoss(f, idio_id, migration);
}

void Credit_portfolio::ploss(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration)
{
    while (id < n)
    {
        l->row(id) = sLoss(seed + id, id, migration);
        id += p;
    }
}

arma::vec Credit_portfolio::loss(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::vec l(n);
    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::ploss, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

double Credit_portfolio::sLoss_without_secur(arma::vec f, unsigned long idio_id, bool migration)
{
    double loss(0);

    for (auto & ii: *this)
    {
        loss += ii->loss(f, idio_id, migration);
    }

    return loss;
}

double Credit_portfolio::sLoss_without_secur(unsigned long seed, unsigned long idio_id, bool migration)
{
    arma::vec f = v_rand(seed);
    return sLoss_without_secur(f, idio_id, migration);
}

void Credit_portfolio::ploss_without_secur(arma::vec *l, unsigned long n, unsigned long seed, unsigned long id, unsigned long p, bool migration)
{
    while (id < n)
    {
        l->row(id) = sLoss_without_secur(seed + id, id, migration);
        id += p;
    }
}

arma::vec Credit_portfolio::loss_without_secur(unsigned long n, unsigned long seed, TP::ThreadPool * pool, bool migration)
{
    arma::vec l(n);
    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::ploss_without_secur, this, &l, n, seed, ii, pool->size(), migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return l;
}

Scenario Credit_portfolio::pd_c(arma::vec t, double scenario, bool migration)
{
    Scenario vec(this->getN());
    size_t jj = 0;
    size_t hh = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec[jj] = (*ii)[kk].p_states_c(t[hh], scenario, migration);
            jj++;
        };
        hh++;
    }

    return vec;
}


Scenario Credit_portfolio::pd_c(double scenario, bool migration)
{
    Scenario vec(this->getN());
    size_t jj = 0;

    for (auto & ii: *this)
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            vec.at(jj) = (*ii)[kk].p_states_c(scenario, migration);
            jj++;
        }
    }

    return vec;
}

void Credit_portfolio::pd_c_fill(std::vector<Scenario> * pd_c_mig, size_t * ii, CreditRisk::Integrator::PointsAndWeigths * points, bool migration)
{
    size_t jj;
    while (*ii < pd_c_mig->size())
    {
        mu_p.lock();
        jj = *ii;
        *ii = *ii + 1;
        mu_p.unlock();
        if (jj < pd_c_mig->size())
        {
            pd_c_mig->at(jj) = this->pd_c(points->points(jj), migration);
        }
    }
}

std::shared_ptr<std::vector<Scenario>> Credit_portfolio::pd_c(CreditRisk::Integrator::PointsAndWeigths points, TP::ThreadPool * pool, bool migration)
{
    std::shared_ptr<std::vector<Scenario>> pd_c_mig(new std::vector<Scenario>(points.points.size()));

    vector<std::future<void>> futures(pool->size());

    size_t jj = 0;
    // this->pd_c_fill(&pd_c, &jj, &points);

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::pd_c_fill, this, pd_c_mig.get(), &jj, &points, migration);
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return pd_c_mig;
}


/*
 FUNCTIONS
*/

arma::vec Credit_portfolio::get_t_secur(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, arma::vec k1s, double scenario)
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


double Credit_portfolio::K (double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    double k = 0;

    for (size_t ii = 0; ii < n->size(); ii++)
    {
        k += saddle::K(s, n->at(ii), eadxlgd->at(ii), pd_c->at(ii));
    }

    return k;
}

double Credit_portfolio::K1(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    double k1 = 0;

    for (size_t ii = 0; ii < n->size(); ii++)
    {
        k1 += saddle::K1(s, n->at(ii), eadxlgd->at(ii), pd_c->at(ii));
    }

    return k1;
}

double Credit_portfolio::K1_secur(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    double k1 = 0;
    size_t jj = 0;

    for (auto &ii: *(this))
    {
        double k1p = 0;
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            k1p += saddle::K1(s, n->at(jj), eadxlgd->at(jj), pd_c->at(jj));
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

arma::vec Credit_portfolio::K1_secur_vec(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    arma::vec k1(this->size(), arma::fill::zeros);
    size_t jj = 0;
    size_t pp = 0;

    for (auto &ii: *(this))
    {
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            k1.at(pp) += saddle::K1(s, n->at(jj), eadxlgd->at(jj), pd_c->at(jj));
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

double Credit_portfolio::K2(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    double k2 = 0;

    for (size_t ii = 0; ii < n->size(); ii++)
    {
        k2 += saddle::K2(s, n->at(ii), eadxlgd->at(ii), pd_c->at(ii));
    }

    return k2;
}

std::tuple<double, double, double> Credit_portfolio::K012(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    return saddle::K012(s, n, eadxlgd, pd_c);
}

std::tuple<double, double>         Credit_portfolio::K12(double s, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    return saddle::K12(s, n, eadxlgd, pd_c);
}


std::tuple<double, double>         Credit_portfolio::K12_secur(double s, arma::vec *n, LStates *eadxlgd, Scenario *pd_c)
{
    std::tuple<double, double> k12(0, 0);

    double dnum, dnum2, dden, k1;
    size_t jj = 0;

    for (auto &ii: *(this))
    {
        double k1p = 0;
        for (size_t kk = 0; kk < ii->size(); kk++)
        {
            dnum = saddle::num(s, eadxlgd->at(jj), pd_c->at(jj));
            dnum2 = saddle::num(s, eadxlgd->at(jj), pd_c->at(jj));
            dden = saddle::den(s, eadxlgd->at(jj), pd_c->at(jj));
            k1 = n->at(jj) * dnum / dden;

            k1p = n->at(jj) * k1;
            std::get<1>(k12) += n->at(jj) * ((dnum2 / dden) - pow(k1, 2));
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

double Credit_portfolio::fitSaddle_n(double s, double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    return this->K1(s, n, eadxlgd, pd_c) - loss;
}

double Credit_portfolio::getSaddle(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double s0, double a, double b, double tol)
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

double Credit_portfolio::getSaddleBrent(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double a, double b, double xtol, double rtol)
{
    return CreditRisk::Optim::root_Brentq(&Credit_portfolio::fitSaddle_n, *this, a, b, xtol, rtol, 100, loss, n, eadxlgd, pd_c);
}

std::tuple<double, double> Credit_portfolio::getSaddleNewton(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double s0, double tol)
{
    std::tuple<double, double> k12(this->K12(s0, n, eadxlgd, pd_c));
    double fs(std::get<0>(k12) - loss);

    int ii(0);

    while ((abs(fs) > tol) && (std::get<1>(k12) > 1e-7) && (ii < 200))
    {
        s0 -= fs / std::get<1>(k12);
        k12 = this->K12(s0, n, eadxlgd, pd_c);
        fs = std::get<0>(k12) - loss;
        ii++;
    }

    return std::make_tuple(s0, fs);
}

double Credit_portfolio::fitSaddle_n_secur(double s, double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c)
{
    return this->K1_secur(s, n, eadxlgd, pd_c) - loss;
}

double Credit_portfolio::getSaddle_secur(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double s0, double a, double b, double tol)
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

double Credit_portfolio::getSaddleBrent_secur(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double a, double b, double xtol, double rtol)
{
    return CreditRisk::Optim::root_Brentq(&Credit_portfolio::fitSaddle_n_secur, *this, a, b, xtol, rtol, 100, loss, n, eadxlgd, pd_c);
}

std::tuple<double, double> Credit_portfolio::getSaddleNewton_secur(double loss, arma::vec * n, LStates * eadxlgd, Scenario * pd_c, double s0, double tol)
{
    std::tuple<double, double> k12(this->K12_secur(s0, n, eadxlgd, pd_c));
    double fs(std::get<0>(k12) - loss);

    int ii(0);

    while ((abs(fs) > tol) && (std::get<1>(k12) > 1e-7) && (ii < 200))
    {
        s0 -= fs / std::get<1>(k12);
        k12 = this->K12_secur(s0, n, eadxlgd, pd_c);
        fs = std::get<0>(k12) - loss;
        ii++;
    }

    return std::make_tuple(s0, fs);
}

/*
 *
 */

void Credit_portfolio::saddle_point(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                                    CreditRisk::Integrator::PointsAndWeigths * points, arma::vec * saddle_points, size_t id, size_t p)
{
    double s = -1000, prob;
    std::tuple<double, double, double> k012;

    while (id < points->points.size())
    {
        s = getSaddle(loss, n, eadxlgd, &pd_c->at(id), s);

        k012 = this->K012(s, n, eadxlgd, &pd_c->at(id));

        prob = exp(std::get<0>(k012) - s * std::get<1>(k012) + 0.5 * std::get<2>(k012) * pow(s, 2)) * CreditRisk::Utils::pnorm(-sqrt(std::get<2>(k012) * pow(s, 2)));
        prob = isnan(prob) ? 0 : prob;

        if (s >= 0) prob = 1 - prob;
        (*saddle_points)[id] = prob * points->weigths[id];

        id += p;
    }
}

double Credit_portfolio::cdf(double loss, arma::vec *n, LStates *eadxlgd, std::vector<Scenario> * pd_c,
                             CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool)
{
    arma::vec prob(points->points.size());
    //this->saddle_point(loss, n, eadxlgd, pd_c, points, &prob, 0, 1);

    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::saddle_point, this, loss, n, eadxlgd, pd_c, points, &prob, ii, pool->size());
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    return arma::accu(prob);
}

double Credit_portfolio::fitQuantile(double loss, double prob, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                                     CreditRisk::Integrator::PointsAndWeigths & points, TP::ThreadPool * pool)
{
    return this->cdf(loss, n, eadxlgd, pd_c, &points, pool) - prob;
}

double Credit_portfolio::quantile(double prob, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                                  CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool, double xtol, double rtol)
{
    // return root_secant(&Credit_portfolio::fitQuantile_pd, *this, 0.01, 0.99, 1e-9, prob, n, eadxlgd, pd_c, *points, p);
    return CreditRisk::Optim::root_Brentq(&Credit_portfolio::fitQuantile, *this, 0.01, 0.9, xtol, rtol, 100, prob, n, eadxlgd, pd_c, *points, pool);
}

void Credit_portfolio::contrib_without_secur(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c, arma::vec * con,
                                             arma::vec * c_contrib, CreditRisk::Integrator::PointsAndWeigths * points, size_t id, size_t p)
{
    double s = -1000, i_contrib;
    std::tuple<double, double, double> k012;

    while (id < points->points.size())
    {
        s = getSaddle(loss, n, eadxlgd, &pd_c->at(id), s);

        k012 = K012(s, n, eadxlgd, &pd_c->at(id));

        (*con)[id] = points->weigths[id] * (exp(std::get<0>(k012) - std::get<1>(k012) * s) / (sqrt(std::get<2>(k012))));

        for (size_t ii = 0; ii < c_contrib->size(); ii++)
        {
            i_contrib = (*con)[id] * saddle::K1(s, 1, eadxlgd->at(ii), pd_c->at(id).at(ii));
            mu_p.lock();
            (*c_contrib)[ii] += i_contrib;
            mu_p.unlock();
        }

        id += p;
    }
}

arma::vec Credit_portfolio::getContrib_without_secur(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                                                     CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool)
{
    arma::vec con(points->points.size(), arma::fill::zeros);
    arma::vec c_contrib(this->getN(), arma::fill::zeros);

    vector<std::future<void>> futures(pool->size());

    //this->contrib_without_secur(loss, n, eadxlgd, pd_c, &con, &c_contrib, points, 0, 1);

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::contrib_without_secur, this, loss, n, eadxlgd, pd_c, &con, &c_contrib, points, ii, pool->size());
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    double t_contrib = arma::accu(con);

    c_contrib /= t_contrib;

    return c_contrib;
}

void Credit_portfolio::contrib(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c, arma::vec * con,
                               arma::vec * c_contrib, Integrator::PointsAndWeigths * points, size_t id, size_t p)
{
    double s = -1000, i_contrib;
    std::tuple<double, double, double> k012;
    arma::vec k1s_port(this->size());
    arma::vec ts(this->size());

    while (id < points->points.size())
    {
        s = getSaddle_secur(loss, n, eadxlgd, &pd_c->at(id), s);

        k1s_port = this->K1_secur_vec(s, n, eadxlgd, &pd_c->at(id));
        ts = this->get_t_secur(s, n, eadxlgd, &pd_c->at(id), k1s_port, points->points.at(id));
        // mirar

        pd_c->at(id) = this->pd_c(ts, points->points.at(id));

        k012 = K012(s, n, eadxlgd, &pd_c->at(id));

        (*con)[id] = points->weigths[id] * (exp(std::get<0>(k012) - std::get<1>(k012) * s) / (sqrt(std::get<2>(k012))));

        for (size_t ii = 0; ii < c_contrib->size(); ii++)
        {
            i_contrib = (*con)[id] * saddle::K1(s, 1, eadxlgd->at(ii), pd_c->at(id).at(ii));
            //printf("%.20f\n", saddle::K1(s, 1, (*eadxlgd)[ii], (*pd_c)[ii]));
            mu_p.lock();
            (*c_contrib)[ii] += i_contrib;
            mu_p.unlock();
        }

        id += p;
    }
}

arma::vec Credit_portfolio::getContrib(double loss, arma::vec * n, LStates * eadxlgd, std::vector<Scenario> * pd_c,
                                       CreditRisk::Integrator::PointsAndWeigths * points, TP::ThreadPool * pool)
{
    arma::vec con(points->points.size(), arma::fill::zeros);
    arma::vec c_contrib(this->getN(), arma::fill::zeros);

    vector<std::future<void>> futures(pool->size());

    for (size_t ii = 0; ii < pool->size(); ii++)
    {
        futures.at(ii) = pool->post(&Credit_portfolio::contrib, this, loss, n, eadxlgd, pd_c, &con, &c_contrib, points, ii, pool->size());
    }

    for (auto & ii: futures)
    {
        ii.get();
    }

    double t_contrib = arma::accu(con);

    c_contrib /= t_contrib;

    return c_contrib;
}

double Credit_portfolio::EVA(LStates *eadxlgd, arma::vec contrib)
{
    size_t jj(0);
    double eva = 0;

    for (auto & ii: *this)
    {
        for (auto & kk: *ii)
        {
            eva += kk.EVA(eadxlgd->at(jj).back(), contrib[jj], ii->CtI, ii->rf, ii->tax, ii->HR);
            jj++;
        }
    }

    return eva;
}

/*
arma::vec Credit_portfolio::minimize_EAD_constant(arma::vec * n, std::vector<Scenario> * pd_c, CreditRisk::Integrator::PointsAndWeigths * points,
                                                  TP::ThreadPool * pool, double total_ead_var, std::vector<double> x0,
                                                  std::vector<double> lower, std::vector<double> upper)
{
    struct Arg_data
    {
        CreditRisk::Credit_portfolio * credit_portfolio;
        arma::vec * ns;
        std::vector<Scenario> * pd_c;
        CreditRisk::Integrator::PointsAndWeigths * points;
        std::vector<double> EAD_p;
        double total_ead_var, lower, upper, new_T_EAD;
        TP::ThreadPool * pool;
        size_t iter;

        Arg_data(CreditRisk::Credit_portfolio * credit_portfolio,
                 arma::vec * n, std::vector<Scenario> * pd_c,
                 CreditRisk::Integrator::PointsAndWeigths * points,
                 double total_ead_var, double lower, double upper, TP::ThreadPool * pool) :
            credit_portfolio(credit_portfolio), ns(n),
            pd_c(pd_c), points(points),
            EAD_p(credit_portfolio->get_portfolios_EADs()),
            total_ead_var(total_ead_var), lower(lower), upper(upper),
            new_T_EAD(credit_portfolio->T_EAD * (1 + total_ead_var)),
            pool(pool), iter(0) {}

        ~Arg_data() = default;

        size_t get_n()
        {
            return this->credit_portfolio->size() - 1;
        }

        std::vector<double> get_xn(std::vector<double> x)
        {
            double xn = new_T_EAD;
            std::vector<double> sol(x.size() + 1);

            for (size_t ii = 0; ii < x.size(); ii++)
            {
                xn -= (1 + x[ii]) * this->EAD_p[ii];
            }

            xn /= this->EAD_p[this->EAD_p.size() - 1];

            for (size_t ii = 0; ii < x.size(); ii++)
            {
                sol[ii] = x[ii];
            }

            sol[x.size()] = (xn - 1);

            return sol;
        }

        double check_ead(std::vector<double> x)
        {
            double t_ead = 0;

            for (size_t ii = 0; ii < x.size(); ii++)
            {
                t_ead += (1 + x[ii]) * this->EAD_p[ii];
            }

            return t_ead;
        }

        arma::vec std_eadxlgds(std::vector<double> x)
        {
            arma::vec std_eadsxlgds = arma::vec(this->credit_portfolio->getN());
            double T_EADxLGD = 0;
            size_t jj = 0;
            size_t kk = 0;

            for (auto & ii: *this->credit_portfolio)
            {
                for (size_t hh = 0; hh < ii->size(); hh++)
                {
                    std_eadsxlgds[jj] = (*ii)[hh].ead * (1 + x[kk]) * (*ii)[hh].lgd_addon;
                    T_EADxLGD += std_eadsxlgds.at(jj) * this->ns->at(jj);
                    jj++;
                }
                kk++;
            }

            std_eadsxlgds /= T_EADxLGD;
            return std_eadsxlgds;
        }


        double evaluate(std::vector<double> x)
        {
            arma::vec std_eadsxlgds = arma::vec(this->credit_portfolio->getN());
            double T_EADxLGD = 0;
            size_t jj = 0;
            size_t kk = 0;

            for (auto & ii: *this->credit_portfolio)
            {
                for (size_t hh = 0; hh < ii->size(); hh++)
                {
                    std_eadsxlgds[jj] = (*ii)[hh].ead * (1 + x[kk]) * (*ii)[hh].lgd_addon;
                    T_EADxLGD += std_eadsxlgds[jj] * this->ns->at(jj);
                    jj++;
                }
                kk++;
            }

            std_eadsxlgds /= T_EADxLGD;

            double loss = credit_portfolio->quantile(0.9995, this->ns, std_eadsxlgds, pd_c, points, this->pool, 1e-13, 1e-7);
            arma::vec contrib = this->credit_portfolio->getContrib_without_secur(loss, this->ns, std_eadsxlgds, this->pd_c, this->points, this->pool);

            return  this->credit_portfolio->EVA(std_eadsxlgds * T_EADxLGD, contrib * T_EADxLGD);
        }

    };

    auto fitnes = [] (const std::vector<double> &x, std::vector<double> &grad, void * args)
    {
        Q_UNUSED(grad);
        Q_UNUSED(args);

        Arg_data * parameters = static_cast<Arg_data *>(args);

        std::vector<double> sol = parameters->get_xn(x);

        if ((sol.at(sol.size() - 1) > parameters->upper) | (sol.at(sol.size() - 1) < parameters->lower)) return 1e10;
        double eva = parameters->evaluate(sol);
        parameters->iter++;

        //printf("iter %i\r", iter);
        std::cout << "Iter: " << parameters->iter << " f(x)= " << eva << " EAD: " << std::setprecision(16) << parameters->check_ead(sol);
        for (auto & ii: sol) std::cout << " " << ii << " ";
        std::cout << std::endl;

        return -eva;
    };

    Arg_data fitness = Arg_data(this, n, pd_c, points, total_ead_var, lower.at(lower.size() - 1),
                                upper.at(upper.size() - 1), pool);

    vector<double> down(fitness.get_n());

    for (size_t ii = 0; ii < down.size(); ii++)
    {
        down.at(ii) = lower.at(ii);
    }

    vector<double> up(fitness.get_n());

    for (size_t ii = 0; ii < up.size(); ii++)
    {
        up.at(ii) = upper.at(ii);
    }

    nlopt::opt optimizer(nlopt::GN_MLSL, fitness.get_n()); // GN_ISRES GN_ESCH GN_MLSL LN_COBYLA GN_CRS2_LM LN_AUGLAG_EQ NLOPT_LN_BOBYQA
    optimizer.set_lower_bounds(down);
    optimizer.set_upper_bounds(up);

    optimizer.set_min_objective(fitnes, (void*)&fitness);

    optimizer.set_xtol_rel(1e-9);
    optimizer.set_maxeval(10000);

    vector<double> init(fitness.get_n());

    for (size_t ii = 0; ii < init.size(); ii++)
    {
        init.at(ii) = x0.at(ii);
    }

    double minf;

    try{
        nlopt::result result = optimizer.optimize(init, minf);
        Q_UNUSED(result);
        std::cout << "found minimum at f(" << ") = "
                  << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    std::vector<double> res = fitness.get_xn(init);

    return res;
}
*/

}
