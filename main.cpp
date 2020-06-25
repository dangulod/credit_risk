#include <iostream>
#include <armadillo>
#include <credit_portfolio.h>
#include <chrono>
#include <nlopt.hpp>
#include <portfolio_optim.h>

using namespace std;

class Fitness_parameters
{
public:
    CreditRisk::Credit_portfolio * credit_portfolio;
    CreditRisk::Integrator::PointsAndWeigths points;
    arma::mat pd_c;
    arma::vec ns;
    std::vector<double> EAD_p;
    double total_ead_var, ead_var, new_T_EAD;

    Fitness_parameters() = delete;
    Fitness_parameters(CreditRisk::Credit_portfolio * credit_portfolio, CreditRisk::Integrator::PointsAndWeigths points, double total_ead_var, double ead_var):
        credit_portfolio(credit_portfolio), points(points),
        pd_c(credit_portfolio->pd_c(points)), ns(credit_portfolio->get_Ns()),
        EAD_p(credit_portfolio->get_portfolios_EADs()),
        total_ead_var(total_ead_var), ead_var(ead_var), new_T_EAD(credit_portfolio->T_EAD * (1 + total_ead_var)) {}
    Fitness_parameters(const Fitness_parameters & value) = delete;
    Fitness_parameters(Fitness_parameters && value) = default;
    ~Fitness_parameters() = default;

    size_t get_n()
    {
        return this->credit_portfolio->size() - 1;
    }

    std::vector<double> get_xn(unsigned n, const double *x)
    {
        double xn = new_T_EAD;
        std::vector<double> sol(n + 1);

        for (size_t ii = 0; ii < n; ii++)
        {
            xn -= (1 + x[ii]) * this->EAD_p[ii];
        }

        xn /= this->EAD_p[this->EAD_p.size() - 1];

        for (size_t ii = 0; ii < n; ii++)
        {
            sol[ii] = x[ii];
        }

        sol[n] = (xn - 1);

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
                T_EADxLGD += std_eadsxlgds[jj] * this->ns[jj];
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
                T_EADxLGD += std_eadsxlgds[jj] * this->ns[jj];
                jj++;
            }
            kk++;
        }

        std_eadsxlgds /= T_EADxLGD;

        double loss = credit_portfolio->quantile(0.9995, this->ns, std_eadsxlgds, pd_c, &points, 1e-13, 1e-7, 1);
        arma::vec contrib = this->credit_portfolio->getContrib_without_secur(loss, this->ns, std_eadsxlgds, this->pd_c, &this->points);

        return  this->credit_portfolio->EVA(std_eadsxlgds * T_EADxLGD, contrib * T_EADxLGD);
    }

};

static int iter = 0;

double Fitness_function(unsigned n, const double *x, double *grad, void *my_func_data)
{
    Fitness_parameters * parameters = static_cast<Fitness_parameters *>(my_func_data);

    std::vector<double> sol = parameters->get_xn(n, x);

    if (fabs(sol[n]) > parameters->ead_var) return 1e10;
    double eva = parameters->evaluate(sol);
    iter++;

    //printf("iter %i\r", iter);
    std::cout << "Iter: " << iter << " f(x)= " << eva << " EAD: " << std::setprecision(16) << parameters->check_ead(sol);
    for (auto & ii: sol) std::cout << " " << ii << " ";
    std::cout << std::endl;

    return -eva;
}

int main()
{
    /*
    CreditRisk::Credit_portfolio p = CreditRisk::Credit_portfolio::from_csv("/opt/share/data/titus/Portfolio.csv",
                                                    "/opt/share/data/titus/Fund.csv",
                                                    "/opt/share/data/titus/counter.csv",
                                                    "/opt/share/data/titus/cor.csv",
                                                    1);

    pt::write_json("/opt/share/data/titus/titus.json", p.to_ptree());
    //std::string file = "/opt/share/data/ES.json";
    std::string file = "/opt/share/data/optim/optim.json";
    pt::ptree pt;
    pt::read_json(file, pt);

    CreditRisk::Credit_portfolio p0 = CreditRisk::Credit_portfolio::from_ptree(pt);
    */

    /*
    CreditRisk::Integrator::PointsAndWeigths points = CreditRisk::Integrator::gki();
    arma::vec ns = p0.get_Ns();
    arma::vec eadxlgd = p0.get_std_EADxLGDs();
    arma::mat pd_c = p0.pd_c(points);

    double loss = p0.quantile(0.9995, ns, eadxlgd, pd_c, &points);
    arma::vec contrib = p0.getContrib(loss, ns, eadxlgd, pd_c, &points);
    double total = 0;

    for (size_t ii = 0; ii < contrib.size(); ii++)
    {
        total += contrib.at(ii) * ns.at(ii);
    }

    std::cout << "loss: " << loss  * p0.T_EADxLGD << std::endl;


    */
    // ========================= SECURITIZATIONS ===========================================
    /*
    std::string file = "/opt/share/data/titus/titus.json";
    // std::string file = "/opt/share/data/ES.json";
    pt::ptree pt;
    pt::read_json(file, pt);

    CreditRisk::Credit_portfolio p0 = CreditRisk::Credit_portfolio::from_ptree(pt);

    printf("==== COMPUTING LOSS ====\n");
    size_t n = 1e5;
    unsigned long seed = 987654321;
    double quantile = 0.9995;

    arma::vec total2 = p0.loss_without_secur(n, seed);
    arma::mat loss2 = p0.loss_portfolio_without_secur(n, seed);

    ofstream file_in2("/tmp/losses_sin.csv");

    for (auto &ii : p0)
    {
        file_in2 << ii->name << ",";
    }
    file_in2 << endl;
    loss2.save(file_in2, arma::csv_ascii);

    file_in2.close();

    double capital2 = CreditRisk::Utils::quantile(total2, quantile);

    printf("economic capital: %.20f\n", capital2);
    printf("economic capital std: %.20f\n", capital2 / p0.T_EADxLGD);

    ofstream file_in("/tmp/losses_con.csv");
    arma::vec total = p0.loss(n, seed);
    arma::mat loss = p0.loss_portfolio(n, seed);

    for (auto &ii : p0)
    {
        file_in << ii->name << ",";
    }
    file_in << endl;
    loss.save(file_in, arma::csv_ascii);

    file_in.close();

    double capital = CreditRisk::Utils::quantile(total, quantile);

    printf("economic capital: %.20f\n", capital);
    printf("economic capital std: %.20f\n", capital / p0.T_EADxLGD);
    printf("==== ALLOCATING CAPITAL ====\n");
    // economic capital: 355169.03234453254844993353
    // economic capital std: 0.00000392895696221294

    CreditRisk::Integrator::PointsAndWeigths points = CreditRisk::Integrator::gki();
    arma::vec ns = p0.get_Ns();
    arma::vec eadxlgd = p0.get_std_EADxLGDs();
    arma::mat pd_c = p0.pd_c(points);

    dynamic_cast<CreditRisk::Fund*>(p0.at(1).get())->fundParam.at(0).at /= p0.T_EADxLGD;
    dynamic_cast<CreditRisk::Fund*>(p0.at(1).get())->fundParam.at(0).de /= p0.T_EADxLGD;
    dynamic_cast<CreditRisk::Fund*>(p0.at(2).get())->fundParam.at(0).at /= p0.T_EADxLGD;
    dynamic_cast<CreditRisk::Fund*>(p0.at(2).get())->fundParam.at(0).de /= p0.T_EADxLGD;
    arma::vec contrib2 = p0.getContrib_without_secur(capital2 / p0.T_EADxLGD, ns, eadxlgd, pd_c, &points) * p0.T_EADxLGD;
    arma::vec contrib = p0.getContrib(capital / p0.T_EADxLGD, ns, eadxlgd, pd_c, &points) * p0.T_EADxLGD;

    ofstream file_c("/tmp/contrib_sin.csv");

    file_c << "Allocation" << endl;

    contrib2.save(file_c, arma::csv_ascii);

    file_c.close();

    ofstream file_c2("/tmp/contrib_con.csv");

    file_c2 << "Allocation" << endl;

    contrib.save(file_c2, arma::csv_ascii);

    file_c2.close();
    */
    /*
    printf("SIN TITUS\n");

    for (auto & ii: contrib2)
    {
        printf("%.20f\n", ii);
    }

    printf("CON TITUS\n");

    for (auto & ii: contrib)
    {
        printf("%.20f\n", ii);
    }
    */
    //std::string file = "/opt/share/data/ES.json";


    // ========================= OPTIMIZATION ===========================================

    std::string file = "/opt/share/data/optim/optim.json";
    pt::ptree pt;
    pt::read_json(file, pt);

    CreditRisk::Credit_portfolio p0 = CreditRisk::Credit_portfolio::from_ptree(pt);

    CreditRisk::Integrator::PointsAndWeigths points(CreditRisk::Integrator::ghi());
    arma::mat pd_c = p0.pd_c(points);
    arma::vec eadxlgd_std = p0.get_std_EADxLGDs();
    arma::vec ns = p0.get_Ns();
    double loss = p0.quantile(0.9995, ns, eadxlgd_std, pd_c, &points, 1e-12, 1e-6, 1);

    printf("t_eadxlgd %.20f\n", p0.T_EADxLGD);
    printf("t_ead %.20f\n", p0.T_EAD);
    printf("loss %.20f\n", loss);
    printf("loss %.20f\n", loss * p0.T_EADxLGD);
    printf("EL %.20f\n", p0.getPE());
    /*

    arma::vec contrib = p0.getContrib_without_secur(loss, ns, eadxlgd_std, pd_c, &points, 1);

    printf("eva %.20f\n", p0.EVA(eadxlgd_std * p0.T_EADxLGD, contrib  * p0.T_EADxLGD));

    CreditRisk::Portfolio_optim l(&p0, &points);

    arma::vec lb(p0.size());
    arma::vec ub(p0.size());
    l.total_ead_var(0);

    lb.fill(-0.1);
    ub.fill(0.1);

    l.lower_constraints(lb);
    l.upper_constraints(ub);

    arma::vec x(p0.size(), arma::fill::zeros);

    printf("evaluate: %.20f\n", l.evaluate(x));

    x = l.optim_portfolio_growth(x);

    printf("evaluate: %.20f\n", l.evaluate(x));
    std::cout << "growths: " << std::endl;
    x.print();
    */

    Fitness_parameters fitness = Fitness_parameters(&p0, CreditRisk::Integrator::ghi(), 0, 0.1);

    vector<double> lower(fitness.get_n());
    std::fill(lower.begin(), lower.end(), -0.1);

    vector<double> upper(fitness.get_n());
    std::fill(upper.begin(), upper.end(), 0.1);

    nlopt::opt optimizer(nlopt::LN_AUGLAG_EQ, fitness.get_n()); // GN_ISRES GN_ESCH GN_MLSL LN_COBYLA GN_CRS2_LM LN_AUGLAG_EQ NLOPT_LN_BOBYQA
    optimizer.set_lower_bounds(lower);
    optimizer.set_upper_bounds(upper);

    optimizer.set_min_objective(Fitness_function, (void*)&fitness);

    optimizer.set_xtol_rel(1e-9);
    optimizer.set_maxeval(10000);

    std::vector<double> x0(fitness.get_n());
    std::fill(x0.begin(), x0.end(), 0);

    auto dx = std::chrono::high_resolution_clock::now();

    double minf;

    try{
        nlopt::result result = optimizer.optimize(x0, minf);
        std::cout << "found minimum at f(" << ") = "
                  << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    for (auto & ii: x0) printf("%.20f\n", ii);

    auto dy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dif = dy - dx;
    std::cout << dif.count() << " seconds" << std::endl;

    return 0;
}

/*
 *
 * SADDLE POINT
 *
 */

/*
    arma::mat pd_c = cp.pd_c(points);
    arma::vec eadxlgd_std = cp.get_std_EADxLGDs();
    arma::vec ns = cp.get_Ns();

    auto dx = std::chrono::high_resolution_clock::now();

    double loss = cp.quantile(0.9995, cp.get_Ns(), cp.get_std_EADxLGDs(), cp.pd_c(points), &points, 1e-13, 1e-7);
    double prob = cp.cdf(loss, ns, eadxlgd_std, pd_c, &points, 1);
    arma::vec contrib = cp.getContrib_without_secur(loss, ns, eadxlgd_std, pd_c);

    double t_contrib = 0;

    for (size_t ii = 0; ii < cp.getN(); ii++)
    {
        t_contrib += ns[ii] * contrib[ii];
    }

    double eva = cp.EVA(eadxlgd_std * cp.T_EADxLGD, contrib * cp.T_EADxLGD);

    std::cout << "loss: " << loss << std::endl;
    std::cout << "probability: " << prob << std::endl;
    std::cout << "sum(contrib): " << t_contrib << std::endl;
    std::cout << "eva: " << eva << std::endl;

    auto dy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dif = dy - dx;
    std::cout << dif.count() << " seconds" << std::endl;

*/

/*
 *
 * MONTE CARLO TEST
 *
 */

/*

Fund * spv = dynamic_cast<Fund*>(cp[1].get());

double bi_at, bi_de, br_at, br_de, br2_at, br2_de, f_bi_at, f_bi_de, f_br_at, f_br_de, f_br2_at, f_br2_de;

for (size_t ii = 0; ii < 100; ii++)
{
    arma::vec f = cp.m_rand(1, 987654321 + ii).t();

    br_at = root_Brentq(&Fund::__fit_T_loss, *spv, 0, 1, 1e-12, 1e-6, 50, spv->fundParam[ii].at, f, ii);
    bi_at = root_bisection(&Fund::__fit_T_loss, *spv, 0, 1, 1e-12, true, spv->fundParam[ii].at, f, ii);
    br2_at = root_Brent(&Fund::__fit_T_loss, *spv, 0, 1, 1e-12, spv->fundParam[ii].at, f, ii);
    br_de = root_Brentq(&Fund::__fit_T_loss, *spv, 0, 1, 1e-12, 1e-6, 50, spv->fundParam[ii].de, f, ii);
    bi_de = root_bisection(&Fund::__fit_T_loss, *spv, 0, 1, 1e-12, false, spv->fundParam[ii].de, f, ii);
    br2_de = root_Brent(&Fund::__fit_T_loss, *spv, 0, 1, 1e-12, spv->fundParam[ii].de, f, ii);

    f_br_at = spv->__fit_T_loss(br_at, spv->fundParam[ii].at, f, ii);
    f_bi_at = spv->__fit_T_loss(bi_at, spv->fundParam[ii].at, f, ii);
    f_br2_at = spv->__fit_T_loss(br2_at, spv->fundParam[ii].at, f, ii);

    f_br_de = spv->__fit_T_loss(br_de, spv->fundParam[ii].de, f, ii);;
    f_bi_de = spv->__fit_T_loss(bi_de, spv->fundParam[ii].de, f, ii);
    f_br2_de = spv->__fit_T_loss(br2_de, spv->fundParam[ii].de, f, ii);;

    printf("===== ITER %i =====\n", ii);
    printf("Bisection t_at = %.20f, f(t_at) = %.20f\n", bi_at, f_bi_at);
    printf("Brent t_at = %.20f, f(t_at) = %.20f\n", br_at, f_br_at);
    printf("Brent2 t_at = %.20f, f(t_at) = %.20f\n", br2_at, f_br2_at);

    printf("Bisection t_de = %.20f, f(t_de) = %.20f\n", bi_de, f_bi_de);
    printf("Brent t_de = %.20f, f(t_de) = %.20f\n", br_de, f_br_de);
    printf("Brent2 t_de = %.20f, f(t_de) = %.20f\n\n", br2_de, f_br2_de);
}
*/


/*
auto dx = std::chrono::high_resolution_clock::now();

arma::mat l_ru_s = cp.loss_ru(n, seed);

auto dy = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> dif = dy - dx;
std::cout << dif.count() << " seconds" << std::endl;

printf("====== WITHOUT SECURITIZATIONS =====\n");

dx = std::chrono::high_resolution_clock::now();

arma::mat l_ru_w = cp.loss_ru_without_secur(n, seed);

dy = std::chrono::high_resolution_clock::now();
dif = dy - dx;
std::cout << dif.count() << " seconds" << std::endl;
*/
