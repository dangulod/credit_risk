# Credit_Risk

```cpp
#include <iostream>
#include <armadillo>
#include <credit_portfolio.h>
#include <chrono>
#include "ThreadPool/threadPool.hpp"
#include <spread.h>
#include <transition.h>

int main()
{
    TP::ThreadPool pool(8);
    pool.init();

    pt::ptree pt;
    pt::read_json("/opt/share/data/optim/optim.json", pt);

    CreditRisk::Credit_portfolio p = CreditRisk::Credit_portfolio::from_ptree(pt);

    p.m_rand(1e2, 987654321, &pool); // aleatorios
    p.getIdio(1e2, &pool); // idiosincraticos
    p.getCWIs(1e2, 987654321, &pool); // CWIs

    /*
     * Moncecarlo with / without securitization effect
     * and with migration (true) or without (false)
     * migration.
     *
     * 1e2: number of simulations
     * 987654321: seed
     */

    p.loss(1e2, 987654321, &pool, true);
    p.loss(1e2, 987654321, &pool, false);

    p.loss_without_secur(1e2, 987654321, &pool, false);
    p.loss_without_secur(1e2, 987654321, &pool, false);

    p.loss_portfolio(1e2, 987654321, &pool, true);
    p.loss_portfolio(1e2, 987654321, &pool, false);

    p.loss_portfolio_without_secur(1e2, 987654321, &pool, true);
    p.loss_portfolio_without_secur(1e2, 987654321, &pool, false);

    p.loss_ru(1e2, 987654321, &pool, true);
    p.loss_ru(1e2, 987654321, &pool, false);

    p.loss_ru_without_secur(1e2, 987654321, &pool, true);
    p.loss_ru_without_secur(1e2, 987654321, &pool, false);

    p.margin_loss(1e2, 987654321, &pool, true);
    p.margin_loss(1e2, 987654321, &pool, false);

    p.margin_loss_without_secur(1e2, 987654321, &pool, false);
    p.margin_loss_without_secur(1e2, 987654321, &pool, false);

    /*
     * Saddlepoint
     *
     */

    // Generar puntos de la integral gki (100 puntos) ghi (7 puntos

    CreditRisk::Integrator::PointsAndWeigths points = CreditRisk::Integrator::gki();

    pool.init();

    auto ns = p.get_Ns();
    auto pd_c = p.pd_c(points, &pool, true); // true/false migration
    auto std_eadxlgds = p.get_std_states(true); // true/false migration

    double q = p.quantile(0.9995, &ns, std_eadxlgds.get(), pd_c.get(), &points, &pool);

    std::cout << "Loss at 99.95%: " <<  q * p.T_EADxLGD << std::endl;

    arma::vec contrib = p.getContrib(q, &ns, std_eadxlgds.get(), pd_c.get(), &points, &pool) * p.T_EADxLGD;
    arma::vec contrib_without = p.getContrib_without_secur(q, &ns, std_eadxlgds.get(), pd_c.get(), &points, &pool)* p.T_EADxLGD;

    double eva = p.EVA(std_eadxlgds.get(), contrib);

    pool.shutdown();
    
    return 0;
}
```
