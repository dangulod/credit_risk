#include <gtest/gtest.h>
#include <CreditRisk/credit_portfolio.h>

TEST(Constructor, Element)
{
    CreditRisk::Element element(123456,
                                1,
                                100,
                                0.2,
                                0.2,
                                0.3,
                                0.3,
                                0.15,
                                1,
                                CreditRisk::Element::Treatment::Retail,
                                {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});


    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             -0.2,
                                             0.2,
                                             0.3,
                                             0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             1.02,
                                             0.2,
                                             0.3,
                                             0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             0.2,
                                             -0.2,
                                             0.3,
                                             0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             0.2,
                                             1.02,
                                             0.3,
                                             0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             0.2,
                                             0.2,
                                             -0.3,
                                             0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             0.2,
                                             0.2,
                                             1.3,
                                             0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             0.2,
                                             0.2,
                                             0.3,
                                             -0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             0.2,
                                             0.2,
                                             0.3,
                                             1.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             0.2,
                                             0.2,
                                             0.3,
                                             0.3,
                                             -1.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             -0.2,
                                             0.2,
                                             0.3,
                                             0.3,
                                             1.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
    EXPECT_THROW(CreditRisk::Element element(123456,
                                             1,
                                             100,
                                             -0.2,
                                             0.2,
                                             0.3,
                                             0.3,
                                             0.15,
                                             1,
                                             CreditRisk::Element::Treatment::Retail,
                                             {10000, {0.99, 0.1, 0.1, 0.1, 0.1}});,
                 std::invalid_argument);
}


TEST(MonteCarlo, Element)
{
    CreditRisk::Transition transition = CreditRisk::Transition::from_csv("data/from_csv/transition.csv");
    CreditRisk::Spread spread = CreditRisk::Spread::from_csv("data/from_csv/spreads.csv");

    CreditRisk::Element retail(123456,
                               1,
                               100,
                               0.2,
                               0.2,
                               0.3,
                               0.3,
                               0.15,
                               1,
                               CreditRisk::Element::Treatment::Retail,
                               {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});

    CreditRisk::Element wholesale(123456,
                                  1,
                                  100,
                                  0.2,
                                  0.2,
                                  0.3,
                                  0.3,
                                  0.15,
                                  3,
                                  CreditRisk::Element::Treatment::Wholesale,
                                  {10000, {0.1, 0.1, 0.1, 0.1, 0.1}});

    wholesale.setMigration(&transition, &spread, 0.01);

    EXPECT_EQ(retail.el(), 100 * 0.2 * 0.3);
    EXPECT_EQ(retail.pd_c(-2), 0.29190800538692729393);
    EXPECT_EQ(retail.pd_c(-1), 0.24210862911746902637);
    EXPECT_EQ(retail.pd_c(1), 0.15793795125526247092);
    EXPECT_EQ(retail.pd_c(2), 0.12410965098694100306);

    EXPECT_EQ(retail.pd_c(0.5, -2), 0.16820127328020784141);
    EXPECT_EQ(retail.pd_c(0.5, -1), 0.13285117905005683347);
    EXPECT_EQ(retail.pd_c(0.5, 1), 0.07832230945888923879);
    EXPECT_EQ(retail.pd_c(0.5, 2), 0.05842146674082777241);

    EXPECT_EQ(wholesale.getT(-3), 0.00605354695124538621);
    EXPECT_EQ(retail.getT(-3), 0);

    EXPECT_EQ(wholesale.loss(-2), 30);

    EXPECT_EQ(wholesale.loss(0.5, -2.), 30);
    EXPECT_EQ(retail.loss(0.5, -2.), 5.04603819840623479820);

    EXPECT_EQ(wholesale.p_states_c(-2, true).at(0), 0.36344658346807290350);
    EXPECT_EQ(wholesale.l_states(true).at(1), 6.50930603980670596798);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
