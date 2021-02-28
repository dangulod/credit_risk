#include <gtest/gtest.h>
#include <CreditRisk/credit_portfolio.h>

//static CreditRisk::Credit_portfolio p_json = CreditRisk::Credit_portfolio::from_ptree()

TEST(From_ect, PortfolioTest) {
    CreditRisk::Credit_portfolio p = CreditRisk::Credit_portfolio::from_ect(
                "data/from_ECT/DATA_CRED_WHOL.txt",
                "data/from_ECT/DATA_CRED_RETAIL.txt",
                "data/from_ECT/CORREL.txt",
                "data/from_ECT/SPV_TRANCHES.csv",
                "data/from_ECT/PTRANS.txt",
                "data/from_ECT/SPREADS.txt"
                );

    EXPECT_EQ(p.getN(), 5798);
    EXPECT_EQ(p.size(), 7);
}

#ifdef USE_OPENXLSX
TEST(From_xlsx, PortfolioTest) {
    CreditRisk::Credit_portfolio p = CreditRisk::Credit_portfolio::from_xlsx_ps("test/from_xlsx/SCIB_CM_12_2020_v1.xlsx",
                                                                                "test/from_xlsx/transition.csv",
                                                                                "test/from_xlsx/spreads.csv");

    EXPECT_EQ(p.getN(), 5798);
    EXPECT_EQ(p.size(), 7);
}
#endif

TEST(From_json, PortfolioTest) {
    pt::ptree pt;
    pt::read_json("data/from_json/portfolio_test.json", pt);
    CreditRisk::Credit_portfolio p = CreditRisk::Credit_portfolio::from_ptree(pt);
    EXPECT_EQ(p.getN(), 5798);
    EXPECT_EQ(p.size(), 7);
}

TEST(From_csv, PortfolioTest) {
    CreditRisk::Credit_portfolio p = CreditRisk::Credit_portfolio::from_csv(
                "data/from_csv/Portfolio.csv",
                "data/from_csv/Fund.csv",
                "data/from_csv/counter.csv",
                "data/from_csv/cor.csv",
                17,
                "data/from_csv/transition.csv",
                "data/from_csv/spreads.csv"
                );

    EXPECT_EQ(p.getN(), 13624);
    EXPECT_EQ(p.size(), 3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

