cmake_minimum_required(VERSION 3.15)


macro(add_CR_test _TEST_NAME)
    add_executable(${_TEST_NAME}_tests ${_TEST_NAME}_tests.cpp)
    target_link_libraries(${_TEST_NAME}_tests PRIVATE
        GTest::GTest
        CreditRisk
        )
    add_test(NAME CR_${_TEST_NAME}_tests COMMAND ${_TEST_NAME}_tests)
    # Group under the "tests" project folder in IDEs such as Visual Studio.
    set_property(TARGET ${_TEST_NAME}_tests PROPERTY FOLDER "tests")
endmacro()

add_CR_test(reading)

add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:reading_tests>/data/from_ECT)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_ECT/CORREL.txt $<TARGET_FILE_DIR:reading_tests>/data/from_ECT/CORREL.txt)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_ECT/DATA_CRED_RETAIL.txt $<TARGET_FILE_DIR:reading_tests>/data/from_ECT/DATA_CRED_RETAIL.txt)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_ECT/DATA_CRED_WHOL.txt $<TARGET_FILE_DIR:reading_tests>/data/from_ECT/DATA_CRED_WHOL.txt)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_ECT/PTRANS.txt $<TARGET_FILE_DIR:reading_tests>/data/from_ECT/PTRANS.txt)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_ECT/SPREADS.txt $<TARGET_FILE_DIR:reading_tests>/data/from_ECT/SPREADS.txt)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_ECT/SPV_TRANCHES.csv $<TARGET_FILE_DIR:reading_tests>/data/from_ECT/SPV_TRANCHES.csv)


add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:reading_tests>/data/from_json)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_json/portfolio_test.json $<TARGET_FILE_DIR:reading_tests>/data/from_json/portfolio_test.json)

add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:reading_tests>/data/from_csv)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_csv/cor.csv $<TARGET_FILE_DIR:reading_tests>/data/from_csv/cor.csv)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_csv/counter.csv $<TARGET_FILE_DIR:reading_tests>/data/from_csv/counter.csv)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_csv/Fund.csv $<TARGET_FILE_DIR:reading_tests>/data/from_csv/Fund.csv)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_csv/Portfolio.csv $<TARGET_FILE_DIR:reading_tests>/data/from_csv/Portfolio.csv)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_csv/spreads.csv $<TARGET_FILE_DIR:reading_tests>/data/from_csv/spreads.csv)
add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_csv/transition.csv $<TARGET_FILE_DIR:reading_tests>/data/from_csv/transition.csv)

IF (USE_OPENXLSX)
    add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:reading_tests>/data/from_xslx)
    add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_xlsx/SCIB_CM_12_2020_v1.xlsx $<TARGET_FILE_DIR:reading_tests>/data/from_xlsx/SCIB_CM_12_2020_v1.xlsx)
    add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_xlsx/spreads.csv $<TARGET_FILE_DIR:reading_tests>/data/from_xlsx/spreads.csv)
    add_custom_command(TARGET reading_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data/from_xlsx/transition.csv $<TARGET_FILE_DIR:reading_tests>/data/from_xlsx/transition.csv)
ENDIF()
