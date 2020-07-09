cmake_minimum_required(VERSION 3.5)

project(Credit_Risk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB MODEL_SOURCES
    src/Elements/*.cpp
    src/Utils/*.cpp
    src/Equation/*.cpp
    src/FactorCor/*.cpp
    src/Portfolio/*.cpp
    src/Fund/*.cpp
    src/Credit_portfolio/*.cpp
    src/Integrator/*.cpp
    src/Migration/*.cpp
    )

include_directories(
    "include/CreditRisk"
    "include"
    )

add_library(CreditRisk ${MODEL_SOURCES})

add_executable(${PROJECT_NAME} main.cpp)

set(LIBS pthread armadillo openblas nlopt)

target_link_libraries(CreditRisk ${LIBS})

target_link_libraries(${PROJECT_NAME} CreditRisk)
