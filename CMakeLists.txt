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

IF (WIN32)
    link_directories("/lib")
ELSE()
    link_directories("/usr/local/lib")
ENDIF()

IF (WIN32)
    include_directories("include/CreditRisk"
        "include"
        "/armadillo/include"
        "/boost"
        "/nlopt"
        "/OpenXLSX")
ELSE()
    include_directories(
        "include/CreditRisk"
        "include")
ENDIF()

add_library(CreditRisk STATIC ${MODEL_SOURCES})

add_executable(${PROJECT_NAME} main.cpp)

IF (WIN32)
    set(LIBS pthread openblas nlopt OpenXLSX-static)
ELSE()
    set(LIBS pthread armadillo openblas nlopt OpenXLSX)
ENDIF()

target_link_libraries(CreditRisk ${LIBS})

IF (WIN32)
    target_compile_definitions(CreditRisk PRIVATE OPENXLSX_STATIC_DEFINE)
ENDIF()

target_link_libraries(${PROJECT_NAME} CreditRisk)



