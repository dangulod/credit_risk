cmake_minimum_required(VERSION 2.8)

project(Credit_Risk CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

macro(CR_option OPTION_NAME OPTION_TEXT OPTION_DEFAULT)
    option(${OPTION_NAME} ${OPTION_TEXT} ${OPTION_DEFAULT})
    if(DEFINED ENV{${OPTION_NAME}})
        # Allow setting the option through an environment variable
        set(${OPTION_NAME} $ENV{${OPTION_NAME}})
    endif()
    if(${OPTION_NAME})
        add_definitions(-D${OPTION_NAME})
    endif()
    message(STATUS "  ${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

#==============================================================================
# version
set (CR_MAJOR_VERSION "1")
set (CR_MINOR_VERSION "0")
set (CR_BUGFIX_VERSION "0")
set (CR_VERSION_STRING ${CR_MAJOR_VERSION}.${CR_MINOR_VERSION}.${CR_BUGFIX_VERSION})
message("=======================================================")
message ("##### Credit Risk version ${CR_VERSION_STRING} #####")
message("=======================================================")
message(STATUS "Checking dependencies")
message("=======================================================")
find_package(BLAS REQUIRED)
find_package(NLopt REQUIRED)
find_package(Armadillo REQUIRED)

message(STATUS "BLAS found at " ${BLAS_LIBRARIES})
message(STATUS "Armadilo found version " ${ARMADILLO_VERSION_STRING})
message(STATUS "NLopt found version " ${NLOPT_VERSION})


message(STATUS "C++ Requests CMake Options")
message("=======================================================")
CR_option(USE_OPENXLSX
    "If ON, this project will be compiled with OPENXLSX support" OFF)
CR_option(BUILD_CR_TESTS "Set to ON to build Credit Risk tests." ON)
CR_option(USE_SYSTEM_GTEST
    "If ON, this project will look in the system paths for an installed gtest library" ON)
message("=======================================================")

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
ENDIF()

IF (WIN32)
    include_directories(
        "include/CreditRisk"
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

# GTest configuration
if(BUILD_CR_TESTS)
    if(USE_SYSTEM_GTEST)
        find_package(GTest)
    endif()
    if(NOT USE_SYSTEM_GTEST OR NOT GTEST_FOUND)
        include(FetchContent)
        message(STATUS "Not using system gtest, using built-in googletest project instead.")
        if(MSVC)
            # By default, GTest compiles on Windows in CRT static linkage mode. We use this
            # variable to force it into using the CRT in dynamic linkage (DLL), just as CPR
            # does.
            set(gtest_force_shared_crt ON CACHE BOOL "Force gtest to use the shared c runtime")
        endif()
        FetchContent_Declare(googletest
                                    GIT_REPOSITORY https://github.com/google/googletest.git
                                    GIT_TAG release-1.10.0)
        FetchContent_MakeAvailable(googletest)

        add_library(gtest_int INTERFACE)
        target_link_libraries(gtest_int INTERFACE gtest)
        target_include_directories(gtest_int INTERFACE ${googletest_SOURCE_DIR}/include)

        add_library(GTest::GTest ALIAS gtest_int)

        # Group under the "tests/gtest" project folder in IDEs such as Visual Studio.
    set_property(TARGET gtest PROPERTY FOLDER "tests/gtest")
    set_property(TARGET gtest_main PROPERTY FOLDER "tests/gtest")
    endif()
endif()


add_library(CreditRisk STATIC ${MODEL_SOURCES})

add_executable(${PROJECT_NAME} main.cpp)

list(APPEND LIBS pthread openblas NLopt::nlopt)

IF (NOT WIN32)
    list(APPEND LIBS armadillo)
ENDIF()

IF (USE_OPENXLSX)
    IF (WIN32)
        list(APPEND LIBS OpenXLSX-static)
        ELSE ()
            list(APPEND LIBS OpenXLSX)
    ENDIF()
ENDIF()

target_link_libraries(CreditRisk ${LIBS})

IF (USE_OPENXLSX)
    IF (WIN32)
        target_compile_definitions(CreditRisk PRIVATE OPENXLSX_STATIC_DEFINE)
    ENDIF()
ENDIF()

target_link_libraries(${PROJECT_NAME} CreditRisk)

if(BUILD_CR_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()


