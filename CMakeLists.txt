cmake_minimum_required(VERSION 3.20)

project(
    mantix
    VERSION 0.1.0
    DESCRIPTION "Low-level arbitrary-precision binary floating-point library"
    LANGUAGES C
)

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Mantix requires Clang (found ${CMAKE_C_COMPILER_ID}).")
endif()
if(CMAKE_C_COMPILER_VERSION VERSION_LESS 17)
    message(FATAL_ERROR
        "Mantix requires Clang >= 17 (found ${CMAKE_C_COMPILER_VERSION}).")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CTest)

option(MANTIX_BUILD_EXAMPLES "Build Mantix examples" ON)
option(MANTIX_BUILD_BENCHMARKS "Build Mantix microbenchmarks" OFF)
option(MANTIX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(MANTIX_SOURCES
    src/mantix.c
    src/limb.c
)

add_library(mantix
    ${MANTIX_SOURCES}
)
add_library(mantix::mantix ALIAS mantix)

target_compile_features(mantix PUBLIC c_std_17)
target_link_libraries(mantix PUBLIC m)
target_include_directories(
    mantix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(
        mantix
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
    )

    if(MANTIX_ENABLE_SANITIZERS)
        target_compile_options(mantix PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(mantix PUBLIC -fsanitize=address,undefined)
    endif()
endif()

if(MANTIX_BUILD_EXAMPLES)
    add_executable(mantix_example examples/basic.c)
    target_link_libraries(mantix_example PRIVATE mantix::mantix)
endif()

if(BUILD_TESTING)
    add_executable(mantix_tests tests/test_mantix.c)
    target_link_libraries(mantix_tests PRIVATE mantix::mantix)
    add_test(NAME mantix.unit COMMAND mantix_tests)

    add_executable(mantix_limb_tests tests/test_limb.c)
    target_link_libraries(mantix_limb_tests PRIVATE mantix::mantix)
    add_test(NAME mantix.limb COMMAND mantix_limb_tests)
    if(MANTIX_ENABLE_SANITIZERS)
        # LeakSanitizer cannot attach under common container ptrace policies.
        # ASan heap/bounds checks and UBSan remain enabled.
        set_tests_properties(mantix.unit mantix.limb PROPERTIES
            ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endif()
endif()

if(MANTIX_BUILD_BENCHMARKS)
    add_executable(mantix_bench benchmarks/bench.c)
    target_link_libraries(mantix_bench PRIVATE mantix::mantix)
    target_compile_definitions(mantix_bench PRIVATE _POSIX_C_SOURCE=200809L)
    target_compile_options(mantix_bench PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GMP QUIET IMPORTED_TARGET gmp)
        pkg_check_modules(MPFR QUIET IMPORTED_TARGET mpfr)
    endif()
    if(TARGET PkgConfig::GMP)
        target_link_libraries(mantix_bench PRIVATE PkgConfig::GMP)
        target_compile_definitions(mantix_bench PRIVATE MANTIX_HAVE_GMP=1)
        message(STATUS "Mantix benchmarks: GMP comparison enabled")
    else()
        message(STATUS "Mantix benchmarks: GMP comparison disabled (not found)")
    endif()
    if(TARGET PkgConfig::MPFR)
        target_link_libraries(mantix_bench PRIVATE PkgConfig::MPFR)
        target_compile_definitions(mantix_bench PRIVATE MANTIX_HAVE_MPFR=1)
        message(STATUS "Mantix benchmarks: MPFR comparison enabled")
    else()
        message(STATUS "Mantix benchmarks: MPFR comparison disabled (not found)")
    endif()
endif()

install(
    TARGETS mantix
    EXPORT mantixTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT mantixTargets
    FILE mantixTargets.cmake
    NAMESPACE mantix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mantix
)
