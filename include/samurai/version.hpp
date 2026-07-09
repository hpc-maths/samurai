// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <fmt/color.h>
#include <fmt/format.h>

#include "samurai_config.hpp" // samurai::disable_color

// --------------------------------------------------------------------------
// samurai's own version, injected by CMake through target_compile_definitions
// (see the top-level CMakeLists.txt). Fallback to 0.0.0 so that the header
// stays usable even when built outside of the CMake project.
// --------------------------------------------------------------------------
#ifndef SAMURAI_VERSION_MAJOR
#define SAMURAI_VERSION_MAJOR 0
#endif
#ifndef SAMURAI_VERSION_MINOR
#define SAMURAI_VERSION_MINOR 0
#endif
#ifndef SAMURAI_VERSION_PATCH
#define SAMURAI_VERSION_PATCH 0
#endif

// --------------------------------------------------------------------------
// Best-effort detection of the dependencies' version headers. Each header is
// small and self-contained; `__has_include` (guaranteed since C++17) keeps the
// detection fully optional and portable: a dependency that is not part of the
// build simply does not appear in the report.
// --------------------------------------------------------------------------
#if __has_include(<fmt/base.h>)
#include <fmt/base.h> // FMT_VERSION
#elif __has_include(<fmt/core.h>)
#include <fmt/core.h>
#endif
#if __has_include(<CLI/Version.hpp>)
#include <CLI/Version.hpp> // CLI11_VERSION
#endif
#if __has_include(<pugixml.hpp>)
#include <pugixml.hpp> // PUGIXML_VERSION
#endif
#if __has_include(<highfive/H5Version.hpp>)
#include <highfive/H5Version.hpp> // HIGHFIVE_VERSION_STRING
#endif
#if __has_include(<H5public.h>)
#include <H5public.h> // H5_VERS_*
#endif
#if __has_include(<xtensor/core/xtensor_config.hpp>)
#include <xtensor/core/xtensor_config.hpp> // XTENSOR_VERSION_*
#endif
#if __has_include(<xtl/xtl_config.hpp>)
#include <xtl/xtl_config.hpp> // XTL_VERSION_*
#endif
#if __has_include(<nlohmann/detail/abi_macros.hpp>)
#include <nlohmann/detail/abi_macros.hpp> // NLOHMANN_JSON_VERSION_*
#endif
#if __has_include(<boost/version.hpp>)
#include <boost/version.hpp> // BOOST_VERSION / BOOST_LIB_VERSION
#endif

#ifdef SAMURAI_WITH_PETSC
#include <petscversion.h> // PETSC_VERSION_*
#endif
#ifdef SAMURAI_WITH_MPI
#include <mpi.h> // MPI_VERSION / MPI_SUBVERSION
#endif
#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3) || defined(SAMURAI_FLUX_CONTAINER_EIGEN3) || defined(SAMURAI_STATIC_MAT_CONTAINER_EIGEN3)
#if __has_include(<Eigen/Core>)
#include <Eigen/Core> // EIGEN_WORLD_VERSION / EIGEN_MAJOR_VERSION / EIGEN_MINOR_VERSION
#endif
#endif

namespace samurai
{
    /// A single (name, version) entry of the dependency report.
    struct dependency_info
    {
        std::string name;
        std::string version;
    };

    /// samurai's version, e.g. "0.31.2".
    inline std::string version()
    {
        return fmt::format("{}.{}.{}", SAMURAI_VERSION_MAJOR, SAMURAI_VERSION_MINOR, SAMURAI_VERSION_PATCH);
    }

    /// The compiler used to build the current translation unit, e.g. "Clang 18.1.0".
    inline std::string compiler_info()
    {
#if defined(__clang__)
        return fmt::format("Clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
        return fmt::format("GCC {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
        return fmt::format("MSVC {}", _MSC_VER);
#else
        return "unknown";
#endif
    }

    /// The version of every dependency detected at compile time.
    inline std::vector<dependency_info> dependencies_info()
    {
        std::vector<dependency_info> deps;

#ifdef FMT_VERSION
        deps.push_back({"fmt", fmt::format("{}.{}.{}", FMT_VERSION / 10000, (FMT_VERSION % 10000) / 100, FMT_VERSION % 100)});
#endif
#ifdef CLI11_VERSION
        deps.push_back({"CLI11", CLI11_VERSION});
#endif
#ifdef PUGIXML_VERSION
        deps.push_back({"pugixml", fmt::format("{}.{}", PUGIXML_VERSION / 1000, (PUGIXML_VERSION % 1000) / 10)});
#endif
#ifdef HIGHFIVE_VERSION_STRING
        deps.push_back({"HighFive", HIGHFIVE_VERSION_STRING});
#endif
#ifdef H5_VERS_MAJOR
        deps.push_back({"HDF5", fmt::format("{}.{}.{}", H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE)});
#endif
#ifdef XTENSOR_VERSION_MAJOR
        deps.push_back({"xtensor", fmt::format("{}.{}.{}", XTENSOR_VERSION_MAJOR, XTENSOR_VERSION_MINOR, XTENSOR_VERSION_PATCH)});
#endif
#ifdef XTL_VERSION_MAJOR
        deps.push_back({"xtl", fmt::format("{}.{}.{}", XTL_VERSION_MAJOR, XTL_VERSION_MINOR, XTL_VERSION_PATCH)});
#endif
#ifdef EIGEN_WORLD_VERSION
        deps.push_back({"Eigen", fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION)});
#endif
        // Feature-gated dependencies: reported only when the corresponding build
        // option is enabled, so the list mirrors what this build actually uses
        // (and not merely what happens to be installed in the environment).
#if (defined(SAMURAI_WITH_PARMETIS) || defined(SAMURAI_WITH_PTSCOTCH)) && defined(NLOHMANN_JSON_VERSION_MAJOR)
        deps.push_back(
            {"nlohmann_json", fmt::format("{}.{}.{}", NLOHMANN_JSON_VERSION_MAJOR, NLOHMANN_JSON_VERSION_MINOR, NLOHMANN_JSON_VERSION_PATCH)});
#endif
#if defined(SAMURAI_WITH_MPI) && defined(BOOST_VERSION)
        deps.push_back({"Boost", fmt::format("{}.{}.{}", BOOST_VERSION / 100000, (BOOST_VERSION / 100) % 1000, BOOST_VERSION % 100)});
#endif
#if defined(SAMURAI_WITH_PETSC) && defined(PETSC_VERSION_MAJOR)
        deps.push_back({"PETSc", fmt::format("{}.{}.{}", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR)});
#endif
#if defined(SAMURAI_WITH_MPI) && defined(MPI_VERSION)
        deps.push_back({"MPI (standard)", fmt::format("{}.{}", MPI_VERSION, MPI_SUBVERSION)});
#endif

        return deps;
    }

    /// The build-time configuration (enabled features, selected containers, ...).
    inline std::vector<dependency_info> build_info()
    {
        std::vector<dependency_info> info;

        const auto on_off = [](bool enabled)
        {
            return enabled ? "ON" : "OFF";
        };

#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
        info.push_back({"field container", "eigen3"});
#else
        info.push_back({"field container", "xtensor"});
#endif

        bool with_mpi = false;
#ifdef SAMURAI_WITH_MPI
        with_mpi = true;
#endif
        info.push_back({"MPI", on_off(with_mpi)});

        bool with_petsc = false;
#ifdef SAMURAI_WITH_PETSC
        with_petsc = true;
#endif
        info.push_back({"PETSc", on_off(with_petsc)});

        bool with_openmp = false;
#ifdef SAMURAI_WITH_OPENMP
        with_openmp = true;
#endif
        info.push_back({"OpenMP", on_off(with_openmp)});

        bool with_parmetis = false;
#ifdef SAMURAI_WITH_PARMETIS
        with_parmetis = true;
#endif
        info.push_back({"ParMETIS", on_off(with_parmetis)});

        bool with_ptscotch = false;
#ifdef SAMURAI_WITH_PTSCOTCH
        with_ptscotch = true;
#endif
        info.push_back({"PT-Scotch", on_off(with_ptscotch)});

        return info;
    }

    /// Pretty-print samurai's version, its dependencies and the build configuration.
    inline void print_info(std::ostream& os = std::cout)
    {
        const auto deps  = dependencies_info();
        const auto build = build_info();

        // Column width computed from the widest label so the two colons align.
        std::size_t width = std::string("samurai").size();
        for (const auto& d : deps)
        {
            width = std::max(width, d.name.size());
        }
        for (const auto& b : build)
        {
            width = std::max(width, b.name.size());
        }

        const auto title = disable_color ? fmt::text_style() : fmt::emphasis::bold;

        os << fmt::format(title, "samurai {}\n", version());
        os << "\n";
        os << fmt::format(title, "Dependencies\n");
        for (const auto& d : deps)
        {
            os << fmt::format("  {:<{}} : {}\n", d.name, width, d.version);
        }
        os << "\n";
        os << fmt::format(title, "Build configuration\n");
        for (const auto& b : build)
        {
            os << fmt::format("  {:<{}} : {}\n", b.name, width, b.version);
        }
        os << fmt::format("  {:<{}} : {} (C++{})\n", "compiler", width, compiler_info(), __cplusplus / 100 % 100);
    }
}
