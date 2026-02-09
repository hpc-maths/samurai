// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once
#include "../numeric/error.hpp"
#include "../samurai.hpp"

#include "fv/cell_based/cell_based_scheme__nonlin.hpp"
#include "fv/cell_based/explicit_cell_based_scheme.hpp"
#include "fv/explicit_operator_sum.hpp"
#include "fv/flux_based/explicit_flux_based_scheme__lin_het.hpp"
#include "fv/flux_based/explicit_flux_based_scheme__lin_hom.hpp"
#include "fv/flux_based/explicit_flux_based_scheme__nonlin.hpp"
#include "fv/scheme_operators.hpp"

#include "fv/operators/buoyancy.hpp"
#include "fv/operators/convection_lin.hpp"
#include "fv/operators/convection_nonlin.hpp"
#include "fv/operators/diffusion.hpp"
#include "fv/operators/diffusion_cell_based.hpp"
#include "fv/operators/divergence.hpp"
#include "fv/operators/flux_divergence.hpp"
#include "fv/operators/gradient.hpp"
#include "fv/operators/identity.hpp"
#include "fv/operators/zero_operator.hpp"

#ifdef SAMURAI_WITH_PETSC
#include "../petsc/manual_assembly.hpp"
#include "../petsc/solver_helpers.hpp"
#endif
