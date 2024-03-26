// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once
#include "../numeric/error.hpp"

#include "fv/cell_based/cell_based_scheme__nonlin.hpp"
#include "fv/cell_based/explicit_cell_based_scheme.hpp"
#include "fv/explicit_operator_sum.hpp"
#include "fv/flux_based/explicit_flux_based_scheme__lin_het.hpp"
#include "fv/flux_based/explicit_flux_based_scheme__lin_hom.hpp"
#include "fv/flux_based/explicit_flux_based_scheme__nonlin.hpp"
#include "fv/scheme_operators.hpp"

#include "fv/operators/convection_lin.hpp"
#include "fv/operators/convection_nonlin.hpp"
#include "fv/operators/diffusion.hpp"
#include "fv/operators/diffusion_old.hpp"
#include "fv/operators/divergence.hpp"
#include "fv/operators/flux_divergence.hpp"
#include "fv/operators/gradient.hpp"
#include "fv/operators/identity.hpp"
#include "fv/operators/zero_operator.hpp"
