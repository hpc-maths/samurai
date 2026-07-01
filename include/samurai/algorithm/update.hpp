// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Umbrella header for the multiresolution ghost- and field-update routines. It
// aggregates the focused headers below to expose the full update API. Include
// this header (update.hpp) to get the complete set of update_* entry points.
// API:
//
//   update_basic_ghost.hpp  update_ghost
//   update_outer_ghost.hpp  project_bc / predict_bc / project_corner_below /
//                           update_outer_ghosts
//   update_subdomain.hpp    outer_subdomain_corner / update_tag_subdomains
//   update_periodic.hpp     update_ghost_periodic / update_tag_periodic
//   update_fields.hpp       update_fields / update_field
//   update_ghost_mr.hpp     update_ghost_mr + its aggregated MPI implementation

#include "update_basic_ghost.hpp"
#include "update_fields.hpp"
#include "update_ghost_mr.hpp"
#include "update_outer_ghost.hpp"
#include "update_periodic.hpp"
#include "update_subdomain.hpp"
