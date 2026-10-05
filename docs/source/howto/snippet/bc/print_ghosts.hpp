#pragma once

#include <iostream>
#include <string>

#include <samurai/algorithm.hpp>
#include <samurai/stencil.hpp>

// Print the first layer of ghosts on one side of the unit square,
// with the center of each ghost and its value.
template <class Field>
void print_ghosts(const std::string& name,
                  const Field& u,
                  const samurai::DirectionVector<2>& side)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    std::cout << name << " ghosts:\n";
    const auto& mesh = u.mesh();
    samurai::for_each_cell(mesh[mesh_id_t::reference],
                           [&](const auto& cell)
                           {
                               const auto x   = cell.center();
                               const double h = cell.length;
                               bool on_side   = true;
                               for (std::size_t d = 0; d < 2; ++d)
                               {
                                   // Range of the ghost centers along d
                                   double lo = 0;
                                   double hi = 1;
                                   if (side[d] < 0)
                                   {
                                       lo = -h;
                                       hi = 0;
                                   }
                                   else if (side[d] > 0)
                                   {
                                       lo = 1;
                                       hi = 1 + h;
                                   }
                                   on_side = on_side && x[d] > lo && x[d] < hi;
                               }
                               if (on_side)
                               {
                                   std::cout << "  (" << x[0] << ", " << x[1]
                                             << "): " << u[cell] << "\n";
                               }
                           });
}
