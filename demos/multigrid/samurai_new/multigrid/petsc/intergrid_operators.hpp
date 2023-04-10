#pragma once
#include "../LevelCtx.hpp"
#include <samurai/numeric/prediction.hpp>
#include <samurai/numeric/projection.hpp>

namespace samurai_new
{
    namespace petsc
    {
        namespace multigrid
        {

            template <class Mesh>
            void prolong(const Mesh& coarse_mesh, const Mesh& fine_mesh, const double* carray, double* farray, int prediction_order)
            {
                using mesh_id_t                  = typename Mesh::mesh_id_t;
                static constexpr std::size_t dim = Mesh::dim;

                auto& cm = coarse_mesh[mesh_id_t::cells];
                auto& fm = fine_mesh[mesh_id_t::cells];

                for (std::size_t level = fine_mesh.min_level(); level <= fine_mesh.max_level(); ++level)
                {
                    // Shared cells between coarse and fine mesh
                    auto shared = samurai::intersection(cm[level], fm[level]);
                    shared(
                        [&](auto& i, auto&)
                        {
                            auto index_f = fine_mesh.get_index(level, i.start);
                            auto index_c = coarse_mesh.get_index(level, i.start);
                            for (std::size_t ii = 0; ii < i.size(); ++ii)
                            {
                                farray[index_f + ii] = carray[index_c + ii];
                            }
                        });

                    // Coarse cells to fine cells: prediction
                    auto others = samurai::intersection(cm[level - 1], fm[level]).on(level - 1);
                    others(
                        [&](auto& i, const auto& index)
                        {
                            // static_assert(dim == 1 || dim == 2, "prolong()
                            // not implemented for this dimension");
                            assert((dim == 1 || dim == 2) && "prolong() not implemented for this dimension");
                            if constexpr (dim == 1)
                            {
                                auto i_f = fine_mesh.get_index(level, (2 * i).start);
                                auto i_c = coarse_mesh.get_index(level - 1, i.start);
                                if (prediction_order == 0)
                                {
                                    for (std::size_t ii = 0; ii < i.size(); ++ii)
                                    {
                                        // Prediction 1D (order 0)
                                        farray[i_f + 2 * ii]     = carray[i_c + ii];
                                        farray[i_f + 2 * ii + 1] = carray[i_c + ii];
                                    }
                                }
                                else if (prediction_order == 1)
                                {
                                    for (std::size_t ii = 0; ii < i.size(); ++ii)
                                    {
                                        // Prediction 1D (order 1)
                                        farray[i_f + 2 * ii] = carray[i_c + ii] - 1. / 8 * (carray[i_c + ii + 1] - carray[i_c + ii - 1]);
                                        farray[i_f + 2 * ii + 1] = carray[i_c + ii] + 1. / 8 * (carray[i_c + ii + 1] - carray[i_c + ii - 1]);
                                    }
                                }
                            }
                            else if constexpr (dim == 2)
                            {
                                auto j = index[0];

                                auto i_c_j = coarse_mesh.get_index(level - 1, i.start, j);

                                auto i_f_2j   = fine_mesh.get_index(level, (2 * i).start, 2 * j);
                                auto i_f_2jp1 = fine_mesh.get_index(level, (2 * i).start, 2 * j + 1);

                                if (prediction_order == 0)
                                {
                                    for (std::size_t ii = 0; ii < i.size(); ++ii)
                                    {
                                        // Prediction 2D (order 0)
                                        farray[i_f_2j + 2 * ii]       = carray[i_c_j + ii];
                                        farray[i_f_2j + 2 * ii + 1]   = carray[i_c_j + ii];
                                        farray[i_f_2jp1 + 2 * ii]     = carray[i_c_j + ii];
                                        farray[i_f_2jp1 + 2 * ii + 1] = carray[i_c_j + ii];
                                    }
                                }
                                else if (prediction_order == 1)
                                {
                                    auto i_c_jm1 = coarse_mesh.get_index(level - 1, i.start, j - 1);
                                    auto i_c_jp1 = coarse_mesh.get_index(level - 1, i.start, j + 1);

                                    for (std::size_t ii = 0; ii < i.size(); ++ii)
                                    {
                                        // Prediction 2D (order 1)
                                        double qs_i  = -1. / 8 * (carray[i_c_j + ii + 1] - carray[i_c_j + ii - 1]);
                                        double qs_j  = -1. / 8 * (carray[i_c_jp1 + ii] - carray[i_c_jm1 + ii]);
                                        double qs_ij = 1. / 64
                                                     * (carray[i_c_jp1 + ii + 1] - carray[i_c_jp1 + ii - 1] - carray[i_c_jm1 + ii + 1]
                                                        + carray[i_c_jm1 + ii - 1]);
                                        farray[i_f_2j + 2 * ii]       = carray[i_c_j + ii] + qs_i + qs_j - qs_ij;
                                        farray[i_f_2j + 2 * ii + 1]   = carray[i_c_j + ii] - qs_i + qs_j + qs_ij;
                                        farray[i_f_2jp1 + 2 * ii]     = carray[i_c_j + ii] + qs_i - qs_j + qs_ij;
                                        farray[i_f_2jp1 + 2 * ii + 1] = carray[i_c_j + ii] - qs_i - qs_j - qs_ij;
                                    }
                                }
                            }
                        });

                    // Boundary
                    if constexpr (dim == 1)
                    {
                        auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level], fine_mesh.domain()).on(level);
                        fine_boundary(
                            [&](const auto& i, auto)
                            {
                                auto i_f = fine_mesh.get_index(level, i.start);
                                auto i_c = coarse_mesh.get_index(level - 1, (i / 2).start);
                                for (std::size_t ii = 0; ii < i.size(); ++ii)
                                {
                                    farray[i_f + ii] = carray[i_c + ii];
                                }
                            });
                    }
                    else if constexpr (dim == 2)
                    {
                        xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{
                            {1,  0 },
                            {-1, 0 },
                            {0,  1 },
                            {0,  -1}
                        };
                        for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                        {
                            auto s               = xt::view(stencil, is);
                            auto fine_boundary   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level], s),
                                                                     fine_mesh.domain());
                            auto coarse_boundary = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level - 1], s),
                                                                       coarse_mesh.domain());
                            auto bdry_intersect  = samurai::intersection(coarse_boundary, fine_boundary).on(level - 1);
                            bdry_intersect(
                                [&](const auto& i, const auto& index)
                                {
                                    auto j   = index[0];
                                    auto i_c = coarse_mesh.get_index(level - 1, i.start, j);
                                    if (s(0) == -1) // left boundary
                                    {
                                        // In this case, i_f_2j and i_f_2jp1 do
                                        // not exist because (2*1).start is
                                        // outside of the boundary, so the index
                                        // does not exist (if it did, it would
                                        // be negative; in practive, the
                                        // variable seems to contain a random
                                        // value). Consequently, we use
                                        // (2*i+1).start to get the index, and
                                        // to compensate, we only use 2*ii
                                        // instead of 2*ii+1 in the affectation.
                                        auto i_f_2j   = fine_mesh.get_index(level, (2 * i + 1).start, 2 * j);
                                        auto i_f_2jp1 = fine_mesh.get_index(level, (2 * i + 1).start, 2 * j + 1);
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            farray[i_f_2j + 2 * ii]   = carray[i_c + ii];
                                            farray[i_f_2jp1 + 2 * ii] = carray[i_c + ii];
                                        }
                                    }
                                    else if (s(0) == 1) // right boundary
                                    {
                                        auto i_f_2j   = fine_mesh.get_index(level, (2 * i).start, 2 * j);
                                        auto i_f_2jp1 = fine_mesh.get_index(level, (2 * i).start, 2 * j + 1);
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            farray[i_f_2j + 2 * ii]   = carray[i_c + ii];
                                            farray[i_f_2jp1 + 2 * ii] = carray[i_c + ii];
                                        }
                                    }
                                    else if (s(1) == -1) // bottom boundary
                                    {
                                        auto i_f_2jp1 = fine_mesh.get_index(level, (2 * i).start, 2 * j + 1);
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            farray[i_f_2jp1 + 2 * ii]     = carray[i_c + ii];
                                            farray[i_f_2jp1 + 2 * ii + 1] = carray[i_c + ii];
                                        }
                                    }
                                    else if (s(1) == 1) // top boundary
                                    {
                                        auto i_f_2j = fine_mesh.get_index(level, (2 * i).start, 2 * j);
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            farray[i_f_2j + 2 * ii]     = carray[i_c + ii];
                                            farray[i_f_2j + 2 * ii + 1] = carray[i_c + ii];
                                        }
                                    }
                                });
                        }
                    }
                }
            }

            template <class Mesh>
            void set_prolong_matrix(const Mesh& coarse_mesh, const Mesh& fine_mesh, Mat& P, int prediction_order)
            {
                using mesh_id_t                  = typename Mesh::mesh_id_t;
                static constexpr std::size_t dim = Mesh::dim;

                auto& cm = coarse_mesh[mesh_id_t::cells];
                auto& fm = fine_mesh[mesh_id_t::cells];

                for (std::size_t level = fine_mesh.min_level(); level <= fine_mesh.max_level(); ++level)
                {
                    // Shared cells between coarse and fine mesh
                    auto shared = samurai::intersection(cm[level], fm[level]);
                    shared(
                        [&](auto& i, auto&)
                        {
                            auto index_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                            auto index_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                            for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                            {
                                // farray[index_f + ii] = carray[index_c + ii];
                                MatSetValue(P, index_f + ii, index_c + ii, 1, INSERT_VALUES);
                            }
                        });

                    // Coarse cells to fine cells: prediction
                    auto others = samurai::intersection(cm[level - 1], fm[level]).on(level - 1);
                    others(
                        [&](auto& i, const auto& index)
                        {
                            // static_assert(dim == 1 || dim == 2,
                            // "set_prolong_matrix() not implemented for this
                            // dimension");

                            if constexpr (dim == 1)
                            {
                                auto i_f = static_cast<int>(fine_mesh.get_index(level, (2 * i).start));
                                auto i_c = static_cast<int>(coarse_mesh.get_index(level - 1, i.start));
                                if (prediction_order == 0)
                                {
                                    for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                    {
                                        // Prediction 1D (order 0)
                                        MatSetValue(P, i_f + 2 * ii, i_c + ii, 1, INSERT_VALUES);
                                        MatSetValue(P, i_f + 2 * ii + 1, i_c + ii, 1, INSERT_VALUES);
                                    }
                                }
                                else if (prediction_order == 1)
                                {
                                    for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                    {
                                        // Prediction 1D (order 1)
                                        // farray[i_f + 2*ii  ] = carray[i_c+ii]
                                        // - 1./8*(carray[i_c+ii + 1] -
                                        // carray[i_c+ii - 1]);
                                        MatSetValue(P, i_f + 2 * ii, i_c + ii, 1, INSERT_VALUES);
                                        MatSetValue(P, i_f + 2 * ii, i_c + ii + 1, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, i_f + 2 * ii, i_c + ii - 1, 1. / 8, INSERT_VALUES);
                                        // farray[i_f + 2*ii+1] = carray[i_c+ii]
                                        // + 1./8*(carray[i_c+ii + 1] -
                                        // carray[i_c+ii - 1]);
                                        MatSetValue(P, i_f + 2 * ii + 1, i_c + ii, 1, INSERT_VALUES);
                                        MatSetValue(P, i_f + 2 * ii + 1, i_c + ii + 1, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, i_f + 2 * ii + 1, i_c + ii - 1, -1. / 8, INSERT_VALUES);
                                    }
                                }
                            }
                            else if constexpr (dim == 2)
                            {
                                auto j = index[0];

                                auto i_c_j = static_cast<int>(coarse_mesh.get_index(level - 1, i.start, j));

                                auto i_f_2j   = static_cast<int>(fine_mesh.get_index(level, (2 * i).start, 2 * j));
                                auto i_f_2jp1 = static_cast<int>(fine_mesh.get_index(level, (2 * i).start, 2 * j + 1));

                                if (prediction_order == 0)
                                {
                                    for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                    {
                                        // Prediction 2D (order 0)
                                        auto fine_bottom_left  = i_f_2j + 2 * ii;
                                        auto fine_bottom_right = i_f_2j + 2 * ii + 1;
                                        auto fine_top_left     = i_f_2jp1 + 2 * ii;
                                        auto fine_top_right    = i_f_2jp1 + 2 * ii + 1;

                                        auto coarse = i_c_j + ii;

                                        MatSetValue(P, fine_bottom_left, coarse, 1, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, coarse, 1, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, coarse, 1, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, coarse, 1, INSERT_VALUES);
                                    }
                                }
                                else if (prediction_order == 1)
                                {
                                    auto i_c_jm1 = static_cast<int>(coarse_mesh.get_index(level - 1, i.start, j - 1));
                                    auto i_c_jp1 = static_cast<int>(coarse_mesh.get_index(level - 1, i.start, j + 1));

                                    for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                    {
                                        // Prediction 2D (order 1)
                                        auto fine_bottom_left  = i_f_2j + 2 * ii;
                                        auto fine_bottom_right = i_f_2j + 2 * ii + 1;
                                        auto fine_top_left     = i_f_2jp1 + 2 * ii;
                                        auto fine_top_right    = i_f_2jp1 + 2 * ii + 1;

                                        auto center       = i_c_j + ii;
                                        auto right        = i_c_j + ii + 1;
                                        auto left         = i_c_j + ii - 1;
                                        auto top          = i_c_jp1 + ii;
                                        auto bottom       = i_c_jm1 + ii;
                                        auto top_right    = i_c_jp1 + ii + 1;
                                        auto top_left     = i_c_jp1 + ii - 1;
                                        auto bottom_right = i_c_jm1 + ii + 1;
                                        auto bottom_left  = i_c_jm1 + ii - 1;
                                        /*
                                        double qs_i  = -1./8*(carray[right] -
                                        carray[left]); double qs_j  =
                                        -1./8*(carray[top] - carray[bottom]);
                                        double qs_ij = 1./64*(carray[top_right]
                                        - carray[top_left] -
                                        carray[bottom_right] +
                                        carray[bottom_left]);*/

                                        // farray[fine_bottom_left] =
                                        // carray[center] + qs_i + qs_j - qs_ij;
                                        MatSetValue(P, fine_bottom_left, center, 1, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, right, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, left, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, top, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, bottom, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, top_right, -1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, top_left, 1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, bottom_right, 1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_left, bottom_left, -1. / 64, INSERT_VALUES);

                                        // farray[fine_bottom_right] =
                                        // carray[center] - qs_i + qs_j + qs_ij;
                                        MatSetValue(P, fine_bottom_right, center, 1, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, right, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, left, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, top, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, bottom, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, top_right, 1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, top_left, -1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, bottom_right, -1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_bottom_right, bottom_left, 1. / 64, INSERT_VALUES);

                                        // farray[fine_top_left] =
                                        // carray[center] + qs_i - qs_j + qs_ij;
                                        MatSetValue(P, fine_top_left, center, 1, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, right, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, left, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, top, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, bottom, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, top_right, 1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, top_left, -1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, bottom_right, -1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_top_left, bottom_left, 1. / 64, INSERT_VALUES);

                                        // farray[fine_top_right] =
                                        // carray[center] - qs_i - qs_j - qs_ij;
                                        MatSetValue(P, fine_top_right, center, 1, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, right, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, left, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, top, 1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, bottom, -1. / 8, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, top_right, -1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, top_left, 1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, bottom_right, 1. / 64, INSERT_VALUES);
                                        MatSetValue(P, fine_top_right, bottom_left, -1. / 64, INSERT_VALUES);
                                    }
                                }
                                else
                                {
                                    assert(false
                                           && "set_prolong_matrix() not "
                                              "implemented for dim > 2.");
                                }
                            }
                        });

                    // Boundary
                    if constexpr (dim == 1)
                    {
                        auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level], fine_mesh.domain()).on(level);
                        fine_boundary(
                            [&](const auto& i, auto)
                            {
                                auto i_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                                auto i_c = static_cast<int>(coarse_mesh.get_index(level - 1, (i / 2).start));
                                for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                {
                                    // farray[i_f + ii] = carray[i_c + ii];
                                    MatSetValue(P, i_f + ii, i_c + ii, 1, INSERT_VALUES);
                                }
                            });
                    }
                    else if constexpr (dim == 2)
                    {
                        xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{
                            {1,  0 },
                            {-1, 0 },
                            {0,  1 },
                            {0,  -1}
                        };
                        for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                        {
                            auto s               = xt::view(stencil, is);
                            auto fine_boundary   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level], s),
                                                                     fine_mesh.domain());
                            auto coarse_boundary = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level - 1], s),
                                                                       coarse_mesh.domain());
                            auto bdry_intersect  = samurai::intersection(coarse_boundary, fine_boundary).on(level - 1);
                            bdry_intersect(
                                [&](const auto& i, const auto& index)
                                {
                                    auto j   = index[0];
                                    auto i_c = static_cast<int>(coarse_mesh.get_index(level - 1, i.start, j));
                                    if (s(0) == -1) // left boundary
                                    {
                                        // In this case, i_f_2j and i_f_2jp1 do
                                        // not exist because (2*1).start is
                                        // outside of the boundary, so the index
                                        // does not exist (if it did, it would
                                        // be negative; in practive, the
                                        // variable seems to contain a random
                                        // value). Consequently, we use
                                        // (2*i+1).start to get the index, and
                                        // to compensate, we only use 2*ii
                                        // instead of 2*ii+1 in the affectation.
                                        auto i_f_2j   = static_cast<int>(fine_mesh.get_index(level, (2 * i + 1).start, 2 * j));
                                        auto i_f_2jp1 = static_cast<int>(fine_mesh.get_index(level, (2 * i + 1).start, 2 * j + 1));
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            auto coarse            = i_c + ii;
                                            auto fine_bottom_right = i_f_2j + 2 * ii;
                                            auto fine_top_right    = i_f_2jp1 + 2 * ii;
                                            // farray[fine_bottom_right] =
                                            // carray[coarse];
                                            MatSetValue(P, fine_bottom_right, coarse, 1, INSERT_VALUES);
                                            // farray[i_f_2jp1 + 2*ii] =
                                            // carray[coarse];
                                            MatSetValue(P, fine_top_right, coarse, 1, INSERT_VALUES);
                                        }
                                    }
                                    else if (s(0) == 1) // right boundary
                                    {
                                        auto fine_bottom = static_cast<int>(fine_mesh.get_index(level, (2 * i).start, 2 * j));
                                        auto fine_top    = static_cast<int>(fine_mesh.get_index(level, (2 * i).start, 2 * j + 1));
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            auto left   = 2 * ii;
                                            auto coarse = i_c + ii;
                                            MatSetValue(P, fine_bottom + left, coarse, 1, INSERT_VALUES);
                                            MatSetValue(P, fine_top + left, coarse, 1, INSERT_VALUES);
                                        }
                                    }
                                    else if (s(1) == -1) // bottom boundary
                                    {
                                        auto fine_top = static_cast<int>(fine_mesh.get_index(level, (2 * i).start, 2 * j + 1));
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            auto left   = 2 * ii;
                                            auto right  = 2 * ii + 1;
                                            auto coarse = i_c + ii;
                                            MatSetValue(P, fine_top + left, coarse, 1, INSERT_VALUES);
                                            MatSetValue(P, fine_top + right, coarse, 1, INSERT_VALUES);
                                        }
                                    }
                                    else if (s(1) == 1) // top boundary
                                    {
                                        auto fine_bottom = static_cast<int>(fine_mesh.get_index(level, (2 * i).start, 2 * j));
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            auto left   = 2 * ii;
                                            auto right  = 2 * ii + 1;
                                            auto coarse = i_c + ii;
                                            MatSetValue(P, fine_bottom + left, coarse, 1, INSERT_VALUES);
                                            MatSetValue(P, fine_bottom + right, coarse, 1, INSERT_VALUES);
                                        }
                                    }
                                });
                        }
                    }
                }
            }

            template <class Field>
            Field prolong(const Field& coarse_field, typename Field::mesh_t& fine_mesh, int prediction_order)
            {
                using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
                static constexpr std::size_t dim = Field::mesh_t::dim;

                auto coarse_mesh = coarse_field.mesh();

                auto& cm = coarse_mesh[mesh_id_t::cells];
                auto& fm = fine_mesh[mesh_id_t::cells];

                auto fine_field = samurai::make_field<double, 1>("fine", fine_mesh);
                fine_field.fill(0);

                for (std::size_t level = fine_mesh.min_level(); level <= fine_mesh.max_level(); ++level)
                {
                    // Shared cells between coarse and fine mesh
                    auto shared = samurai::intersection(cm[level], fm[level]);
                    shared(
                        [&](auto& i, auto&)
                        {
                            fine_field(level, i) = coarse_field(level, i);
                        });

                    // Coarse cells to fine cells: prediction
                    auto others = samurai::intersection(cm[level - 1], fm[level]).on(level - 1);
                    if (prediction_order == 0)
                    {
                        others.apply_op(samurai::prediction<0, true>(fine_field, coarse_field));
                    }
                    else if (prediction_order == 1)
                    {
                        others.apply_op(samurai::prediction<1, true>(fine_field, coarse_field));
                    }

                    // Boundary
                    if constexpr (dim == 1)
                    {
                        auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level], fine_mesh.domain()).on(level);
                        fine_boundary(
                            [&](const auto& i, auto)
                            {
                                fine_field(level, i) = coarse_field(level - 1, i / 2);
                            });
                    }
                    else if constexpr (dim == 2)
                    {
                        xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{
                            {1,  0 },
                            {-1, 0 },
                            {0,  1 },
                            {0,  -1}
                        };
                        for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                        {
                            auto s               = xt::view(stencil, is);
                            auto fine_boundary   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level], s),
                                                                     fine_mesh.domain());
                            auto coarse_boundary = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level - 1], s),
                                                                       coarse_mesh.domain());
                            auto bdry_intersect  = samurai::intersection(coarse_boundary, fine_boundary).on(level - 1);
                            bdry_intersect(
                                [&](const auto& i, const auto& index)
                                {
                                    auto j = index[0];
                                    if (s(0) == -1) // left boundary
                                    {
                                        fine_field(level, 2 * i + 1, 2 * j)     = coarse_field(level - 1, i, j);
                                        fine_field(level, 2 * i + 1, 2 * j + 1) = coarse_field(level - 1, i, j);
                                    }
                                    else if (s(0) == 1) // right boundary
                                    {
                                        fine_field(level, 2 * i, 2 * j)     = coarse_field(level - 1, i, j);
                                        fine_field(level, 2 * i, 2 * j + 1) = coarse_field(level - 1, i, j);
                                    }
                                    else if (s(1) == -1) // bottom boundary
                                    {
                                        fine_field(level, 2 * i, 2 * j + 1)     = coarse_field(level - 1, i, j);
                                        fine_field(level, 2 * i + 1, 2 * j + 1) = coarse_field(level - 1, i, j);
                                    }
                                    else if (s(1) == 1) // top boundary
                                    {
                                        fine_field(level, 2 * i, 2 * j)     = coarse_field(level - 1, i, j);
                                        fine_field(level, 2 * i + 1, 2 * j) = coarse_field(level - 1, i, j);
                                    }
                                });
                        }
                    }
                }

                return fine_field;
            }

            template <class Mesh>
            void restrict(const Mesh& fine_mesh, const Mesh& coarse_mesh, const double* farray, double* carray)
            {
                using mesh_id_t                  = typename Mesh::mesh_id_t;
                static constexpr std::size_t dim = Mesh::dim;

                auto& cm = coarse_mesh[mesh_id_t::cells];
                auto& fm = fine_mesh[mesh_id_t::cells];

                for (std::size_t i = 0; i < coarse_mesh.nb_cells(); ++i)
                {
                    carray[i] = 0;
                }

                for (std::size_t level = coarse_mesh.min_level(); level <= coarse_mesh.max_level(); ++level)
                {
                    // Shared cells between coarse and fine mesh
                    auto shared = samurai::intersection(cm[level], fm[level]);
                    shared(
                        [&](auto& i, auto&)
                        {
                            auto index_f = fine_mesh.get_index(level, i.start);
                            auto index_c = coarse_mesh.get_index(level, i.start);
                            for (std::size_t ii = 0; ii < i.size(); ++ii)
                            {
                                carray[index_c + ii] = farray[index_f + ii];
                            }
                        });

                    // Fine cells to coarse cells: projection
                    auto others = samurai::intersection(cm[level], fm[level + 1]).on(level);
                    others(
                        [&](auto& i, const auto& index)
                        {
                            if constexpr (dim == 1)
                            {
                                auto i_f = fine_mesh.get_index(level + 1, (2 * i).start);
                                auto i_c = coarse_mesh.get_index(level, i.start);
                                for (std::size_t ii = 0; ii < i.size(); ++ii)
                                {
                                    // Projection 1D
                                    carray[i_c + ii] = 0.5 * (farray[i_f + 2 * ii] + farray[i_f + 2 * ii + 1]);
                                }
                            }
                            else if constexpr (dim == 2)
                            {
                                auto j        = index[0];
                                auto i_c      = coarse_mesh.get_index(level, i.start, j);
                                auto i_f_2j   = fine_mesh.get_index(level + 1, (2 * i).start, 2 * j);
                                auto i_f_2jp1 = fine_mesh.get_index(level + 1, (2 * i).start, 2 * j + 1);
                                for (std::size_t ii = 0; ii < i.size(); ++ii)
                                {
                                    // Projection 2D
                                    carray[i_c + ii] = 0.25
                                                     * (farray[i_f_2j + 2 * ii] + farray[i_f_2jp1 + 2 * ii] + farray[i_f_2j + 2 * ii + 1]
                                                        + farray[i_f_2jp1 + 2 * ii + 1]);
                                }
                            }
                        });

                    // Boundary
                    if constexpr (dim == 1)
                    {
                        auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level + 1], fine_mesh.domain()).on(level + 1);
                        fine_boundary(
                            [&](const auto& i, auto)
                            {
                                // cf(level, i>>1) = fine_field(level + 1, i);
                                auto i_f = fine_mesh.get_index(level + 1, i.start);
                                auto i_c = coarse_mesh.get_index(level, (i / 2).start);
                                for (std::size_t ii = 0; ii < i.size(); ++ii)
                                {
                                    carray[i_c + ii] = farray[i_f + ii];
                                }
                            });
                    }
                    else if constexpr (dim == 2)
                    {
                        xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{
                            {1,  0 },
                            {-1, 0 },
                            {0,  1 },
                            {0,  -1}
                        };
                        for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                        {
                            auto s                  = xt::view(stencil, is);
                            auto fine_boundary_is   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level + 1], s),
                                                                        fine_mesh.domain());
                            auto coarse_boundary_is = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level], s),
                                                                          coarse_mesh.domain());
                            auto bdry_intersect     = samurai::intersection(coarse_boundary_is, fine_boundary_is).on(level);
                            bdry_intersect(
                                [&](const auto& i, const auto& index)
                                {
                                    auto j = index[0];

                                    auto i_c      = coarse_mesh.get_index(level, i.start, j);
                                    auto i_f_2j   = fine_mesh.get_index(level + 1, (2 * i).start, 2 * j);
                                    auto i_f_2jp1 = fine_mesh.get_index(level + 1, (2 * i).start, 2 * j + 1);

                                    if (s(0) == -1) // left boundary
                                    {
                                        // coarse_field(level, i, j) =
                                        // 0.5*(fine_field(level+1, 2*i+1, 2*j
                                        // ) + fine_field(level+1, 2*i+1,
                                        // 2*j+1)); for (int ii=0;
                                        // ii<static_cast<int>(i.size()); ++ii)
                                        // carray[i_c + ii] = 0.5*(farray[i_f_2j
                                        // + 2*ii+1] + farray[i_f_2jp1 +
                                        // 2*ii+1]);

                                        // In this case, i_f_2j and i_f_2jp1 do
                                        // not exist because (2*1).start is
                                        // outside of the boundary, so the index
                                        // does not exist (if it did, it would
                                        // be negative; in practive, the
                                        // variable seems to contain a random
                                        // value). Consequently, we use
                                        // (2*i+1).start to get the index, and
                                        // to compensate, we only use 2*ii
                                        // instead of 2*ii+1 in the affectation.
                                        i_f_2j   = fine_mesh.get_index(level + 1, (2 * i + 1).start, 2 * j);
                                        i_f_2jp1 = fine_mesh.get_index(level + 1, (2 * i + 1).start, 2 * j + 1);
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            carray[i_c + ii] = 0.5 * (farray[i_f_2j + 2 * ii] + farray[i_f_2jp1 + 2 * ii]);
                                        }
                                    }
                                    else if (s(0) == 1) // right boundary
                                    {
                                        // coarse_field(level, i, j) =
                                        // 0.5*(fine_field(level+1, 2*i  , 2*j
                                        // ) + fine_field(level+1, 2*i  ,
                                        // 2*j+1));
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            carray[i_c + ii] = 0.5 * (farray[i_f_2j + 2 * ii] + farray[i_f_2jp1 + 2 * ii]);
                                        }
                                    }
                                    else if (s(1) == -1) // bottom boundary
                                    {
                                        // coarse_field(level, i, j) =
                                        // 0.5*(fine_field(level+1, 2*i  ,
                                        // 2*j+1) + fine_field(level+1, 2*i+1,
                                        // 2*j+1));
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            carray[i_c + ii] = 0.5 * (farray[i_f_2jp1 + 2 * ii] + farray[i_f_2jp1 + 2 * ii + 1]);
                                        }
                                    }
                                    else if (s(1) == 1) // top boundary
                                    {
                                        // coarse_field(level, i, j) =
                                        // 0.5*(fine_field(level+1, 2*i  , 2*j
                                        // ) + fine_field(level+1, 2*i+1, 2*j
                                        // ));
                                        for (std::size_t ii = 0; ii < i.size(); ++ii)
                                        {
                                            carray[i_c + ii] = 0.5 * (farray[i_f_2j + 2 * ii] + farray[i_f_2j + 2 * ii + 1]);
                                        }
                                    }
                                    else
                                    {
                                        assert(false);
                                    }
                                });
                        }
                    }
                }
            }

            template <class Mesh>
            void set_restrict_matrix(const Mesh& fine_mesh, const Mesh& coarse_mesh, Mat& R)
            {
                using mesh_id_t                  = typename Mesh::mesh_id_t;
                static constexpr std::size_t dim = Mesh::dim;

                auto& cm = coarse_mesh[mesh_id_t::cells];
                auto& fm = fine_mesh[mesh_id_t::cells];

                for (std::size_t level = coarse_mesh.min_level(); level <= coarse_mesh.max_level(); ++level)
                {
                    // Shared cells between coarse and fine mesh
                    auto shared = samurai::intersection(cm[level], fm[level]);
                    shared(
                        [&](auto& i, auto&)
                        {
                            auto index_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                            auto index_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                            for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                            {
                                // carray[index_c + ii] = farray[index_f + ii];
                                MatSetValue(R, index_c + ii, index_f + ii, 1, INSERT_VALUES);
                            }
                        });

                    // Fine cells to coarse cells: projection
                    auto others = samurai::intersection(cm[level], fm[level + 1]).on(level);
                    others(
                        [&](auto& i, const auto& index)
                        {
                            if constexpr (dim == 1)
                            {
                                auto i_f = static_cast<int>(fine_mesh.get_index(level + 1, (2 * i).start));
                                auto i_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                                for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                {
                                    // Projection (here in 1D only)
                                    // carray[i_c + ii] = 0.5*(farray[i_f +
                                    // 2*ii] + farray[i_f + 2*ii+1]);
                                    MatSetValue(R, i_c + ii, i_f + 2 * ii, 0.5, INSERT_VALUES);
                                    MatSetValue(R, i_c + ii, i_f + 2 * ii + 1, 0.5, INSERT_VALUES);
                                }
                            }
                            else if constexpr (dim == 2)
                            {
                                auto j        = index[0];
                                auto i_c      = static_cast<int>(coarse_mesh.get_index(level, i.start, j));
                                auto i_f_2j   = static_cast<int>(fine_mesh.get_index(level + 1, (2 * i).start, 2 * j));
                                auto i_f_2jp1 = static_cast<int>(fine_mesh.get_index(level + 1, (2 * i).start, 2 * j + 1));
                                for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                {
                                    // Projection 2D
                                    // carray[i_c + ii] = 0.25*(farray[i_f_2j +
                                    // 2*ii  ] + farray[i_f_2jp1 + 2*ii  ] +
                                    // farray[i_f_2j   + 2*ii+1] +
                                    // farray[i_f_2jp1 + 2*ii+1]);
                                    MatSetValue(R, i_c + ii, i_f_2j + 2 * ii, 0.25, INSERT_VALUES);
                                    MatSetValue(R, i_c + ii, i_f_2jp1 + 2 * ii, 0.25, INSERT_VALUES);
                                    MatSetValue(R, i_c + ii, i_f_2j + 2 * ii + 1, 0.25, INSERT_VALUES);
                                    MatSetValue(R, i_c + ii, i_f_2jp1 + 2 * ii + 1, 0.25, INSERT_VALUES);
                                }
                            }
                        });

                    // Boundary
                    if constexpr (dim == 1)
                    {
                        auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level + 1], fine_mesh.domain()).on(level + 1);
                        fine_boundary(
                            [&](const auto& i, auto)
                            {
                                // cf(level, i>>1) = fine_field(level + 1, i);
                                auto i_f = static_cast<int>(fine_mesh.get_index(level + 1, i.start));
                                auto i_c = static_cast<int>(coarse_mesh.get_index(level, (i / 2).start));
                                for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                {
                                    // carray[i_c + ii] = farray[i_f + ii];
                                    MatSetValue(R, i_c + ii, i_f + ii, 1, INSERT_VALUES);
                                }
                            });
                    }
                    else if constexpr (dim == 2)
                    {
                        xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{
                            {1,  0 },
                            {-1, 0 },
                            {0,  1 },
                            {0,  -1}
                        };
                        for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                        {
                            auto s                  = xt::view(stencil, is);
                            auto fine_boundary_is   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level + 1], s),
                                                                        fine_mesh.domain());
                            auto coarse_boundary_is = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level], s),
                                                                          coarse_mesh.domain());
                            auto bdry_intersect     = samurai::intersection(coarse_boundary_is, fine_boundary_is).on(level);
                            bdry_intersect(
                                [&](const auto& i, const auto& index)
                                {
                                    auto j = index[0];

                                    auto i_c      = static_cast<int>(coarse_mesh.get_index(level, i.start, j));
                                    auto i_f_2j   = static_cast<int>(fine_mesh.get_index(level + 1, (2 * i).start, 2 * j));
                                    auto i_f_2jp1 = static_cast<int>(fine_mesh.get_index(level + 1, (2 * i).start, 2 * j + 1));

                                    if (s(0) == -1) // left boundary
                                    {
                                        // In this case, i_f_2j and i_f_2jp1 do
                                        // not exist because (2*1).start is
                                        // outside of the boundary, so the index
                                        // does not exist (if it did, it would
                                        // be negative; in practice, the
                                        // variable seems to contain a random
                                        // value). Consequently, we use
                                        // (2*i+1).start to get the index, and
                                        // to compensate, we only use 2*ii
                                        // instead of 2*ii+1 in the affectation.
                                        i_f_2j   = static_cast<int>(fine_mesh.get_index(level + 1, (2 * i + 1).start, 2 * j));
                                        i_f_2jp1 = static_cast<int>(fine_mesh.get_index(level + 1, (2 * i + 1).start, 2 * j + 1));
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            // carray[i_c + ii] =
                                            // 0.5*(farray[i_f_2j + 2*ii] +
                                            // farray[i_f_2jp1 + 2*ii]);
                                            MatSetValue(R, i_c + ii, i_f_2j + 2 * ii, 0.5, INSERT_VALUES);
                                            MatSetValue(R, i_c + ii, i_f_2jp1 + 2 * ii, 0.5, INSERT_VALUES);
                                        }
                                    }
                                    else if (s(0) == 1) // right boundary
                                    {
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            // carray[i_c + ii] =
                                            // 0.5*(farray[i_f_2j + 2*ii] +
                                            // farray[i_f_2jp1 + 2*ii]);
                                            MatSetValue(R, i_c + ii, i_f_2j + 2 * ii, 0.5, INSERT_VALUES);
                                            MatSetValue(R, i_c + ii, i_f_2jp1 + 2 * ii, 0.5, INSERT_VALUES);
                                        }
                                    }
                                    else if (s(1) == -1) // bottom boundary
                                    {
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            // carray[i_c + ii] =
                                            // 0.5*(farray[i_f_2jp1 + 2*ii] +
                                            // farray[i_f_2jp1 + 2*ii+1]);
                                            MatSetValue(R, i_c + ii, i_f_2jp1 + 2 * ii, 0.5, INSERT_VALUES);
                                            MatSetValue(R, i_c + ii, i_f_2jp1 + 2 * ii + 1, 0.5, INSERT_VALUES);
                                        }
                                    }
                                    else if (s(1) == 1) // top boundary
                                    {
                                        for (int ii = 0; ii < static_cast<int>(i.size()); ++ii)
                                        {
                                            // carray[i_c + ii] =
                                            // 0.5*(farray[i_f_2j + 2*ii] +
                                            // farray[i_f_2j + 2*ii+1]);
                                            MatSetValue(R, i_c + ii, i_f_2j + 2 * ii, 0.5, INSERT_VALUES);
                                            MatSetValue(R, i_c + ii, i_f_2j + 2 * ii + 1, 0.5, INSERT_VALUES);
                                        }
                                    }
                                    else
                                    {
                                        assert(false
                                               && "set_rectrict_matrix() not "
                                                  "implemented for dim > 2.");
                                    }
                                });
                        }
                    }
                }
            }

            template <class Field>
            auto restrict(const Field& fine_field, typename Field::mesh_t& coarse_mesh)
            {
                using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
                static constexpr std::size_t dim = Field::mesh_t::dim;

                auto fine_mesh = fine_field.mesh();

                auto& cm = coarse_mesh[mesh_id_t::cells];
                auto& fm = fine_mesh[mesh_id_t::cells];

                auto coarse_field = samurai::make_field<double, 1>("coarse", coarse_mesh);
                coarse_field.fill(0);

                for (std::size_t level = coarse_mesh.min_level(); level <= coarse_mesh.max_level(); ++level)
                {
                    // Shared cells between coarse and fine mesh
                    auto shared = samurai::intersection(cm[level], fm[level]);
                    shared(
                        [&](auto& i, auto&)
                        {
                            coarse_field(level, i) = fine_field(level, i);
                        });

                    // Fine cells to coarse cells: projection
                    auto others = samurai::intersection(cm[level], fm[level + 1]).on(level);
                    others.apply_op(samurai::projection(coarse_field, fine_field));

                    // Boundary
                    if constexpr (dim == 1)
                    {
                        auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level + 1], fine_mesh.domain());
                        fine_boundary(
                            [&](const auto& i, auto)
                            {
                                coarse_field(level, i / 2) = fine_field(level + 1, i);
                            });
                    }
                    else if constexpr (dim == 2)
                    {
                        // !!! This code might be specific to the square domain
                        // !!!

                        xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{
                            {1,  0 },
                            {-1, 0 },
                            {0,  1 },
                            {0,  -1}
                        };

                        /*auto coarse_boundary =
                        samurai::difference(coarse_mesh[mesh_id_t::reference][level],
                        coarse_mesh.domain()); auto fine_boundary =
                        samurai::difference(fine_mesh[mesh_id_t::reference][level+1],
                        fine_mesh.domain()); auto coarse_corners =
                        coarse_boundary; auto fine_corners = fine_boundary;*/

                        for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                        {
                            auto s                  = xt::view(stencil, is);
                            auto fine_boundary_is   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level + 1], s),
                                                                        fine_mesh.domain());
                            auto coarse_boundary_is = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level], s),
                                                                          coarse_mesh.domain());
                            auto bdry_intersect     = samurai::intersection(coarse_boundary_is, fine_boundary_is).on(level);
                            bdry_intersect(
                                [&](const auto& i, const auto& index)
                                {
                                    auto j = index[0];
                                    if (s(0) == -1) // left boundary
                                    {
                                        coarse_field(level, i, j) = 0.5
                                                                  * (fine_field(level + 1, 2 * i + 1, 2 * j)
                                                                     + fine_field(level + 1, 2 * i + 1, 2 * j + 1));
                                    }
                                    else if (s(0) == 1) // right boundary
                                    {
                                        coarse_field(level, i, j) = 0.5
                                                                  * (fine_field(level + 1, 2 * i, 2 * j)
                                                                     + fine_field(level + 1, 2 * i, 2 * j + 1));
                                    }
                                    else if (s(1) == -1) // bottom boundary
                                    {
                                        coarse_field(level, i, j) = 0.5
                                                                  * (fine_field(level + 1, 2 * i, 2 * j + 1)
                                                                     + fine_field(level + 1, 2 * i + 1, 2 * j + 1));
                                    }
                                    else if (s(1) == 1) // top boundary
                                    {
                                        coarse_field(level, i, j) = 0.5
                                                                  * (fine_field(level + 1, 2 * i, 2 * j)
                                                                     + fine_field(level + 1, 2 * i + 1, 2 * j));
                                    }
                                    else
                                    {
                                        assert(false);
                                    }
                                });
                            // coarse_corners =
                            // samurai::difference(coarse_corners,
                            // coarse_boundary_is); fine_corners =
                            // samurai::difference(fine_corners,
                            // fine_boundary_is);
                        }

                        // Corners ghosts
                        /*xt::xtensor_fixed<int, xt::xshape<4, 2>>
                        diagonal_stencil{{1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
                        for (std::size_t is = 0; is<diagonal_stencil.shape()[0];
                        ++is)
                        {
                            auto s = xt::view(diagonal_stencil, is);
                            auto fine_corners_is   =
                        samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level+1],
                        s), fine_mesh.domain()); auto coarse_corners_is =
                        samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level],
                        s), coarse_mesh.domain()); auto corner_intersect =
                        samurai::intersection(coarse_corners_is,
                        fine_corners_is).on(level); corner_intersect([&](const
                        auto& i, const auto& index)
                            {
                                auto j = index[0];
                                std::cout << "avant i = " << i << ", j = " << j
                        << ":" << coarse_field(level, i, j) << std::endl; if
                        (s(0) == 1 && s(1) == 1) // top-right
                                    coarse_field(level, i, j) =
                        fine_field(level+1, 2*i  , 2*j  ); else if (s(0) == -1
                        && s(1) == 1) // top-left coarse_field(level, i, j) =
                        fine_field(level+1, 2*i+1, 2*j  ); else if (s(0) == -1
                        && s(1) == -1) // bottom-left coarse_field(level, i, j)
                        = fine_field(level+1, 2*i+1, 2*j+1); else if (s(0) == 1
                        && s(1) == -1) // bottom-right coarse_field(level, i, j)
                        = fine_field(level+1, 2*i  , 2*j+1); std::cout << "apres
                        i = " << i << ", j = " << j << ":" <<
                        coarse_field(level, i, j) << std::endl;
                            });
                        }*/
                    }
                }

                return coarse_field;
            }

        }
    }
} // namespace
