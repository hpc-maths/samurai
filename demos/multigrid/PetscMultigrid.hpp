#pragma once
#include <samurai/numeric/projection.hpp>
#include <samurai/numeric/prediction.hpp>
#include "LevelCtx.hpp"

template<class Dsctzr>
class PetscMultigrid
{
public:
    static void prolong(LevelCtx<Dsctzr>& coarse, LevelCtx<Dsctzr>& fine, const double* carray, double* farray)
    {
        auto coarse_mesh = coarse.mesh();
        auto fine_mesh = fine.mesh();
        using mesh_id_t = typename decltype(coarse_mesh)::mesh_id_t;

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        for(std::size_t level = fine_mesh.min_level(); level <= fine_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                auto index_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                auto index_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    farray[index_f + ii] = carray[index_c + ii];
                }
            });

            // Coarse cells to fine cells: prediction
            auto others = samurai::intersection(cm[level - 1], fm[level]).on(level-1);
            others([&](auto& i, auto&)
            {
                auto i_f = static_cast<int>(fine_mesh.get_index(level, (2*i).start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level-1, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    // Prediction (here in 1D only)
                    farray[i_f + 2*ii  ] = carray[i_c+ii] - 1./8*(carray[i_c+ii + 1] - carray[i_c+ii - 1]);
                    farray[i_f + 2*ii+1] = carray[i_c+ii] + 1./8*(carray[i_c+ii + 1] - carray[i_c+ii - 1]);
                }
            });

            // Boundary
            auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level],
                                                     fine_mesh.domain()).on(level);
            fine_boundary([&](const auto& i, auto)
            {
                auto i_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level-1, (i/2).start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    farray[i_f + ii] = carray[i_c + ii];
                }
            });
        }
    }

    static void set_prolong_matrix(LevelCtx<Dsctzr>& coarse, LevelCtx<Dsctzr>& fine, Mat& P)
    {
        auto coarse_mesh = coarse.mesh();
        auto fine_mesh = fine.mesh();
        using mesh_id_t = typename decltype(coarse_mesh)::mesh_id_t;

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        for(std::size_t level = fine_mesh.min_level(); level <= fine_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                auto index_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                auto index_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    //farray[index_f + ii] = carray[index_c + ii];
                    MatSetValue(P, index_f + ii, index_c + ii, 1, INSERT_VALUES);
                }
            });

            // Coarse cells to fine cells: prediction
            auto others = samurai::intersection(cm[level - 1], fm[level]).on(level-1);
            others([&](auto& i, auto&)
            {
                auto i_f = static_cast<int>(fine_mesh.get_index(level, (2*i).start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level-1, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    // Prediction (here in 1D only)
                    //farray[i_f + 2*ii  ] = carray[i_c+ii] - 1./8*(carray[i_c+ii + 1] - carray[i_c+ii - 1]);
                    MatSetValue(P, i_f + 2*ii,     i_c+ii    ,  1   , INSERT_VALUES);
                    MatSetValue(P, i_f + 2*ii,     i_c+ii + 1, -1./8, INSERT_VALUES);
                    MatSetValue(P, i_f + 2*ii,     i_c+ii - 1,  1./8, INSERT_VALUES);
                    //farray[i_f + 2*ii+1] = carray[i_c+ii] + 1./8*(carray[i_c+ii + 1] - carray[i_c+ii - 1]);
                    MatSetValue(P, i_f + 2*ii + 1, i_c+ii    ,  1   , INSERT_VALUES);
                    MatSetValue(P, i_f + 2*ii + 1, i_c+ii + 1,  1./8, INSERT_VALUES);
                    MatSetValue(P, i_f + 2*ii + 1, i_c+ii - 1, -1./8, INSERT_VALUES);
                    
                }
            });

            // Boundary
            auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level],
                                                     fine_mesh.domain()).on(level);
            fine_boundary([&](const auto& i, auto)
            {
                auto i_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level-1, (i/2).start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    //farray[i_f + ii] = carray[i_c + ii];
                    MatSetValue(P, i_f + ii, i_c + ii, 1, INSERT_VALUES);
                }
            });
        }
    }

    template<class Field>
    static Field prolong(const Field& coarse_field, typename Field::mesh_t& fine_mesh)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto coarse_mesh = coarse_field.mesh();

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        auto fine_field = samurai::make_field<double, 1>("fine", fine_mesh);
        fine_field.fill(0);

        for(std::size_t level = fine_mesh.min_level(); level <= fine_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                fine_field(level, i) = coarse_field(level, i);
            });

            // Coarse cells to fine cells: prediction
            auto others = samurai::intersection(cm[level - 1], fm[level]).on(level-1);
            others.apply_op(samurai::prediction<1, true>(fine_field, coarse_field));

            // Boundary
            if (Field::mesh_t::dim == 1)
            {
                auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level],
                                                        fine_mesh.domain()).on(level);
                fine_boundary([&](const auto& i, auto)
                {
                    fine_field(level, i) = coarse_field(level - 1, i/2);
                });
            }
            else if (Field::mesh_t::dim == 2)
            {
                xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (std::size_t is = 0; is<stencil.shape()[0]; ++is)
                {
                    auto s = xt::view(stencil, is);
                    auto fine_boundary   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level], s), fine_mesh.domain());
                    auto coarse_boundary = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level-1], s), coarse_mesh.domain());
                    auto bdry_intersect = samurai::intersection(coarse_boundary, fine_boundary).on(level-1);
                    bdry_intersect([&](const auto& i, const auto& index)
                    {
                        auto j = index[0];
                        if (s(0) == -1) // left boundary
                        {
                            fine_field(level, 2*i+1, 2*j)   = coarse_field(level-1, i, j);
                            fine_field(level, 2*i+1, 2*j+1) = coarse_field(level-1, i, j);
                        }
                        else if (s(0) == 1) // right boundary
                        {
                            fine_field(level, 2*i, 2*j)   = coarse_field(level-1, i, j);
                            fine_field(level, 2*i, 2*j+1) = coarse_field(level-1, i, j);
                        }
                        else if (s(1) == -1) // bottom boundary
                        {
                            fine_field(level, 2*i,   2*j+1) = coarse_field(level-1, i, j);
                            fine_field(level, 2*i+1, 2*j+1) = coarse_field(level-1, i, j);
                        }
                        else if (s(1) == 1) // top boundary
                        {
                            fine_field(level, 2*i,   2*j) = coarse_field(level-1, i, j);
                            fine_field(level, 2*i+1, 2*j) = coarse_field(level-1, i, j);
                        }
                    });
                }

            }
            else
                assert(false);
        }

        return fine_field;
    }

    






    static void restrict(LevelCtx<Dsctzr>& fine, LevelCtx<Dsctzr>& coarse, const double* farray, double* carray)
    {
        auto coarse_mesh = coarse.mesh();
        auto fine_mesh = fine.mesh();
        using mesh_id_t = typename decltype(coarse_mesh)::mesh_id_t;

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        for(std::size_t level = coarse_mesh.min_level(); level <= coarse_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                auto index_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                auto index_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    carray[index_c + ii] = farray[index_f + ii];
                }
            });

            // Fine cells to coarse cells: projection
            auto others = samurai::intersection(cm[level], fm[level + 1]).on(level);
            //others.apply_op(samurai::projection(coarse_field, fine_field));
            others([&](auto& i, auto&)
            {
                auto i_f = static_cast<int>(fine_mesh.get_index(level+1, (2*i).start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    // Projection (here in 1D only)
                    carray[i_c + ii] = 0.5*(farray[i_f + 2*ii] + farray[i_f + 2*ii+1]);
                }
            });

            // Boundary
            auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level+1],
                                                    fine_mesh.domain()).on(level+1);
            fine_boundary([&](const auto& i, auto)
            {
                //cf(level, i>>1) = fine_field(level + 1, i);
                auto i_f = static_cast<int>(fine_mesh.get_index(level+1, i.start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level, (i/2).start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    carray[i_c + ii] = farray[i_f + ii];
                }
            });
        }
    }

    static void set_restrict_matrix(LevelCtx<Dsctzr>& fine, LevelCtx<Dsctzr>& coarse, Mat& R)
    {
        auto coarse_mesh = coarse.mesh();
        auto fine_mesh = fine.mesh();
        using mesh_id_t = typename decltype(coarse_mesh)::mesh_id_t;

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        for(std::size_t level = coarse_mesh.min_level(); level <= coarse_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                auto index_f = static_cast<int>(fine_mesh.get_index(level, i.start));
                auto index_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    //carray[index_c + ii] = farray[index_f + ii];
                    MatSetValue(R, index_c + ii, index_f + ii, 1, INSERT_VALUES);
                }
            });

            // Fine cells to coarse cells: projection
            auto others = samurai::intersection(cm[level], fm[level + 1]).on(level);
            //others.apply_op(samurai::projection(coarse_field, fine_field));
            others([&](auto& i, auto&)
            {
                auto i_f = static_cast<int>(fine_mesh.get_index(level+1, (2*i).start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    // Projection (here in 1D only)
                    //carray[i_c + ii] = 0.5*(farray[i_f + 2*ii] + farray[i_f + 2*ii+1]);
                    MatSetValue(R, i_c + ii, i_f + 2*ii    , 0.5, INSERT_VALUES);
                    MatSetValue(R, i_c + ii, i_f + 2*ii + 1, 0.5, INSERT_VALUES);
                }
            });

            // Boundary
            auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level+1],
                                                    fine_mesh.domain()).on(level+1);
            fine_boundary([&](const auto& i, auto)
            {
                //cf(level, i>>1) = fine_field(level + 1, i);
                auto i_f = static_cast<int>(fine_mesh.get_index(level+1, i.start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level, (i/2).start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    //carray[i_c + ii] = farray[i_f + ii];
                    MatSetValue(R, i_c + ii, i_f + ii, 1, INSERT_VALUES);
                }
            });
        }
    }



    template<class Field>
    static auto restrict(const Field& fine_field, typename Field::mesh_t& coarse_mesh)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto fine_mesh = fine_field.mesh();

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        auto coarse_field = samurai::make_field<double, 1>("coarse", coarse_mesh);
        coarse_field.fill(0);
        

        for(std::size_t level = coarse_mesh.min_level(); level <= coarse_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                coarse_field(level, i) = fine_field(level, i);
            });

            // Fine cells to coarse cells: projection
            auto others = samurai::intersection(cm[level], fm[level + 1]).on(level);
            others.apply_op(samurai::projection(coarse_field, fine_field));

            // Boundary
            if (Field::mesh_t::dim == 1)
            {
                auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level+1], fine_mesh.domain());
                fine_boundary([&](const auto& i, auto)
                {
                    coarse_field(level, i/2) = fine_field(level+1, i);
                });
            }
            else if (Field::mesh_t::dim == 2)
            {
                // !!! This code might be specific to the square domain !!!

                xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

                /*auto coarse_boundary = samurai::difference(coarse_mesh[mesh_id_t::reference][level], coarse_mesh.domain());
                auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level+1], fine_mesh.domain());
                auto coarse_corners = coarse_boundary;
                auto fine_corners = fine_boundary;*/

                for (std::size_t is = 0; is<stencil.shape()[0]; ++is)
                {
                    auto s = xt::view(stencil, is);
                    auto fine_boundary_is   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level+1], s), fine_mesh.domain());
                    auto coarse_boundary_is = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level], s), coarse_mesh.domain());
                    auto bdry_intersect = samurai::intersection(coarse_boundary_is, fine_boundary_is).on(level);
                    bdry_intersect([&](const auto& i, const auto& index)
                    {
                        auto j = index[0];
                        if (s(0) == -1) // left boundary
                            coarse_field(level, i, j) = 0.5*(fine_field(level+1, 2*i+1, 2*j  ) + fine_field(level+1, 2*i+1, 2*j+1));
                        else if (s(0) == 1) // right boundary
                            coarse_field(level, i, j) = 0.5*(fine_field(level+1, 2*i  , 2*j  ) + fine_field(level+1, 2*i  , 2*j+1));
                        else if (s(1) == -1) // bottom boundary
                            coarse_field(level, i, j) = 0.5*(fine_field(level+1, 2*i  , 2*j+1) + fine_field(level+1, 2*i+1, 2*j+1));
                        else if (s(1) == 1) // top boundary
                            coarse_field(level, i, j) = 0.5*(fine_field(level+1, 2*i  , 2*j  ) + fine_field(level+1, 2*i+1, 2*j  ));
                        else
                            assert(false);
                    });
                    //coarse_corners = samurai::difference(coarse_corners, coarse_boundary_is);
                    //fine_corners = samurai::difference(fine_corners, fine_boundary_is);
                }

                // Corners ghosts
                /*xt::xtensor_fixed<int, xt::xshape<4, 2>> diagonal_stencil{{1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
                for (std::size_t is = 0; is<diagonal_stencil.shape()[0]; ++is)
                {
                    auto s = xt::view(diagonal_stencil, is);
                    auto fine_corners_is   = samurai::difference(samurai::translate(fine_mesh[mesh_id_t::cells][level+1], s), fine_mesh.domain());
                    auto coarse_corners_is = samurai::difference(samurai::translate(coarse_mesh[mesh_id_t::cells][level], s), coarse_mesh.domain());
                    auto corner_intersect = samurai::intersection(coarse_corners_is, fine_corners_is).on(level);
                    corner_intersect([&](const auto& i, const auto& index)
                    {
                        auto j = index[0];
                        std::cout << "avant i = " << i << ", j = " << j << ":" << coarse_field(level, i, j) << std::endl;
                        if (s(0) == 1 && s(1) == 1) // top-right
                            coarse_field(level, i, j) = fine_field(level+1, 2*i  , 2*j  );
                        else if (s(0) == -1 && s(1) == 1) // top-left
                            coarse_field(level, i, j) = fine_field(level+1, 2*i+1, 2*j  );
                        else if (s(0) == -1 && s(1) == -1) // bottom-left
                            coarse_field(level, i, j) = fine_field(level+1, 2*i+1, 2*j+1);
                        else if (s(0) == 1 && s(1) == -1) // bottom-right
                            coarse_field(level, i, j) = fine_field(level+1, 2*i  , 2*j+1);
                        std::cout << "apres i = " << i << ", j = " << j << ":" << coarse_field(level, i, j) << std::endl;
                    });
                }*/
            }
            else
                assert(false);
        }

        return coarse_field;
    }
};