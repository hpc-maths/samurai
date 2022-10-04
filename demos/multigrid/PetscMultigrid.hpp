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

            // Others
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

            // Others
            auto others = samurai::intersection(cm[level - 1], fm[level]).on(level-1);
            others([&](auto& i, auto&)
            {
                auto i_f = static_cast<int>(fine_mesh.get_index(level, (2*i).start));
                auto i_c = static_cast<int>(coarse_mesh.get_index(level-1, i.start));
                for(int ii=0; ii<i.size(); ++ii)
                {
                    
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
        auto fine_field = samurai::make_field<double, 1>("fine", fine_mesh);

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        for(std::size_t level = fine_mesh.min_level(); level <= fine_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                fine_field(level, i) = coarse_field(level, i);
            });

            // Others
            auto others = samurai::intersection(cm[level - 1], fm[level]).on(level-1);
            others.apply_op(samurai::prediction<1, true>(fine_field, coarse_field));
        }

        // Boundary
        auto update_bc = [&](auto& ff, std::size_t level)
        {
            auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level],
                                                    fine_mesh.domain()).on(level);
            fine_boundary([&](const auto& i, auto)
            {
                ff(level, i) = coarse_field(level - 1, i/2);
            });
        };
        samurai::update_ghost_mr(fine_field, update_bc);

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

            // Others
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

            // Others
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
        auto coarse_field = samurai::make_field<double, 1>("coarse", coarse_mesh);

        auto& cm = coarse_mesh[mesh_id_t::cells];
        auto& fm = fine_mesh[mesh_id_t::cells];

        // samurai::for_each_interval(coarse_mesh, [&](std::size_t level, auto& i, auto&)
        // {
        //     std::cout << "i " << i << " 2*i " << 2*i << std::endl;
        //     coarse_field(level, i) = 0.5*(fineField(level + 1, 2*i) + fineField(level + 1, 2*i + 1));
        // });

        for(std::size_t level = coarse_mesh.min_level(); level <= coarse_mesh.max_level(); ++level)
        {
            // Shared cells between coarse and fine mesh
            auto shared = samurai::intersection(cm[level], fm[level]);
            shared([&](auto& i, auto&)
            {
                coarse_field(level, i) = fine_field(level, i);
            });

            // Others
            auto others = samurai::intersection(cm[level], fm[level + 1]).on(level);
            others.apply_op(samurai::projection(coarse_field, fine_field));

            /*// Boundary ghosts
            auto bc_in_ghosts = samurai::difference(mesh[mesh_id_t::reference][level],
                                            mesh[mesh_id_t::cells][level]);
            bc_in_ghosts([&](const auto& i, const auto&)
            {
                coarse_field(level, i) = 0.;
            });*/
        }

        // Boundary
        auto update_bc = [&](auto& cf, std::size_t level)
        {
            //auto coarse_boundary = samurai::difference(coarse_mesh[mesh_id_t::reference][level],
            //                                        coarse_mesh.domain()).on(level);
            auto fine_boundary = samurai::difference(fine_mesh[mesh_id_t::reference][level+1],
                                                    fine_mesh.domain()).on(level+1);
            //auto inters_bdry = samurai::intersection(coarse_boundary, fine_boundary).on(level);
            fine_boundary([&](const auto& i, auto)
            {
                cf(level, i>>1) = fine_field(level + 1, i);
            });
            //inters_bdry.apply_op(samurai::projection(cf, fine_field));
        };
        samurai::update_ghost_mr(coarse_field, update_bc);

        return coarse_field;
    }
};