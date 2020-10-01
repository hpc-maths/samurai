#pragma once

#include <array>
#include <memory>

#include <spdlog/spdlog.h>

#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

#include "cell.hpp"
#include "bc.hpp"
#include "field_expression.hpp"
#include "mr/mesh.hpp"
#include "mr/mesh_type.hpp"

namespace mure
{
    template<class TInterval>
    class update_boundary_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(update_boundary_op)

        template<class T, class BC, class stencil_t>
        inline void operator()(Dim<1>, T &field, const BC &bc, const stencil_t& stencil) const
        {

            //std::cout<<std::endl<<"Level = "<<level<<"  interval = "<<i;
            switch(bc.first)
            {
                case mure::BCType::dirichlet:
                {
                    //std::cout<<std::endl<<"Dir value = "<<bc.second;
                    field(level, i) = bc.second;
                    break;
                }
                case mure::BCType::neumann:
                {
                    int n = stencil[0];

                    auto mesh = field.mesh();

                    double dx_level = (1 << (mesh.max_level() - level)) * dx();

                    field(level, i) = n*dx_level*bc.second + field(level, i - stencil[0]);

                    // field(level, i) = n*dx()*bc.second + field(level, i - stencil[0]);


                    //std::cout<<std::endl<<"Level "<<level<<" interval"<<i<<" value "<<field(level, i - stencil[0]);
                    break;
                }
            }
        }

        template<class T, class BC, class stencil_t>
        inline void operator()(Dim<2>, T &field, const BC &bc, const stencil_t& stencil) const
        {
            switch(bc.first)
            {
                case mure::BCType::dirichlet:
                {
                    //std::cout<<std::endl<<"Dirichlet";

                    field(level, i, j) = bc.second;
                    break;
                }
                case mure::BCType::neumann:
                {
                    //std::cout<<std::endl<<"Neumann";
                    //std::cout<<std::endl<<"k = "<<i<<" h = "<<j;

                    //std::cout<<std::endl<<"Neumann - Level "<<level<<" k = "<<i<<" h = "<<j<<" Goes for "<<i - stencil[0]<<" and "<<j - stencil[1]<<std::flush;
                    int n = stencil[0] + stencil[1]; // Think about for diagonals, but it is true that here we do not have any normal vector.
                    field(level, i, j) = n*dx()*bc.second + field(level, i - stencil[0], j - stencil[1]);

                    

                    break;
                }
                case mure::BCType::interpolation:
                {
                    int offset_x = (stencil[0] > 0) ? -1 : ((stencil[0] < 0) ? 1 : 0);
                    int offset_y = (stencil[1] > 0) ? -1 : ((stencil[1] < 0) ? 1 : 0);

                    //std::cout<<std::endl<<"Interpolation";

                    field(level, i, j) = 2.0 * field(level, i - stencil[0], j - stencil[1])
                                       - 1.0 * field(level, i - stencil[0] + offset_x, j - stencil[1] + offset_y);
                    break;
                }
            }
        }
    };

    template<class T, class BC, class stencil_t>
    inline auto update_boundary(T &&field, BC &&bc, stencil_t &&stencil)
    {
        return make_field_operator_function<update_boundary_op>(
            std::forward<T>(field), std::forward<BC>(bc), std::forward<stencil_t>(stencil));
    }


    template<class TInterval>
    class update_boundary_testD1Q2_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(update_boundary_testD1Q2_op)

        template<class T, class stencil_t>
        inline void operator()(Dim<1>, T &field, const stencil_t& stencil) const
        {

            double value_dirichlet = 0.5;
            // A
            // field(level, i) = field(level, i - stencil[0]);

            // // B
            // auto max_level = (field.mesh()).max_level();
            // double pref = static_cast<double>(1 << (max_level - level));


            // if (stencil[0] == -1)   {
            //     double to_enforce = 0.5 - field(1, level, i - stencil[0])[0];
            //     double small_cells_regular = field(0, level, i - stencil[0])[0];

            //     field(0, level, i) =  ((pref - 1.)* small_cells_regular + to_enforce) / pref;

            //     field(1, level, i) = field(1, level, i - stencil[0]); // f-
            // }
            // else
            // {
            //     field(level, i) = field(level, i - stencil[0]);
            // }
            
            // C
            // double to_enforce = 0.5 - field(1, level, i - stencil[0])[0];
            // field(0, level, i) = to_enforce;
            // field(1, level, i) = field(1, level, i - stencil[0]); // f-

            // D
            // auto max_level = (field.mesh()).max_level();
            // double pref = static_cast<double>(1 << (max_level - level));

            // if (stencil[0] == -1)   {
            //     double to_enforce = value_dirichlet - field(1, level, i + 1)[0];
            //     double small_cells_regular = field(0, level, i + 1)[0];

            //     field(0, level, i) = ((pref - 1.)* small_cells_regular + to_enforce) / pref;

            //     field(1, level, i) = field(1, level, i - stencil[0]); // f-
            // }
            // else // farmost left ghost
            // {
            //     field(level, i) = field(level, i + 1);
            // }

            // // E
            // auto max_level = (field.mesh()).max_level();
            // double pref = static_cast<double>(1 << (max_level - level));


            // if (stencil[0] == -1)   {
            //     double to_enforce = value_dirichlet - field(1, level, i - stencil[0])[0];
            //     double small_cells_regular = field(0, level, i - stencil[0])[0];

            //     field(0, level, i) =  0.5 * (1. - pref) * small_cells_regular + 0.5 * (1. + pref) * to_enforce;

            //     field(1, level, i) = 2.* field(1, level, i + 1) - field(1, level, i + 2);
            // } 
            // else
            // {
            //     double to_enforce = value_dirichlet - field(1, level, i + 2)[0];
            //     double small_cells_regular = field(0, level, i + 2)[0];


            //     field(0, level, i) =  0.5 * (1. - 3. * pref) * small_cells_regular + 0.5 * (1. + 3. * pref) * to_enforce;

            //     field(1, level, i) = 3.* field(1, level, i + 2) - 2. * field(1, level, i + 3);            
            // }

            // F
            auto max_level = (field.mesh()).max_level();
            double pref = static_cast<double>(1 << (max_level - level));


            if (stencil[0] == -1)   {
                double to_enforce = value_dirichlet - field(1, level, i - stencil[0])[0];

                double fj0 = field(0, level, i + 1)[0];
                double fj1 = field(0, level, i + 2)[0];

                field(0, level, i) =  to_enforce;

                for (int k = -2; k >= -(1<<(max_level - level)); --k){
                    field(0, level, i) += (fj1 - fj0)/pref * (static_cast<double>(k) + 0.5) + 0.5 * (3. * fj0 - fj1);
                }
                field(0, level, i) = field(0, level, i) / pref;

                field(1, level, i) = 2.* field(1, level, i + 1) - field(1, level, i + 2);
            } 
            else
            {
                field(0, level, i) = 3.* field(0, level, i + 2) - 2. * field(0, level, i + 3);       
                field(1, level, i) = 3.* field(1, level, i + 2) - 2. * field(1, level, i + 3);              
       
            }


            
        }
    };


    template<class T, class stencil_t>
    inline auto update_boundary_testD1Q2(T &&field, stencil_t &&stencil)
    {
        return make_field_operator_function<update_boundary_testD1Q2_op>(
            std::forward<T>(field), std::forward<stencil_t>(stencil));
    }




    template<class TInterval>
    class update_boundary_D2Q4_flat_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(update_boundary_D2Q4_flat_op)

        template<class T, class stencil_t>
        inline void operator()(Dim<2>, T &field, const stencil_t& stencil) const
        {

            field(level, i, j) = field(level, i - stencil[0], j - stencil[1]);
            
        }
    };


    template<class T, class stencil_t>
    inline auto update_boundary_D2Q4_flat(T &&field, stencil_t &&stencil)
    {
        return make_field_operator_function<update_boundary_D2Q4_flat_op>(
            std::forward<T>(field), std::forward<stencil_t>(stencil));
    }

    template<class TInterval>
    class update_boundary_D2Q4_linear_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(update_boundary_D2Q4_linear_op)

        template<class T, class stencil_t>
        inline void operator()(Dim<2>, T &field, const stencil_t& stencil) const
        {
            // Only works along the axis
            field(level, i, j) = 2. * field(level, i - stencil[0], j - stencil[1]) - 1. * field(level, i - 2*stencil[0], j - 2*stencil[1]);
            
        }
    };


    template<class T, class stencil_t>
    inline auto update_boundary_D2Q4_linear(T &&field, stencil_t &&stencil)
    {
        return make_field_operator_function<update_boundary_D2Q4_linear_op>(
            std::forward<T>(field), std::forward<stencil_t>(stencil));
    }




    template<class TInterval>
    class update_boundary_D2Q9_lid_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(update_boundary_D2Q9_lid_op)

        template<class T>
        inline void operator()(Dim<2>, T &field) const
        {

            
            double rho0 = 1.;
            double u0 = 0.05;
            double lambda = 1.;


            double rho = rho0;
            double qx = rho * u0;
            double qy = 0.;

            double cs2 = (lambda*lambda)/ 3.0; // Sound velocity of the lattice squared

            double m0 = rho;
            double m1 = qx;
            double m2 = qy;
            double m3 = (qx*qx+qy*qy)/rho + 2.*rho*cs2;
            double m4 = qx*(cs2+(qy/rho)*(qy/rho));
            double m5 = qy*(cs2+(qx/rho)*(qx/rho));
            double m6 = rho*(cs2+(qx/rho)*(qx/rho))*(cs2+(qy/rho)*(qy/rho));
            double m7 = (qx*qx-qy*qy)/rho;
            double m8 = qx*qy/rho;

            // We come back to the distributions

            double r1 = 1.0 / lambda;
            double r2 = 1.0 / (lambda*lambda);
            double r3 = 1.0 / (lambda*lambda*lambda);
            double r4 = 1.0 / (lambda*lambda*lambda*lambda);

            field(0, level, i, j) = m0                      -     r2*m3                        +     r4*m6                         ;
            field(1, level, i, j) =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
            field(2, level, i, j) =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
            field(3, level, i, j) =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
            field(4, level, i, j) =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
            field(5, level, i, j) =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
            field(6, level, i, j) =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
            field(7, level, i, j) =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
            field(8, level, i, j) =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
   

            // std::cout<<std::endl<<"Updating lid = "<<field(0, level, i, j)<<std::flush;

        }
    };


    template<class T>
    inline auto update_boundary_D2Q9_lid(T &&field)
    {
        return make_field_operator_function<update_boundary_D2Q9_lid_op>(std::forward<T>(field));
    }



    template<class TInterval>
    class update_boundary_d2q9_KH_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(update_boundary_d2q9_KH_op)

        template<class T, class stencil_t>
        inline void operator()(Dim<2>, T &field, const stencil_t& stencil) const
        {
            //std::cout<<std::endl<<"Level "<<level<<"  k = "<<i<<" h = "<<j<<std::flush;
            int n = stencil[0] + stencil[1]; // Think about for diagonals, but it is true that here we do not have any normal vector.
            field(level, i, j) = field(level, i - stencil[0], j - stencil[1]);
                
        }
    };

    template<class T, class stencil_t>
    inline auto update_boundary_D2Q9_KH(T &&field, stencil_t &&stencil)
    {
        return make_field_operator_function<update_boundary_d2q9_KH_op>(
            std::forward<T>(field), std::forward<stencil_t>(stencil));
    }

    template<std::size_t size>
    struct is_scalar_field: std::false_type
    {};

    template<>
    struct is_scalar_field<1>: std::true_type
    {};

    template<class MRConfig, class value_t=double, std::size_t size_=1>
    class Field : public field_expression<Field<MRConfig, value_t, size_>>
    {
      public:

        using Config = MRConfig;
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        using coord_index_t = typename MRConfig::coord_index_t;
        using index_t = typename MRConfig::index_t;
        using interval_t = typename MRConfig::interval_t;

        using value_type = value_t;
        static constexpr auto size = size_;
        using data_type = typename std::conditional<is_scalar_field<size>::value,
                                                    xt::xtensor<value_type, 1>,
                                                    xt::xtensor<value_type, 2>>::type;

        template<std::size_t n>
        void init_data(std::integral_constant<std::size_t, n>)
        {
            m_data.resize({m_mesh->nb_total_cells(), size});
            m_data.fill(0);
        }

        void init_data(std::integral_constant<std::size_t, 1>)
        {
            m_data.resize({m_mesh->nb_total_cells()});
            m_data.fill(0);
        }

        inline Field(std::string name, Mesh<MRConfig> &mesh, const BC<dim> &bc)
            : m_name(name), m_mesh(&mesh), m_bc(bc)
        {
            init_data(std::integral_constant<std::size_t, size>{});
        }

        inline Field(std::string name, Mesh<MRConfig> &mesh)
            : m_name(name), m_mesh(&mesh)
        {
            init_data(std::integral_constant<std::size_t, size>{});
        }

        template<class E>
        inline Field &operator=(const field_expression<E> &e)
        {
            // mesh->for_each_cell(
            //     [&](auto &cell) { (*this)[cell] = e.derived_cast()(cell); });

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                auto subset = intersection((*m_mesh)[MeshType::cells][level],
                                           (*m_mesh)[MeshType::cells][level]);

                subset.apply_op(level, apply_expr(*this, e));
            }
            return *this;
        }

        inline auto operator[](const std::size_t index) const
        {
            return xt::view(m_data, index);
        }

        inline auto operator[](const std::size_t index)
        {
            return xt::view(m_data, index);
        }

        inline auto operator[](const Cell<coord_index_t, dim> &cell) const
        {
            return xt::view(m_data, cell.index);
        }

        inline auto operator[](const Cell<coord_index_t, dim> &cell)
        {
            return xt::view(m_data, cell.index);
        }

        template<class... T>
        inline auto operator()(const std::size_t level, const interval_t &interval, const T... index)
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("WRITE FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step));
        }

        template<class... T>
        inline auto operator()(const std::size_t level, const interval_t &interval,
                        const T... index) const
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("READ FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step));
        }

        template<class... T>
        inline auto operator()(const std::size_t item, const std::size_t level, const interval_t &interval, const T... index)
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("WRITE FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step), item);
        }

        template<class... T>
        inline auto operator()(const std::size_t item, const std::size_t level, const interval_t &interval,
                        const T... index) const
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("READ FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step), item);
        }

        template<class... T>
        inline auto operator()(const std::size_t item_s, const std::size_t item_e, const std::size_t level, const interval_t &interval, const T... index)
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("WRITE FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step), xt::range(item_s, item_e));
        }

        template<class... T>
        inline auto operator()(const std::size_t item_s, const std::size_t item_e, const std::size_t level, const interval_t &interval,
                        const T... index) const
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("READ FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step), xt::range(item_s, item_e));
        }


        inline auto data(MeshType mesh_type) const
        {
            std::array<std::size_t, 2> shape = {m_mesh->nb_cells(mesh_type), size};
            xt::xtensor<double, 2> output(shape);
            std::size_t index = 0;
            m_mesh->for_each_cell([&](auto cell)
                                  {
                                      auto view = xt::view(output, index++);
                                      view = xt::view(m_data, cell.index);
                                  },
                                  mesh_type);
            return output;
        }

        inline auto data_on_level(std::size_t level, MeshType mesh_type) const
        {
            std::array<std::size_t, 2> shape = {m_mesh->nb_cells(level, mesh_type), size};
            xt::xtensor<double, 2> output(shape);
            std::size_t index = 0;
            m_mesh->for_each_cell(level,
                                  [&](auto cell)
                                  {
                                      xt::view(output, index++) = xt::view(m_data, cell.index);
                                  },
                                  mesh_type);
            return output;
        }

        inline auto const array() const
        {
            return m_data;
        }

        inline auto &array()
        {
            return m_data;
        }

        inline std::size_t nb_cells(MeshType mesh_type) const
        {
            return m_mesh->nb_cells(mesh_type);
        }

        inline std::size_t nb_cells(std::size_t level, MeshType mesh_type) const
        {
            return m_mesh->nb_cells(level, mesh_type);
        }

        inline auto name() const
        {
            return m_name;
        }

        inline auto bc() const
        {
            return m_bc;
        }


        inline auto bc()
        {
            return m_bc;
        }

        inline auto mesh() const
        {
            return *m_mesh;
        }

        inline auto mesh()
        {
            return *m_mesh;
        }

        inline auto mesh_ptr()
        {
            return m_mesh;
        }

        inline void update_bc_D2Q9_KH()
        {
            // 4 axis
            std::vector<xt::xtensor_fixed<int, xt::xshape<dim>>> versors {{ 1,  0},
                                                                          {-1,  0},
                                                                          { 0,  1},
                                                                          { 0, -1}};


            auto max_level = m_mesh->max_level();                                                             

            for (std::size_t level = 0; level <= max_level; ++level)    {
                if (!(*m_mesh)[mure::MeshType::all_cells][level].empty())   {
                    
                    for (auto versor : versors) {
                        // The order of the operations is important
                        auto subset1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<(max_level - level)) * versor),
                                                              m_mesh->initial_mesh()),
                                                  (*m_mesh)[mure::MeshType::all_cells][level])
                                     .on(level);

                        subset1.apply_op(level, update_boundary_D2Q9_KH(*this, versor));


                        auto subset2 = intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<(max_level - level)) * versor),
                                                              m_mesh->initial_mesh()),
                                                  (*m_mesh)[mure::MeshType::all_cells][level])
                                     .on(level);

                        subset2.apply_op(level, update_boundary_D2Q9_KH(*this, versor));

                    }

                    // Corners                    
                    // Parallel to the axis
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_p0{ 1,  0};
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_m0{-1,  0};
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_0p{ 0,  1};
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_0m{ 0, -1};

                    // Diagonally
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pp{ 1,  1};
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pm{ 1, -1};
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mp{-1,  1};
                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mm{-1, -1};

                    std::pair<BCType, double> cond {BCType::neumann, 0.0};

                    // //std::cout<<std::endl<<"Level "<<level<<std::flush;

                    // auto subset_pp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_pp), m_mesh->initial_mesh()), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_p0)), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0p)),
                    //                              (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    // subset_pp.apply_op(level, update_boundary(*this, cond, 2*stencil_pp));


                    // auto subset_pm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_pm), m_mesh->initial_mesh()), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_p0)), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0m)),
                    //                              (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    // subset_pm.apply_op(level, update_boundary(*this, cond, 2*stencil_pm));

                    
                    // auto subset_mp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_mp), m_mesh->initial_mesh()), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_m0)), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0p)),
                    //                              (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    // subset_mp.apply_op(level, update_boundary(*this, cond, 2*stencil_mp));

                    
                    // auto subset_mm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_mm), m_mesh->initial_mesh()), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_m0)), 
                    //                                                     translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0m)),
                    //                              (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    // subset_mm.apply_op(level, update_boundary(*this, cond, 2*stencil_mm));     



                    
                   auto subset_pp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_pp), m_mesh->initial_mesh()), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_p0)), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0p)),
                                                 (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    subset_pp.apply_op(level, update_boundary(*this, cond, stencil_pp));


                    auto subset_pm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_pm), m_mesh->initial_mesh()), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_p0)), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0m)),
                                                 (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    subset_pm.apply_op(level, update_boundary(*this, cond, stencil_pm));

                    
                    auto subset_mp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_mp), m_mesh->initial_mesh()), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_m0)), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0p)),
                                                 (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    subset_mp.apply_op(level, update_boundary(*this, cond, stencil_mp));

                    
                    auto subset_mm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_mm), m_mesh->initial_mesh()), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_m0)), 
                                                                        translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_0m)),
                                                 (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

                    subset_mm.apply_op(level, update_boundary(*this, cond, stencil_mm));   


                    // We still have to update the faremost corners
                    //std::cout<<std::endl<<std::endl<<"Foremost corners"<<std::flush;
                    {
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pp_21{ 2,  1};
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pp_12{ 1,  2};

                        auto farmost_pp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_pp), m_mesh->initial_mesh()),
                                                                                        translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_pp_21)),
                                                                             translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_pp_12)), 
                                                       (*m_mesh)[mure::MeshType::all_cells][level])
                                             .on(level);

                        farmost_pp.apply_op(level, update_boundary(*this, cond, stencil_pp));   
                    }

                    {
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pm_21{ 2,  -1};
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pm_12{ 1,  -2};

                        auto farmost_pm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_pm), m_mesh->initial_mesh()),
                                                                                        translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_pm_21)),
                                                                             translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_pm_12)), 
                                                       (*m_mesh)[mure::MeshType::all_cells][level])
                                             .on(level);

                        farmost_pm.apply_op(level, update_boundary(*this, cond, stencil_pm));   
                    }

                    {
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mp_21{ -2,  1};
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mp_12{ -1,  2};

                        auto farmost_mp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_mp), m_mesh->initial_mesh()),
                                                                                        translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_mp_21)),
                                                                             translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_mp_12)), 
                                                       (*m_mesh)[mure::MeshType::all_cells][level])
                                             .on(level);

                        farmost_mp.apply_op(level, update_boundary(*this, cond, stencil_mp)); 

      
                    }
                    
                    {
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mm_21{ -2, -1};
                        xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mm_12{ -1, -2};
                        auto farmost_mm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), 2*(1<<(max_level-level))*stencil_mm), m_mesh->initial_mesh()),
                                                                                        translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_mm_21)),
                                                                             translate(m_mesh->initial_mesh(), (1<<(max_level-level))*stencil_mm_12)), 
                                                       (*m_mesh)[mure::MeshType::all_cells][level])
                                             .on(level);

                        farmost_mm.apply_op(level, update_boundary(*this, cond, stencil_mm)); 
                    }

                }
            }


        }



        inline void update_bc_D2Q4_3_Euler_constant_extension()
        {
        
            const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
            const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

            const xt::xtensor_fixed<int, xt::xshape<2>> pp{1, 1};
            const xt::xtensor_fixed<int, xt::xshape<2>> pm{1, -1};

            size_t max_level = m_mesh->max_level();

            for (std::size_t level = 0; level <= max_level; ++level)  {

                size_t j = max_level - level;


                // E first rank (not projected on the level for future use)
                auto east_1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<j) * xp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // E second rank
                auto east_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * xp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         east_1);
                // The order is important becase the second rank shall take the values stored in the first rank
                east_1.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, xp));
                east_2.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, xp));// By not multiplying by 2 it takes the values in the first rank


                // W first rank
                auto west_1 = intersection(difference(translate(m_mesh->initial_mesh(), (-1) * (1<<j) * xp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // W second rank
                auto west_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (-1) * (1<<j) * xp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         west_1);
                west_1.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, (-1) * xp));
                west_2.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, (-1) * xp));

                // N first rank
                auto north_1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<j) * yp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);

                // N second rank
                auto north_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * yp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         north_1);
                north_1.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, yp));
                north_2.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, yp));

                // S first rank
                auto south_1 = intersection(difference(translate(m_mesh->initial_mesh(), (-1) * (1<<j) * yp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // S second rank
                auto south_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (-1) * (1<<j) * yp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         south_1);
                south_1.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, (-1) * yp));
                south_2.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, (-1) * yp));



                auto east  = union_(east_1,  east_2);
                auto west  = union_(west_1,  west_2);
                auto north = union_(north_1, north_2);
                auto south = union_(south_1, south_2);

               
                auto north_east = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * pp), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(east, north));

                north_east.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * pp)); // Come back inside


            
     

                auto south_east = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * pm), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                             union_(east, south));

                south_east.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * pm)); // Come back inside





                auto north_west = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 *  (-1) * (1<<j) * pm), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(west, north));

                
                north_west.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * (-1) * pm)); // Come back inside


                auto south_west = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 *  (-1) * (1<<j) * pp), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(south, west));

                south_west.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * (-1) * pp)); // Come back inside



            }

        }



        inline void update_bc_D2Q4_3_Euler_linear_extension(std::size_t ite = 0)
        {
        
            const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
            const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

            const xt::xtensor_fixed<int, xt::xshape<2>> pp{1, 1};
            const xt::xtensor_fixed<int, xt::xshape<2>> pm{1, -1};

            size_t max_level = m_mesh->max_level();

            // for (std::size_t level = 0; level <= max_level; ++level)  {
            for (std::size_t level = 0; level <= max_level - 1 - ite; ++level)  {

                size_t j = max_level - level;


                // E first rank (not projected on the level for future use)
                auto east_1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<j) * xp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // E second rank
                auto east_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * xp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         east_1);
                // The order is important becase the second rank shall take the values stored in the first rank
                east_1.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, xp));
                east_2.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, xp));// By not multiplying by 2 it takes the values in the first rank


                // W first rank
                auto west_1 = intersection(difference(translate(m_mesh->initial_mesh(), (-1) * (1<<j) * xp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // W second rank
                auto west_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (-1) * (1<<j) * xp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         west_1);
                west_1.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * xp));
                west_2.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * xp));

                // N first rank
                auto north_1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<j) * yp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);

                // N second rank
                auto north_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * yp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         north_1);
                north_1.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, yp));
                north_2.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, yp));

                // S first rank
                auto south_1 = intersection(difference(translate(m_mesh->initial_mesh(), (-1) * (1<<j) * yp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // S second rank
                auto south_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (-1) * (1<<j) * yp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         south_1);
                south_1.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * yp));
                south_2.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * yp));



                auto east  = union_(east_1,  east_2);
                auto west  = union_(west_1,  west_2);
                auto north = union_(north_1, north_2);
                auto south = union_(south_1, south_2);

               
                auto north_east = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * pp), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(east, north));

                north_east.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * pp)); // Come back inside


            
     

                auto south_east = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * pm), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                             union_(east, south));

                south_east.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * pm)); // Come back inside





                auto north_west = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 *  (-1) * (1<<j) * pm), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(west, north));

                
                north_west.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * (-1) * pm)); // Come back inside


                auto south_west = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 *  (-1) * (1<<j) * pp), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(south, west));

                south_west.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * (-1) * pp)); // Come back inside



            }

        }



        inline void update_bc_D2Q9_lid_driven_cavity(std::size_t ite = 0)
        {
        
            const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
            const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

            const xt::xtensor_fixed<int, xt::xshape<2>> pp{1, 1};
            const xt::xtensor_fixed<int, xt::xshape<2>> pm{1, -1};

            size_t max_level = m_mesh->max_level();

            // for (std::size_t level = 0; level <= max_level; ++level)  {
            // for (std::size_t level = 0; level <= max_level - 1 - ite; ++level)  {
            for (std::size_t level = 0; level <= max_level - ite; ++level)  {

                size_t j = max_level - level;


                // E first rank (not projected on the level for future use)
                auto east_1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<j) * xp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // E second rank
                auto east_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * xp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         east_1);
                // The order is important becase the second rank shall take the values stored in the first rank
                east_1.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, xp));
                east_2.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, xp));// By not multiplying by 2 it takes the values in the first rank


                // W first rank
                auto west_1 = intersection(difference(translate(m_mesh->initial_mesh(), (-1) * (1<<j) * xp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // W second rank
                auto west_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (-1) * (1<<j) * xp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         west_1);
                west_1.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * xp));
                west_2.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * xp));

                // N first rank
                auto north_1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<j) * yp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);

                // N second rank
                auto north_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * yp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         north_1);
                north_1.on(level).apply_op(level, update_boundary_D2Q9_lid(*this));
                north_2.on(level).apply_op(level, update_boundary_D2Q9_lid(*this));

                // S first rank
                auto south_1 = intersection(difference(translate(m_mesh->initial_mesh(), (-1) * (1<<j) * yp), 
                                                      m_mesh->initial_mesh()), 
                                           (*m_mesh)[mure::MeshType::all_cells][level]);
                
                // S second rank
                auto south_2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (-1) * (1<<j) * yp), 
                                                                 m_mesh->initial_mesh()), 
                                                      (*m_mesh)[mure::MeshType::all_cells][level]), 
                                         south_1);
                south_1.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * yp));
                south_2.on(level).apply_op(level, update_boundary_D2Q4_linear(*this, (-1) * yp));



                auto east  = union_(east_1,  east_2);
                auto west  = union_(west_1,  west_2);
                auto north = union_(north_1, north_2);
                auto south = union_(south_1, south_2);

               
                auto north_east = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * pp), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(east, north));

                north_east.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * pp)); // Come back inside


            
     

                auto south_east = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<j) * pm), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                             union_(east, south));

                south_east.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * pm)); // Come back inside





                auto north_west = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 *  (-1) * (1<<j) * pm), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(west, north));

                
                north_west.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * (-1) * pm)); // Come back inside


                auto south_west = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 *  (-1) * (1<<j) * pp), 
                                                                     m_mesh->initial_mesh()), 
                                                         (*m_mesh)[mure::MeshType::all_cells][level]), 
                                            union_(south, west));

                south_west.on(level).apply_op(level, update_boundary_D2Q4_flat(*this, 2 * (-1) * pp)); // Come back inside



            }

        }


        inline void update_bc_D2Q4_3_Euler()
        {

            // std::cout<<std::endl<<"Update BC"<<std::flush;

            // std::cout << m_mesh->initial_mesh() << "\n";
            for (std::size_t level = m_mesh->min_level()-1; level <= m_mesh->max_level(); ++level)
            {
                double length = 1./(1 << level);

                auto to_set = difference((*m_mesh)[mure::MeshType::all_cells][level], m_mesh->initial_mesh()).on(level);

                to_set([&](auto& index, auto &interval, auto) {
                    auto k = interval[0]; 
                    auto h_idx = index[0];

                    // std::cout<<std::endl<<"Level = "<<level<<" Interval "<<k<<" h = "<<h_idx<<std::flush;

                    for (auto k_idx = k.start; k_idx < k.end; ++k_idx)  {
                        double x = length * (k_idx + 0.5);
                        double y = length * (h_idx + 0.5);

                        double rho = 1.0; // Density
                        double qx = 0.0; // x-momentum
                        double qy = 0.0; // y-momentum
                        double e = 0.0;

                        double gm = 1.4;

                        if (x < 0.5)    {
                            if (y < 0.5)    {
                                // 3
                                rho = 1.0;
                                qx = rho * 0.75;
                                qy = rho * 0.5;
                                double p = 1.0;
                                e = p / (gm - 1.) + 0.5 * (qx*qx + qy*qy) / rho;

                            }
                            else
                            {
                                // 2   
                                rho = 2.0;
                                qx = rho * (-0.75);
                                qy = rho * (0.5);
                                double p = 1.0;
                                e = p / (gm - 1.) + 0.5 * (qx*qx + qy*qy) / rho;            
                            }
                        }
                        else
                        {
                            if (y < 0.5)    {
                                // 4
                                rho = 3.0;
                                qx = rho * (0.75);
                                qy = rho * (-0.5);
                                double p = 1.0;
                                e = p / (gm - 1.) + 0.5 * (qx*qx + qy*qy) / rho;     
                            }
                            else
                            {
                                // 1
                                rho = 1.0;
                                qx = rho * (-0.75);
                                qy = rho * (-0.5);
                                double p = 1.0;
                                e = p / (gm - 1.) + 0.5 * (qx*qx + qy*qy) / rho;     
                            }
                        }

                        // Conserved momenti
                        double m0_0 = rho;
                        double m1_0 = qx;
                        double m2_0 = qy;
                        double m3_0 = e;

                        // Non conserved at equilibrium
                        double m0_1 = m1_0;
                        double m0_2 = m2_0;
                        double m0_3 = 0.0;

                        double m1_1 =     (3./2. - gm/2.) * (m1_0*m1_0)/(m0_0)
                                        + (1./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (gm - 1.) * m3_0;
                        double m1_2 = m1_0*m2_0/m0_0;
                        double m1_3 = 0.0;

                        double m2_1 = m1_0*m2_0/m0_0;

                        double m2_2 =     (3./2. - gm/2.) * (m2_0*m2_0)/(m0_0)
                                        + (1./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (gm - 1.) * m3_0;
                        double m2_3 = 0.0;

                        double m3_1 = gm*(m1_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m1_0*m1_0*m1_0)/(m0_0*m0_0) + (gm/2. - 1./2.)*(m1_0*m2_0*m2_0)/(m0_0*m0_0);
                        double m3_2 = gm*(m2_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m2_0*m2_0*m2_0)/(m0_0*m0_0) + (gm/2. - 1./2.)*(m2_0*m1_0*m1_0)/(m0_0*m0_0);
                        double m3_3 = 0.0;

                        double lambda = 4.;

                        // We come back to the distributions
                        (*this)(0, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                        (*this)(1, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
                        (*this)(2, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                        (*this)(3, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
                        (*this)(4, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m1_0 + .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
                        (*this)(5, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m1_0                    + .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;
                        (*this)(6, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m1_0 - .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
                        (*this)(7, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m1_0                    - .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;
                        (*this)(8,  level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m2_0 + .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
                        (*this)(9,  level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m2_0                    + .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;
                        (*this)(10, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m2_0 - .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
                        (*this)(11, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m2_0                    - .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;
                        (*this)(12, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m3_0 + .5/lambda * (m3_1)                    + .25/(lambda*lambda) * m3_3;
                        (*this)(13, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m3_0                    + .5/lambda * (m3_2) - .25/(lambda*lambda) * m3_3;
                        (*this)(14, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m3_0 - .5/lambda * (m3_1)                    + .25/(lambda*lambda) * m3_3;
                        (*this)(15, level, interval_t{k_idx, k_idx + 1}, h_idx) = .25 * m3_0                    - .5/lambda * (m3_2) - .25/(lambda*lambda) * m3_3;

                    }     


                });

            }

        }

        inline void update_bc(std::size_t ite = 0)
        {
            
            // update_bc_D2Q4_3_Euler_constant_extension();
            update_bc_D2Q4_3_Euler_linear_extension(ite); // Works properly

            // update_bc_D2Q4_3_Euler();

            // update_bc_D2Q9_lid_driven_cavity(ite);

            return;

            // update_bc_D2Q9_KH();


            // // A / B
            // for (std::size_t level = 0; level <= m_mesh->max_level(); ++level)
            // {
            //     xt::xtensor_fixed<int, xt::xshape<1>> stencil{-1};

            //     if (!(*m_mesh)[mure::MeshType::all_cells][level].empty())   {

            //         auto ghost_m1 = intersection(difference(translate(m_mesh->initial_mesh(), (1<<(m_mesh->max_level() - level)) * stencil),
            //                             m_mesh->initial_mesh()),
            //                  (*m_mesh)[mure::MeshType::all_cells][level]);

            //         ghost_m1.on(level).apply_op(level, update_boundary_testD1Q2(*this, stencil));

                    // auto ghost_m2 = difference(intersection(difference(translate(m_mesh->initial_mesh(), 2 * (1<<(m_mesh->max_level() - level)) * stencil),
                    //                     m_mesh->initial_mesh()),
                    //          (*m_mesh)[mure::MeshType::all_cells][level]), ghost_m1)
                    //                         .on(level);

            //         ghost_m2.apply_op(level, update_boundary_testD1Q2(*this, 2*stencil));

            //     }
            // }

            










            // return; // Just to exit here


            // for (std::size_t level = 0; level <= m_mesh->max_level(); ++level)
            // {
            //     if (!(*m_mesh)[mure::MeshType::all_cells][level].empty())   {

            //         mure::static_nested_loop<dim, -2, 3>(
            //             [&](auto stencil) {

            //             auto subset = intersection(difference(translate(m_mesh->initial_mesh(), stencil),
            //                             m_mesh->initial_mesh()),
            //                  (*m_mesh)[mure::MeshType::all_cells][level])
            //                                 .on(level);

            //             subset.apply_op(level, update_boundary(*this, m_bc.type[0], stencil));

            //         });

            //     }
            // }


            //std::cout<<std::endl<<std::endl;


            // xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
            // std::size_t index_bc = 0;
            // for (std::size_t d = 0; d < dim; ++d)
            // {

            //     for (std::size_t d1 = 0; d1 < dim; ++d1)
            //         stencil[d1] = 0;
                
            //     int gw = static_cast<int>(m_mesh->ghost_width);

            //     //for (int s = -gw; s <= gw; ++s)
            //     for (int s = -1; s <= 1; ++s)
            //     {
            //         if (s != 0)
            //         {
            //             stencil[d] = s;

            //             for (std::size_t level = 0; level <= m_mesh->max_level(); ++level)
            //             {

            //                 double dx = 1./(1<<level);
            //                 if (!(*m_mesh)[mure::MeshType::all_cells][level].empty())
            //                 {


            //                     // TODO: use union mesh instead
            //                     // auto subset = intersection(difference(translate(m_mesh->initial_mesh(), stencil),
            //                     //                                       m_mesh->initial_mesh()),
            //                     //                            (*m_mesh)[mure::MeshType::all_cells][level])
            //                     //             .on(level);

            //                     int factor = 1 << (m_mesh->max_level() - level);
            //                     // std::cout<<std::endl<<"Factor = "<<factor<<" Stencil "<<stencil;

            //                     // auto originalmesh = intersection(m_mesh->initial_mesh(), m_mesh->initial_mesh());
            //                     // originalmesh([&](auto, auto &interval, auto) {
            //                     //     auto i = interval[0];
            //                     //     std::cout<<std::endl<<"Originalmesh  "<<i;
            //                     // });

            //                     // auto originaltranslated = intersection(difference(translate(m_mesh->initial_mesh(), factor * stencil),
            //                     //                                       m_mesh->initial_mesh()).on(level), (*m_mesh)[mure::MeshType::cells_and_ghosts][level]);
            //                     // originaltranslated([&](auto, auto &interval, auto) {
            //                     //     auto i = interval[0];
            //                     //     std::cout<<std::endl<<"OriginalTranslated "<<i;
            //                     // });

            //                     //std::cout<<std::endl<<"Stencil "<<stencil[0]<<", "<<stencil[1]<<std::endl;

            //                     auto subset = intersection(difference(translate(m_mesh->initial_mesh(), stencil),
            //                                                           m_mesh->initial_mesh()),
            //                                                (*m_mesh)[mure::MeshType::all_cells][level])
            //                                 .on(level);

            //                     // subset([&](auto, auto &interval, auto) {
            //                     //     auto i = interval[0];
            //                     //     std::cout<<std::endl<<"We apply at level "<<level<<" for "<<i;
            //                     // });

            //                     // A changer ... c'est crade
            //                     subset.apply_op(level, update_boundary(*this, m_bc.type[0], stencil));
            //                 }
            //             }
            //             // Cochonnerie a changer ...
            //             if (s%2 != 0)
            //                 index_bc++;
            //         }
            //     }
            // }

            // // Integrate the diagonal velocity in what we have previously coded
            // // yields strange results which are not good. I do this by hand
            // // for the D2Q9.
            // for (std::size_t level = 0; level <= m_mesh->max_level(); ++level)
            // {
            //     double dx = 1./(1<<level);
            //     if (!(*m_mesh)[mure::MeshType::all_cells][level].empty())
            //     {
            //         // Parallel to the axis
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_p0{ 1,  0};
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_m0{-1,  0};
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_0p{ 0,  1};
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_0m{ 0, -1};

            //         // Diagonally
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pp{ 1,  1};
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_pm{ 1, -1};
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mp{-1,  1};
            //         xt::xtensor_fixed<int, xt::xshape<dim>> stencil_mm{-1, -1};

            //         std::pair<BCType, double> cond {BCType::neumann, 0.0};

            //         auto subset_pp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), stencil_pp), m_mesh->initial_mesh()), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_p0)), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_0p)),
            //                                      (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

            //         subset_pp.apply_op(level, update_boundary(*this, cond, stencil_pp));


            //         auto subset_pm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), stencil_pm), m_mesh->initial_mesh()), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_p0)), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_0m)),
            //                                      (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

            //         subset_pm.apply_op(level, update_boundary(*this, cond, stencil_pm));

                    
            //         auto subset_mp = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), stencil_mp), m_mesh->initial_mesh()), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_m0)), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_0p)),
            //                                      (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

            //         subset_mp.apply_op(level, update_boundary(*this, cond, stencil_mp));

                    
            //         auto subset_mm = intersection(difference(difference(difference(translate(m_mesh->initial_mesh(), stencil_mm), m_mesh->initial_mesh()), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_m0)), 
            //                                                             translate(m_mesh->initial_mesh(), stencil_0m)),
            //                                      (*m_mesh)[mure::MeshType::all_cells][level]).on(level);

            //         subset_mm.apply_op(level, update_boundary(*this, cond, stencil_mm));
            //     }
            // }

        }

        inline void to_stream(std::ostream &os) const
        {
            os << "Field " << m_name << "\n";
            m_mesh->for_each_cell(
                [&](auto &cell) {
                    os << cell.level << "[" << cell.center()
                       << "]:" << xt::view(m_data, cell.index) << "\n";
                },
                // MeshType::all_cells);
                MeshType::cells);
        }

      private:
        std::string m_name;
        Mesh<MRConfig> *m_mesh;
        BC<dim> m_bc;
        data_type m_data;
    };

    template<class MRConfig, class T, std::size_t N>
    inline std::ostream &operator<<(std::ostream &out, const Field<MRConfig, T, N> &field)
    {
        field.to_stream(out);
        return out;
    }
}