#pragma once
#include <algorithm>

#include <xtensor/xtensor.hpp>

namespace samurai
{
    template <class value_t, std::size_t size>
    struct point_value
    {
        using type = xt::xtensor_fixed<value_t, xt::xshape<size>>;
    };

    template <class value_t>
    struct point_value<value_t, 1>
    {
        using type = value_t;
    };

    template <class value_t, std::size_t dim, std::size_t size>
    class BoundaryCondition
    {
      public:

        using boundary_point_t = xt::xtensor_fixed<double, xt::xshape<dim>>;
        using boundary_value_t = typename point_value<value_t, size>::type; // if size = 1 --> 'double', else
                                                                            // -->
                                                                            // 'xt::xtensor_fixed<value_t,
                                                                            // xt::xshape<size>>'
        using boundary_part_t = std::function<bool(const boundary_point_t&)>;
        using boundary_cond_t = std::function<boundary_value_t(const boundary_point_t&)>;

        enum BCType : int
        {
            Dirichlet,
            Neumann
        };

      private:

        BCType _bc_type                = BCType::Dirichlet;
        boundary_part_t _boundary_part = nullptr;
        boundary_cond_t _boundary_cond = nullptr;

      public:

        BoundaryCondition(BCType bc_type, boundary_cond_t boundary_cond)
            : _bc_type(bc_type)
            , _boundary_part(_everywhere)
            , _boundary_cond(boundary_cond)
        {
        }

        bool is_dirichlet() const
        {
            return _bc_type == Dirichlet;
        }

        bool is_neumann() const
        {
            return _bc_type == Neumann;
        }

        bool applies_to(const boundary_point_t& boundary_point) const
        {
            return _boundary_part(boundary_point);
        }

        boundary_value_t get_value(const boundary_point_t& boundary_point) const
        {
            return _boundary_cond(boundary_point);
        }

        BoundaryCondition& where(boundary_part_t boundary_part)
        {
            _boundary_part = boundary_part;
            return *this;
        }

        BoundaryCondition& everywhere()
        {
            _boundary_part = _everywhere;
            return *this;
        }

        BoundaryCondition& set_condition(boundary_cond_t boundary_cond)
        {
            _boundary_cond = boundary_cond;
            return *this;
        }

      private:

        static bool _everywhere(const boundary_point_t&)
        {
            return true;
        }
    };

    template <class value_t, std::size_t dim, std::size_t size>
    const BoundaryCondition<value_t, dim, size>& find(const std::vector<BoundaryCondition<value_t, dim, size>>& boundary_conditions,
                                                      const typename BoundaryCondition<value_t, dim, size>::boundary_point_t& boundary_point)
    {
        auto bc = std::find_if(boundary_conditions.cbegin(),
                               boundary_conditions.cend(),
                               [&boundary_point](const auto& bc)
                               {
                                   return bc.applies_to(boundary_point);
                               });

        if (bc != boundary_conditions.cend())
        {
            return *bc;
        }
        std::cout << "No boundary condition found for the point of coordinates " << boundary_point << std::endl;
        assert(false && "No boundary condition found for this point");
    }

    template <class value_t, std::size_t dim, std::size_t size>
    bool has_neumann(const std::vector<BoundaryCondition<value_t, dim, size>>& boundary_conditions)
    {
        auto it = std::find_if(boundary_conditions.begin(),
                               boundary_conditions.end(),
                               [](const auto& bc)
                               {
                                   return bc.is_neumann();
                               });
        return it != boundary_conditions.end();
    }
}
