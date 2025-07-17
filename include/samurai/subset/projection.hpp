// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>

#include <utility>

#include "traversers/projection_traverser.hpp"

namespace samurai
{

    template <class Set>
    class Projection;

    template <class Set>
    struct SetTraits<Projection<Set>>
    {
        template <std::size_t d>
        using traverser_t = ProjectionTraverser<typename SetTraits<Set>::traverser_t<d>>;

        static constexpr std::size_t dim = SetTraits<Set>::dim;
    };

    /*
     * The main issue with projection, Coarsening specifically, is that
     * the interval is extended upon projection e.g. both
     * Proj([1,4),1,0) and Proj([0,3),1,0) results in [0, 2)
     * It means the coarsening operation, unlike the translation e.g.,
     * is a unary, NON BIJECTIVE operation.
     * Thus in more than 1d, if I'm projecting over more than 1 level:
     * Proj([5, 17), [18, 20), 2, 0) = [1, 3)
     * So given a y_proj in [1, 3), we need to traverse accross multiple
     * y.
     * y_proj = 1 => y \in [4, 8)
     * y_proj = 2 => y \in [8, 12)
     *
     */
    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
        using Base = SetBase<Projection<Set>>;

      public:

        template <std::size_t d>
        using traverser_t = typename Base::traverser_t<d>;

        using value_t = typename Base::value_t;

        Projection(const Set& set, const std::size_t level)
            : m_set(set)
            , m_level(level)
        {
            if (m_level < m_set.level())
            {
                m_projectionType = ProjectionType::COARSEN;
                m_shift          = m_set.level() - m_level;
            }
            else
            {
                m_projectionType = ProjectionType::REFINE;
                m_shift          = m_level - m_set.level();
            }
        }

        std::size_t level() const
        {
            return m_level;
        }

        bool exist() const
        {
            return m_set.exist();
        }

        bool empty() const
        {
            return m_set.empty();
        }

        template <class index_t, std::size_t d>
        traverser_t<d> get_traverser(const index_t& _index, std::integral_constant<std::size_t, d> d_ic) const
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != Base::dim - 1)
                {
                    const value_t ymin = _index[d] << m_shift;
                    const value_t ymax = (_index[d] + 1) << m_shift;

                    xt::xtensor_fixed<value_t, xt::xshape<Base::dim - 1>> index(_index << m_shift);

                    std::vector<typename SetTraits<Set>::traverser_t<d>> set_traversers;
                    set_traversers.reserve(size_t(ymax - ymin));

                    for (index[d] = ymin; index[d] != ymax; ++index[d])
                    {
                        set_traversers.push_back(m_set.get_traverser(index, d_ic));
                    }
                    return traverser_t<d>(set_traversers, m_shift);
                }
                else
                {
                    return traverser_t<d>(m_set.get_traverser(_index << m_shift, d_ic), m_projectionType, m_shift);
                }
            }
            else
            {
                return traverser_t<d>(m_set.get_traverser(_index >> m_shift, d_ic), m_projectionType, m_shift);
            }
        }

      private:

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;
    };

}
