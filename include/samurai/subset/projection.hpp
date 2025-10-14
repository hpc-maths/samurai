// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../samurai_config.hpp"
#include "../static_algorithm.hpp"
#include "set_base.hpp"
#include "traversers/projection_traverser.hpp"

namespace samurai
{

    template <class Set>
    class Projection;

    template <class Set>
    struct SetTraits<Projection<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using traverser_t = ProjectionTraverser<typename Set::template traverser_t<d>>;

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    namespace detail
    {
        template <class Set, typename Seq>
        struct ProjectionWork;

        template <class Set, std::size_t... ds>
        struct ProjectionWork<Set, std::index_sequence<ds...>>
        {
            template <std::size_t d>
            using child_traverser_t = typename Set::template traverser_t<d>;

            template <std::size_t d>
            using array_of_child_traverser_t = std::vector<child_traverser_t<d>>;

            using Type = std::tuple<array_of_child_traverser_t<ds>...>;
        };
    } // namespace detail

    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
        using Self                = Projection<Set>;
        using ChildTraverserArray = typename detail::ProjectionWork<Set, std::make_index_sequence<Set::dim>>::Type;

      public:

        SAMURAI_SET_TYPEDEFS

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

        // we need to define a custom copy and move constructor because
        // we do not want to copy m_work_offsetRanges
        Projection(const Projection& other)
            : m_set(other.m_set)
            , m_level(other.m_level)
            , m_projectionType(other.m_projectionType)
            , m_shift(other.m_shift)
        {
        }

        Projection(Projection&& other)
            : m_set(std::move(other.m_set))
            , m_level(std::move(other.m_level))
            , m_projectionType(std::move(other.m_projectionType))
            , m_shift(std::move(other.m_shift))
        {
        }

        inline std::size_t level_impl() const
        {
            return m_level;
        }

        inline bool exist_impl() const
        {
            return m_set.exist();
        }

        inline bool empty_impl() const
        {
            return m_set.empty();
        }

        template <std::size_t d>
        inline void init_get_traverser_work_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d> d_ic) const
        {
            const std::size_t my_work_size_per_traverser = (m_projectionType == ProjectionType::COARSEN and d != Base::dim - 1)
                                                             ? (1 << m_shift)
                                                             : 1;
            const std::size_t my_work_size               = n_traversers * my_work_size_per_traverser;

            auto& childTraversers = std::get<d>(m_work_childTraversers);
            childTraversers.clear();
            childTraversers.reserve(my_work_size);

            m_set.init_get_traverser_work(my_work_size, d_ic);
        }

        template <std::size_t d>
        inline void clear_get_traverser_work_impl(std::integral_constant<std::size_t, d> d_ic) const
        {
            auto& childTraversers = std::get<d>(m_work_childTraversers);

            childTraversers.clear();

            m_set.clear_get_traverser_work(d_ic);
        }

        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser_impl(const index_t& _index, std::integral_constant<std::size_t, d> d_ic) const
        {
            auto& childTraversers = std::get<d>(m_work_childTraversers);

            if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != Base::dim - 1)
                {
                    const std::size_t old_capacity = childTraversers.capacity();

                    const value_t ymin   = _index[d] << m_shift;
                    const value_t ybound = (_index[d] + 1) << m_shift;

                    xt::xtensor_fixed<value_t, xt::xshape<Base::dim - 1>> index(_index << m_shift);

                    const auto childTraversers_begin = childTraversers.end();

                    for (index[d] = ymin; index[d] != ybound; ++index[d])
                    {
                        childTraversers.push_back(m_set.get_traverser(index, d_ic));
                        if (childTraversers.back().is_empty())
                        {
                            childTraversers.pop_back();
                        }
                    }

                    assert(childTraversers.capacity() == old_capacity);

                    return traverser_t<d>(childTraversers_begin, childTraversers.end(), m_shift);
                }
                else
                {
                    childTraversers.push_back(m_set.get_traverser(_index << m_shift, d_ic));
                    return traverser_t<d>(std::prev(childTraversers.end()), m_projectionType, m_shift);
                }
            }
            else
            {
                childTraversers.push_back(m_set.get_traverser(_index >> m_shift, d_ic));
                return traverser_t<d>(std::prev(childTraversers.end()), m_projectionType, m_shift);
            }
        }

      private:

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;

        mutable ChildTraverserArray m_work_childTraversers;
    };

} // namespace samurai
