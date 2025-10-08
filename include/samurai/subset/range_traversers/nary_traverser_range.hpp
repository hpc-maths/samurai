// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "nary_traverser_type.hpp"

namespace samurai
{
    template <NAryTraverserType Op, class... SetTraverserRanges>
    class NaryOperatorTraverserRange;

    template <NAryTraverserType Op, class... SetTraverserRanges>
    struct SetTraverserRangeTraits<NaryOperatorTraverserRange<Op, SetTraverserRanges...>>
    {
        static_assert((IsSetTraverserRange<SetTraverserRanges>::value and ...));

        class Iterator
        {
          public:

            using ChildrenIterators = std::tuple<typename SetTraverserRanges::Iterator...>;

            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = typename NAryTraverserType<Op, typename InnerIterators::value_type...>::Type;
            using reference         = value_type;

            static constexpr std::size_t nIntervals = std::tuple_size<ChildrenIterators>::value;

            Iterator(const ChildrenIterators& innerIterators)
                : m_childrenIterators(innerIterators)
            {
            }

            reference operator*() const
            {
                return std::apply(
                    [](const auto&... childrenIterators) -> void
                    {
                        return reference((*childrenIterators)...);
                    },
                    m_childrenIterators);
            }

            Iterator& operator++()
            {
                static_for<0, nIntervals>::apply(
                    [this](const auto i) -> void
                    {
                        ++std::get<i>(m_childrenIterators);
                    });
                return *this;
            }

            Iterator operator++(int)
            {
                Iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const Iterator& a, const Iterator& b)
            {
                return a.m_childrenIterators == b.m_childrenIterators;
            };

            friend bool operator!=(const Iterator& a, const Iterator& b)
            {
                return a.m_childrenIterators != b.m_childrenIterators;
            };

          private:

            ChildrenIterators m_childrenIterators;
        };
    };

    template <NAryTraverserType Op, class... SetTraverserRanges>
    class NaryOperatorTraverserRange : public SetTraverserRangeBase<NaryOperatorTraverserRange<Op, SetTraverserRanges...>>
    {
        using Self = TranslationTraverserRange<SetTraverserRange>;

      public:

        SAMURAI_SET_TRAVERSER_RANGE_TYPEDEFS

        using Childrens = std::tuple<SetTraverserRanges...>;

        TranslationTraverserRange(const SetTraverserRanges&... set_traverser_ranges)
            : m_set_traverser_ranges(set_traverser_ranges...)
        {
        }

        Iterator begin_impl()
        {
            return std::apply(
                [](auto&... innerIterators) -> Iterator
                {
                    return Iterator(std::make_tuple(innerIterators.begin()...));
                },
                m_set_traverser_ranges);
        }

        Iterator end_impl()
        {
            return std::apply(
                [](auto&... innerIterators) -> Iterator
                {
                    return Iterator(std::make_tuple(innerIterators.end()...));
                },
                m_set_traverser_ranges);
        }

      private:

        Childrens m_set_traverser_ranges;
    };

} // namespace samurai
