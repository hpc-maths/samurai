// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>

#include <fmt/color.h>
#include <fmt/format.h>

#include "algorithm.hpp"
#include "cell_list.hpp"
#include "level_cell_array.hpp"
#include "samurai_config.hpp"
#include "utils.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#endif

namespace samurai
{

    template <class CA, bool is_const>
    class CellArray_iterator;

    template <class iterator>
    class CellArray_reverse_iterator : public std::reverse_iterator<iterator>
    {
      public:

        using base_type  = std::reverse_iterator<iterator>;
        using coord_type = typename iterator::coord_type;

        explicit CellArray_reverse_iterator(iterator&& it)
            : base_type(std::move(it))
        {
        }

        auto index() const
        {
            iterator it = this->base();
            return (--it).index();
        }

        auto level() const
        {
            iterator it = this->base();
            return (--it).level();
        }
    };

    //////////////////////////
    // CellArray definition //
    //////////////////////////

    /** @class CellArray
     *  @brief Array of LevelCellArray.
     *
     *  A box is defined by its minimum and maximum corners.
     *
     *  @tparam dim_ The dimension
     *  @tparam TInterval The type of the intervals (default type is
     * default_config::interval_t).
     *  @tparam max_size_ The size of the array and the maximum levels (default
     * size is default_config::max_level).
     */
    template <std::size_t dim_, class TInterval = default_config::interval_t, std::size_t max_size_ = default_config::max_level>
    class CellArray
    {
      public:

        static constexpr auto dim      = dim_;
        static constexpr auto max_size = max_size_;

        using self_type  = CellArray<dim_, TInterval, max_size_>;
        using interval_t = TInterval;
        using cell_t     = Cell<dim, interval_t>;
        using value_t    = typename interval_t::value_t;
        using index_t    = typename interval_t::index_t;
        using lca_type   = LevelCellArray<dim, TInterval>;
        using cl_type    = CellList<dim, TInterval, max_size>;

        using iterator               = CellArray_iterator<self_type, false>;
        using reverse_iterator       = CellArray_reverse_iterator<iterator>;
        using const_iterator         = CellArray_iterator<const self_type, true>;
        using const_reverse_iterator = CellArray_reverse_iterator<const_iterator>;

        CellArray();
        CellArray(const cl_type& cl, bool with_update_index = true);

        const lca_type& operator[](std::size_t i) const;
        lca_type& operator[](std::size_t i);

        template <typename... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;
        template <class E>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, const xt::xexpression<E>& index) const;
        template <class E>
        const interval_t& get_interval(std::size_t level, const xt::xexpression<E>& coord) const;

        template <typename... T>
        index_t get_index(std::size_t level, value_t i, T... index) const;
        template <class E>
        index_t get_index(std::size_t level, value_t i, const xt::xexpression<E>& others) const;
        template <class E>
        index_t get_index(std::size_t level, const xt::xexpression<E>& coord) const;

        template <typename... T>
        cell_t get_cell(std::size_t level, value_t i, T... index) const;
        template <class E>
        cell_t get_cell(std::size_t level, value_t i, const xt::xexpression<E>& others) const;
        template <class E>
        cell_t get_cell(std::size_t level, const xt::xexpression<E>& coord) const;

        std::size_t nb_cells() const;
        std::size_t nb_cells(std::size_t level) const;

        std::size_t nb_intervals(std::size_t dim) const;
        std::size_t nb_intervals(std::size_t dim, std::size_t level) const;

        std::size_t max_level() const;
        std::size_t min_level() const;

        void update_index();

        void to_stream(std::ostream& os) const;

        iterator begin();
        iterator end();

        reverse_iterator rbegin();
        reverse_iterator rend();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        const_reverse_iterator rend() const;
        const_reverse_iterator rbegin() const;
        const_reverse_iterator rcend() const;
        const_reverse_iterator rcbegin() const;

      private:

        std::array<lca_type, max_size + 1> m_cells;

#ifdef SAMURAI_WITH_MPI
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive& ar, const unsigned long)
        {
            ar& m_cells;
        }
#endif
    };

    ////////////////////////////////////
    // CellArray_iterator definition //
    ///////////////////////////////////

    namespace detail
    {
        template <class CA, bool is_const>
        struct get_lca_iterator_type;

        template <class CA>
        struct get_lca_iterator_type<CA, true>
        {
            using type = LevelCellArray_iterator<const typename CA::lca_type, true>;
        };

        template <class CA>
        struct get_lca_iterator_type<CA, false>
        {
            using type = LevelCellArray_iterator<typename CA::lca_type, false>;
        };
    }

    template <class CA, bool is_const>
    class CellArray_iterator
        : public xtl::xrandom_access_iterator_base3<CellArray_iterator<CA, is_const>, LevelCellArray_iterator<typename CA::lca_type, is_const>>
    {
      public:

        static constexpr std::size_t dim = CA::dim;
        using self_type                  = CellArray_iterator<CA, is_const>;
        using iterator_type              = typename detail::get_lca_iterator_type<CA, is_const>::type;
        using value_type                 = typename iterator_type::value_type;
        using index_type                 = typename iterator_type::index_type;
        using index_type_iterator        = typename iterator_type::index_type_iterator;
        using const_index_type_iterator  = typename iterator_type::const_index_type_iterator;
        using reference                  = typename iterator_type::reference;
        using pointer                    = typename iterator_type::pointer;
        using difference_type            = typename iterator_type::difference_type;
        using iterator_category          = typename iterator_type::iterator_category;

        using coord_type = typename iterator_type::coord_type;

        explicit CellArray_iterator(CA* ca, const iterator_type& lca_it);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;
        const coord_type& index() const;
        std::size_t level() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

      private:

        CA* p_ca;
        iterator_type m_lca_it;
    };

    //////////////////////////////
    // CellArray implementation //
    //////////////////////////////

    /**
     * Default contructor which sets the level for each LevelCellArray.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline CellArray<dim_, TInterval, max_size_>::CellArray()
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level] = {level};
        }
    }

    /**
     * Construction of a CellArray from a CellList
     *
     * @param cl The cell list.
     * @parma with_update_index A boolean indicating if the index of the
     * x-intervals must be computed.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline CellArray<dim_, TInterval, max_size_>::CellArray(const cl_type& cl, bool with_update_index)
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level] = cl[level];
        }

        if (with_update_index)
        {
            update_index();
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::operator[](std::size_t i) const -> const lca_type&
    {
        return m_cells[i];
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::operator[](std::size_t i) -> lca_type&
    {
        return m_cells[i];
    }

    /**
     * Return the x-interval satisfying the input parameters
     *
     * @param level The desired level.
     * @param interval The desired x-interval.
     * @param index The desired indices for the other dimensions.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <typename... T>
    inline auto CellArray<dim_, TInterval, max_size_>::get_interval(std::size_t level, const interval_t& interval, T... index) const
        -> const interval_t&
    {
        return m_cells[level].get_interval(interval, index...);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <class E>
    inline auto
    CellArray<dim_, TInterval, max_size_>::get_interval(std::size_t level, const interval_t& interval, const xt::xexpression<E>& index) const
        -> const interval_t&
    {
        return m_cells[level].get_interval(interval, index);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <class E>
    inline auto CellArray<dim_, TInterval, max_size_>::get_interval(std::size_t level, const xt::xexpression<E>& coord) const
        -> const interval_t&
    {
        return m_cells[level].get_interval(coord);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <typename... T>
    inline auto CellArray<dim_, TInterval, max_size_>::get_index(std::size_t level, value_t i, T... index) const -> index_t
    {
        return m_cells[level].get_index(i, index...);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <class E>
    inline auto CellArray<dim_, TInterval, max_size_>::get_index(std::size_t level, value_t i, const xt::xexpression<E>& others) const
        -> index_t
    {
        return m_cells[level].get_index(i, others);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <class E>
    inline auto CellArray<dim_, TInterval, max_size_>::get_index(std::size_t level, const xt::xexpression<E>& coord) const -> index_t
    {
        return m_cells[level].get_index(coord);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <typename... T>
    inline auto CellArray<dim_, TInterval, max_size_>::get_cell(std::size_t level, value_t i, T... index) const -> cell_t
    {
        return m_cells[level].get_cell(i, index...);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <class E>
    inline auto CellArray<dim_, TInterval, max_size_>::get_cell(std::size_t level, value_t i, const xt::xexpression<E>& others) const
        -> cell_t
    {
        return m_cells[level].get_cell(i, others);
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    template <class E>
    inline auto CellArray<dim_, TInterval, max_size_>::get_cell(std::size_t level, const xt::xexpression<E>& coord) const -> cell_t
    {
        return m_cells[level].get_cell(coord);
    }

    /**
     * Return the number of cells which is the sum of each x-interval size
     * over the levels.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::nb_cells() const
    {
        std::size_t size = 0;
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            size += m_cells[level].nb_cells();
        }
        return size;
    }

    /**
     * Return the number of cells which is the sum of each x-interval size
     * for a given level.
     *
     * @param level The level where to compute the number of cells
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::nb_cells(std::size_t level) const
    {
        return m_cells[level].nb_cells();
    }

    /**
     * Return the number of intervals for a given dimension over the levels.
     *
     * @param d The dimension where to compute the number of intervals
     * @return The number of intervals for the given dimension
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::nb_intervals(std::size_t d) const
    {
        assert(d < dim);
        std::size_t size = 0;
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            size += m_cells[level].nb_intervals(d);
        }
        return size;
    }

    /**
     * Return the number of intervals for a given dimension and a given level.
     *
     * @param d The dimension where to compute the number of intervals
     * @param level The level where to compute the number of intervals
     * @return The number of intervals for the given dimension
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::nb_intervals(std::size_t d, std::size_t level) const
    {
        assert(d < dim);
        return m_cells[level].nb_intervals(d);
    }

    /**
     * Return the maximum level where the array entry is not empty.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::max_level() const
    {
        for (std::size_t level = max_size; level != std::size_t(-1); --level)
        {
            if (!m_cells[level].empty())
            {
                return level;
            }
        }
        return 0;
    }

    /**
     * Return the minimum level where the array entry is not empty.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::min_level() const
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            if (!m_cells[level].empty())
            {
                return level;
            }
        }
        return max_size + 1;
    }

    /**
     * Update the index in the x-intervals allowing to navigate in the
     * Field data structure.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline void CellArray<dim_, TInterval, max_size_>::update_index()
    {
        std::size_t acc_size = 0;
        for_each_interval(*this,
                          [&](auto, auto& interval, auto)
                          {
                              interval.index = safe_subs<index_t>(acc_size, interval.start);
                              acc_size += interval.size();
                          });
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline void CellArray<dim_, TInterval, max_size_>::to_stream(std::ostream& os) const
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            if (!m_cells[level].empty())
            {
                os << fmt::format(disable_color ? fmt::text_style() : fg(fmt::color::steel_blue) | fmt::emphasis::bold,
                                  "┌{0:─^{2}}┐\n"
                                  "│{1: ^{2}}│\n"
                                  "└{0:─^{2}}┘\n",
                                  "",
                                  fmt::format("Level {}", level),
                                  20);
                m_cells[level].to_stream(os);
                os << std::endl;
            }
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::begin() -> iterator
    {
        return iterator(this, m_cells[min_level()].begin());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::end() -> iterator
    {
        return iterator(this, m_cells[max_level()].end());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::end() const -> const_iterator
    {
        return cend();
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::cbegin() const -> const_iterator
    {
        return const_iterator(this, m_cells[min_level()].cbegin());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::cend() const -> const_iterator
    {
        return const_iterator(this, m_cells[max_level()].cend());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::rend() -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::rbegin() const -> const_reverse_iterator
    {
        return rcbegin();
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::rend() const -> const_reverse_iterator
    {
        return rcend();
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::rcbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::rcend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::ostream& operator<<(std::ostream& out, const CellArray<dim_, TInterval, max_size_>& cell_array)
    {
        cell_array.to_stream(out);
        return out;
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline bool operator==(const CellArray<dim_, TInterval, max_size_>& ca1, const CellArray<dim_, TInterval, max_size_>& ca2)
    {
        if (ca1.max_level() != ca2.max_level() || ca1.min_level() != ca2.min_level())
        {
            return false;
        }

        for (std::size_t level = ca1.min_level(); level <= ca1.max_level(); ++level)
        {
            if (!(ca1[level] == ca2[level]))
            {
                return false;
            }
        }
        return true;
    }

    ///////////////////////////////////////
    // CellArray_iterator implementation //
    ///////////////////////////////////////

    template <class CA, bool is_const>
    inline CellArray_iterator<CA, is_const>::CellArray_iterator(CA* ca, const iterator_type& lca_it)
        : p_ca(ca)
        , m_lca_it(lca_it)
    {
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::operator++() -> self_type&
    {
        if (m_lca_it == (*p_ca)[p_ca->max_level()].end())
        {
            return *this;
        }

        ++m_lca_it;
        if (m_lca_it.level() < p_ca->max_level() && m_lca_it == (*p_ca)[m_lca_it.level()].end())
        {
            m_lca_it = (*p_ca)[m_lca_it.level() + 1].begin();
        }
        return *this;
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::operator--() -> self_type&
    {
        if (m_lca_it == (*p_ca)[p_ca->min_level()].begin())
        {
            return *this;
        }

        if (m_lca_it.level() > p_ca->min_level() && m_lca_it == (*p_ca)[m_lca_it.level()].begin())
        {
            m_lca_it = (*p_ca)[m_lca_it.level() - 1].end();
        }
        --m_lca_it;
        return *this;
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::operator+=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            ++(*this);
        }
        return *this;
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::operator-=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            --(*this);
        }
        return *this;
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_lca_it.operator-(rhs.m_lca_it);
        // return m_current_index[0] - rhs.m_current_index[0];
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::operator*() const -> reference
    {
        return m_lca_it.operator*();
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::operator->() const -> pointer
    {
        return m_lca_it.operator->();
    }

    template <class CA, bool is_const>
    inline auto CellArray_iterator<CA, is_const>::index() const -> const coord_type&
    {
        return m_lca_it.index();
    }

    template <class CA, bool is_const>
    inline std::size_t CellArray_iterator<CA, is_const>::level() const
    {
        return m_lca_it.level();
    }

    template <class CA, bool is_const>
    inline bool CellArray_iterator<CA, is_const>::equal(const self_type& rhs) const
    {
        return p_ca == rhs.p_ca && m_lca_it.level() == rhs.m_lca_it.level() && m_lca_it.equal(rhs.m_lca_it);
    }

    template <class CA, bool is_const>
    inline bool CellArray_iterator<CA, is_const>::less_than(const self_type& rhs) const
    {
        return p_ca == rhs.p_ca
            && (m_lca_it.level() < rhs.m_lca_it.level() || (m_lca_it.level() == rhs.m_lca_it.level() && m_lca_it.less_than(rhs.m_lca_it)));
    }

    template <class CA, bool is_const>
    inline bool operator==(const CellArray_iterator<CA, is_const>& it1, const CellArray_iterator<CA, is_const>& it2)
    {
        return it1.equal(it2);
    }

    template <class CA, bool is_const>
    inline bool operator<(const CellArray_iterator<CA, is_const>& it1, const CellArray_iterator<CA, is_const>& it2)
    {
        return it1.less_than(it2);
    }

    template <class CA, bool is_const>
    inline bool operator==(const std::reverse_iterator<CellArray_iterator<CA, is_const>>& it1,
                           const std::reverse_iterator<CellArray_iterator<CA, is_const>>& it2)
    {
        return it1.base().equal(it2.base());
    }

} // namespace samurai
