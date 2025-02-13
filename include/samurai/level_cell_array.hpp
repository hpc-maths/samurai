// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <iterator>
#include <limits>
#include <vector>
#include <type_traits> // fore std::make_signed
 
#ifdef SAMURAI_WITH_MPI
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#endif

#include <fmt/color.h>
#include <fmt/format.h>

#include <ranges>

#include "algorithm.hpp"
#include "box.hpp"
#include "interval.hpp"
#include "level_cell_list.hpp"
#include "mesh_interval.hpp"
#include "samurai_config.hpp"
#include "subset/node.hpp"
#include "utils.hpp"

using namespace xt::placeholders;

namespace samurai
{

    template <class LCA, bool is_const>
    class LevelCellArray_iterator;

    template <class iterator>
    class LevelCellArray_reverse_iterator : public std::reverse_iterator<iterator>
    {
      public:

        using base_type  = std::reverse_iterator<iterator>;
        using coord_type = typename iterator::coord_type;

        explicit LevelCellArray_reverse_iterator(iterator&& it)
            : base_type(std::move(it))
        {
        }

        const coord_type index() const
        {
            iterator it = this->base();
            return (--it).index();
        }

        std::size_t level() const
        {
            iterator it = this->base();
            return (--it).level();
        }
    };

    ///////////////////////////////
    // LevelCellArray definition //
    ///////////////////////////////
    template <std::size_t Dim, class TInterval = default_config::interval_t>
    class LevelCellArray
    {
      public:

        static constexpr auto dim = Dim;
        using interval_t          = TInterval;
        using cell_t              = Cell<dim, interval_t>;
        using index_t             = typename interval_t::index_t;
        using value_t             = typename interval_t::value_t;
        using coord_index_t       = typename interval_t::coord_index_t;
        using mesh_interval_t     = MeshInterval<Dim, TInterval>;
        using indices_t           = typename cell_t::indices_t;
        using coords_t            = typename cell_t::coords_t;

        using iterator               = LevelCellArray_iterator<LevelCellArray<Dim, TInterval>, false>;
        using reverse_iterator       = LevelCellArray_reverse_iterator<iterator>;
        using const_iterator         = LevelCellArray_iterator<const LevelCellArray<Dim, TInterval>, true>;
        using const_reverse_iterator = LevelCellArray_reverse_iterator<const_iterator>;

        using coord_type     = typename iterator::coord_type;
        using all_coord_type = typename iterator::all_coord_type;
        using index_type     = std::array<value_t, dim>;

        static constexpr double default_approx_box_tol = 0.05;

        LevelCellArray();
        LevelCellArray(const LevelCellList<Dim, TInterval>& lcl);

        template <class Op, class StartEndOp, class... S>
        LevelCellArray(Subset<Op, StartEndOp, S...> set);

        LevelCellArray(std::size_t level, const Box<value_t, dim>& box);
        LevelCellArray(std::size_t level,
                       const Box<double, dim>& box,
                       double approx_box_tol = default_approx_box_tol,
                       double scaling_factor = 0);
        LevelCellArray(std::size_t level);
        LevelCellArray(std::size_t level, const coords_t& origin_point, double scaling_factor);
				
				void add_interval (const std::array<value_t, 2>& x_interval, const std::array<value_t, Dim-1>& yz_point);
				void add_interval (const std::array<interval_t, Dim>& interval_nd);
				void add_point    (const indices_t& point);
				void remove_point (const indices_t& point);

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        reverse_iterator rbegin();
        reverse_iterator rend();

        const_reverse_iterator rbegin() const;
        const_reverse_iterator rend() const;
        const_reverse_iterator rcbegin() const;
        const_reverse_iterator rcend() const;

        /// Display to the given stream
        void to_stream(std::ostream& os) const;

        // get_interval
        template <typename... T, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<T, value_t>...>, void>>
        const interval_t& get_interval(const interval_t& interval, T... index) const;
        const interval_t& get_interval(const interval_t& interval, const coord_type& index) const;
        const interval_t& get_interval(const all_coord_type& coord) const;

        // get_index
        template <typename... T, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<T, value_t>...>, void>>
        index_t get_index(value_t i, T... index) const;
        template <class E>
        index_t get_index(value_t i, const xt::xexpression<E>& others) const;
        index_t get_index(const all_coord_type& coord) const;

        // get_cell
        template <typename... T, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<T, value_t>...>, void>>
        cell_t get_cell(value_t i, T... index) const;
        template <class E>
        cell_t get_cell(value_t i, const xt::xexpression<E>& others) const;
        template <class E>
        cell_t get_cell(const xt::xexpression<E>& coord) const;

        void update_index();

        //// checks whether the container is empty
        bool empty() const;

        //// Gives the number of intervals in each dimension
        auto shape() const;

        //// Gives the total number of intervals
        auto nb_intervals() const;

        //// Gives the number of cells
        std::size_t nb_cells() const;
        
        // return the number of cells in the mesh
        size_t nb_cells(const size_t d) const;

				// 

        double cell_length() const;

        const std::vector<interval_t>& operator[](std::size_t d) const;
        std::vector<interval_t>& operator[](std::size_t d);

        const std::vector<std::size_t>& offsets(std::size_t d) const;
        std::vector<std::size_t>& offsets(std::size_t d);

        std::size_t level() const;

        void clear();

        auto min_indices() const;
        auto max_indices() const;
        auto minmax_indices() const;

        auto& origin_point() const;
        void set_origin_point(const coords_t& origin_point);

        auto scaling_factor() const;
        void set_scaling_factor(double scaling_factor);

      private:
#ifdef SAMURAI_WITH_MPI
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive& ar, const unsigned long)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                ar& m_cells[d];
            }
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                ar& m_offsets[d];
            }
            ar & m_level;
        }
#endif
				void add_interval_rec       (const std::array<value_t, 2>& x_interval, const std::array<value_t, Dim-1>& yz_point, const size_t d, const index_t z_interval_idx);
				void add_interval_rec_final (const std::array<value_t, 2>& x_interval, const std::array<value_t, Dim-1>& yz_point,                 const index_t z_interval_idx);

				void add_interval_rec       (const std::array<interval_t, Dim>& interval_nd, const size_t d, const typename interval_t::value_t y, const index_t y_interval_idx);
				void add_interval_rec_final (const std::array<interval_t, Dim>& interval_nd,                 const typename interval_t::value_t y, const index_t y_interval_idx);
				
				void add_point_rec       (const indices_t& point, const size_t d, const index_t interval_idx);
				void add_point_rec_final (const indices_t& point,                 const index_t interval_idx);
				
				void remove_point_rec       (const indices_t& point, const size_t d, const index_t interval_idx);
				void remove_point_rec_final (const indices_t& point,                 const index_t interval_idx);
				
				void update_index(const size_t d); 
				
        /// Recursive construction from a level cell list along dimension > 0
        template <typename TGrid, std::size_t N>
        void init_from_level_cell_list(const TGrid& grid, std::array<value_t, dim - 1> index, std::integral_constant<std::size_t, N>);

        /// Recursive construction from a level cell list for the dimension 0
        template <typename TIntervalList>
        void init_from_level_cell_list(const TIntervalList& interval_list,
                                       const std::array<value_t, dim - 1>& index,
                                       std::integral_constant<std::size_t, 0>);

        void init_from_box(const Box<value_t, dim>& box);

        std::array<std::vector<interval_t>, dim> m_cells;        ///< All intervals in every direction
        std::array<std::vector<std::size_t>, dim - 1> m_offsets; ///< Offsets in interval list for each dim >
                                                                 ///< 1
        std::size_t m_level = 0;
        coords_t m_origin_point;
        double m_scaling_factor = 1;
    };

    ////////////////////////////////////////
    // LevelCellArray_iterator definition //
    ////////////////////////////////////////
    namespace detail
    {
        template <class LCA, bool is_const>
        struct LevelCellArray_iterator_types
        {
            using value_type          = typename LCA::interval_t;
            using index_type          = std::vector<value_type>;
            using index_type_iterator = std::conditional_t<is_const, typename index_type::const_iterator, typename index_type::iterator>;
            using const_index_type_iterator = typename index_type::const_iterator;
            using reference                 = typename index_type_iterator::reference;
            using pointer                   = typename index_type_iterator::pointer;
            using difference_type           = typename index_type_iterator::difference_type;
        };
    } // namespace detail

    template <class LCA, bool is_const>
    class LevelCellArray_iterator
        : public xtl::xrandom_access_iterator_base3<LevelCellArray_iterator<LCA, is_const>, detail::LevelCellArray_iterator_types<LCA, is_const>>
    {
      public:

        static constexpr std::size_t dim = LCA::dim;
        using self_type                  = LevelCellArray_iterator<LCA, is_const>;
        using iterator_type              = detail::LevelCellArray_iterator_types<LCA, is_const>;
        using value_type                 = typename iterator_type::value_type;
        using index_type                 = typename iterator_type::index_type;
        using index_type_iterator        = typename iterator_type::index_type_iterator;
        using const_index_type_iterator  = typename iterator_type::const_index_type_iterator;
        using iterator_container         = std::array<index_type_iterator, dim>;
        using reference                  = typename iterator_type::reference;
        using pointer                    = typename iterator_type::pointer;
        using difference_type            = typename iterator_type::difference_type;
        using iterator_category          = std::random_access_iterator_tag;

        using offset_type          = std::vector<std::size_t>;
        using offset_type_iterator = std::array<typename offset_type::const_iterator, dim - 1>;

        using coord_type     = xt::xtensor_fixed<typename value_type::value_t, xt::xshape<dim - 1>>;
        using all_coord_type = xt::xtensor_fixed<typename value_type::value_t, xt::xshape<dim>>;

        LevelCellArray_iterator(LCA* lca, offset_type_iterator&& offset_index, iterator_container&& current_index, coord_type&& index);

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

        LCA* p_lca;
        offset_type_iterator m_offset_index;
        iterator_container m_current_index;
        mutable coord_type m_index;
    };

    ///////////////////////////////////
    // LevelCellArray implementation //
    ///////////////////////////////////
    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray() // ensures m_offsets size is 1 
			: m_level(0)
			, m_scaling_factor(1)
    {
			for (std::size_t d = 0; d < dim - 1; ++d) 
			{
				m_offsets[d].push_back(0);
			}
		}    
    
    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(const LevelCellList<Dim, TInterval>& lcl)
        : m_level(lcl.level())
        , m_origin_point(lcl.origin_point())
        , m_scaling_factor(lcl.scaling_factor())
    {
        /* Estimating reservation size
         *
         * NOTE: the estimation takes time, more than the time needed for
         * reallocating the vectors... Maybe 2 other solutions:
         * - (highly) overestimating the needed size since the memory will be
         * actually allocated only when touched (at least under Linux)
         * - cnt_x and cnt_yz updated in LevelCellList during the filling
         * process
         *
         * NOTE2: in fact, hard setting the optimal values for cnt_x and cnt_yz
         * doesn't speedup things, strang...
         */
        if (!lcl.empty())
        {
            // Filling cells and offsets from the level cell list
            init_from_level_cell_list(lcl.grid_yz(), {}, std::integral_constant<std::size_t, dim - 1>{});
            // Additionnal offset so that [m_offset[i], m_offset[i+1][ is always
            // valid.
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                m_offsets[d].emplace_back(m_cells[d].size());
            }
        }
    }

    template <std::size_t Dim, class TInterval>
    template <class Op, class StartEndOp, class... S>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(Subset<Op, StartEndOp, S...> set)
    {
        LevelCellList<Dim, TInterval> lcl{static_cast<std::size_t>(set.level())};

        set(
            [&lcl](const auto& i, const auto& index)
            {
                lcl[index].add_interval(i);
            });
        *this = {lcl};
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level, const Box<value_t, dim>& box)
        : m_level{level}
    {
        m_scaling_factor = box.min_length();
        m_origin_point   = box.min_corner();
        init_from_box(box);
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level,
                                                          const Box<double, dim>& box,
                                                          double approx_box_tol,
                                                          double scaling_factor)
        : m_level(level)
    {
        using box_t   = Box<value_t, dim>;
        using point_t = typename box_t::point_t;

        assert(approx_box_tol > 0 || scaling_factor > 0);

        // The computational domain is an approximation of the desired box.
        // If `scaling_factor` is given (i.e. > 0), we take it;
        // otherwise we choose the scaling factor dynamically in order to approximate the desired box
        // up to the tolerance `approx_box_tol`.

        m_origin_point   = box.min_corner();
        auto approx_box  = approximate_box(box, approx_box_tol, scaling_factor);
        m_scaling_factor = scaling_factor;

        point_t start_pt;
        start_pt.fill(0);
        point_t end_pt = approx_box.length() / cell_length();
        init_from_box(box_t{start_pt, end_pt});
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level)
        : m_level{level}
    {
        m_origin_point.fill(0);
			for (std::size_t d = 0; d < dim - 1; ++d) 
			{
				m_offsets[d].push_back(0);
			}
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level, const coords_t& origin_point, double scaling_factor)
        : m_level{level}
        , m_origin_point(origin_point)
        , m_scaling_factor(scaling_factor)
    {
    }
    
    template <std::size_t Dim, class TInterval>
		size_t LevelCellArray<Dim, TInterval>::nb_cells(const size_t d) const 
		{ 
			return std::accumulate(m_cells[d].cbegin(), m_cells[d].cend(), 0u, [](const size_t& value, const interval_t& interval) -> size_t 
			{ 
				return value + interval.size(); 
			}); 
		}
		
		template<std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_interval(const std::array<value_t, 2>& x_interval, const std::array<value_t, Dim-1>& yz_point)
    {
			constexpr size_t d = Dim-1;
			
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			//if (x_interval.is_empty() or not x_interval.is_valid()) { return; }
			if (x_interval[0] >=  x_interval[1]) { return; }
			
			if constexpr (Dim == 1)
			{
				add_interval_rec_final(x_interval, yz_point, 0);
			}
			else
			{
				const auto recursive_call = [this, &x_interval = std::as_const(x_interval), &yz_point = std::as_const(yz_point)](const interval_index_t z_interval_index) -> void 
				{
					if (d > 1) { add_interval_rec(x_interval, yz_point, d-1, z_interval_index);  }
					else       { add_interval_rec_final(x_interval, yz_point, z_interval_index); }
				};
				
				const value_t z = yz_point[d-1];
			
				std::vector<std::size_t>& z_offsets = m_offsets[d-1];
			
				std::vector<interval_t>& intervals = m_cells[d];
			
				assert(z_offsets.size() == nb_cells(d) + 1);
			
				if (intervals.size() == 0)
				{
					intervals.push_back( interval_t(z, z+1) );	
					z_offsets.insert(z_offsets.begin() + 1, z_offsets[0]);
					update_index(d);			
					recursive_call(intervals[0].index);
					return;
				}
				for (const size_t i : std::views::iota(0u, intervals.size()) | std::views::reverse)
				{
					interval_value_t& a = intervals[i].start;
					interval_value_t& b = intervals[i].end;
					interval_index_t& ab_index = intervals[i].index;
					
					if (intervals[i].contains(z))
					{
						recursive_call(ab_index);
						break;
					}
					else if (z == b) // extend the interval [a,b) to [a,b)U{z} = [a,b+1)
					{                // there may be a an interval [a', b') such that b = a' -> extend [a, b) to [a,b')
						z_offsets.insert(z_offsets.begin() + b + ab_index + 1, z_offsets[b + ab_index]);
						++b;
						if (i+1 < intervals.size() and b == intervals[i+1].start)
						{
							b = intervals[i+1].end;
							intervals.erase(intervals.begin() + int(i) + 1); // delete [a',b')	
						}
						update_index(d);
						recursive_call(ab_index);
						break;
					}
					else if (z == a-1) // extend the interval [a,b) to {z}U[a,b) = [a-1,b)
					{                  // there may be an interval [a',b') such that b' = a -> extend [a',b') to [a', b)
						z_offsets.insert(z_offsets.begin() + a + ab_index, z_offsets[a + ab_index]);
						interval_index_t z_interval_index = ab_index;
						--a;
						if (0 < i and intervals[i-1].end == a)
						{
							z_interval_index = intervals[i-1].index;
							intervals[i-1].end = b;
							intervals.erase(intervals.begin() + int(i)); // delete [a,b)
						}
						update_index(d);
						recursive_call( z_interval_index );
						break;
					}
					else if (z > b) // add singleton {y} = [y,y+1) after [a,b)
					{
						z_offsets.insert(z_offsets.begin() + b + ab_index + 1, z_offsets[b + ab_index]);
						const interval_index_t& z_interval_idx = intervals.insert(intervals.begin() + int(i) + 1, interval_t(z, z+1))->index;
						update_index(d);
						recursive_call(z_interval_idx); // probably possible to avoid recursive call here.
						break;
					}
				}
				assert(z_offsets.size() == nb_cells(d) + 1);
			}
		}
		
		template<std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_interval_rec(const std::array<value_t, 2>& x_interval, const std::array<value_t, Dim-1>& yz_point, const size_t d, const index_t z_interval_idx)
    {
			assert(d-2 < Dim-1);
			
			using signed_size_t    = std::make_signed_t<size_t>;
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			const auto recursive_call = [this, d, &x_interval = std::as_const(x_interval), &yz_point = std::as_const(yz_point)](const interval_index_t z_interval_index) -> void 
			{
				if (d > 1) { add_interval_rec(x_interval, yz_point, d-1, z_interval_index);  }
				else       { add_interval_rec_final(x_interval, yz_point, z_interval_index); }
			};
			
			const value_t y = yz_point[d-1];
			const value_t z = yz_point[d];
			
			std::vector<std::size_t>& y_offsets = m_offsets[d-1];
			std::vector<std::size_t>& z_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			assert(y_offsets.size() == nb_cells(d) + 1);
			assert(z + z_interval_idx     < signed_size_t(z_offsets.size()));
			assert(z + z_interval_idx + 1 < signed_size_t(z_offsets.size()));
			
			const size_t i0 = z_offsets[z + z_interval_idx];
			const size_t i1 = z_offsets[z + z_interval_idx + 1];
			
			if (intervals.size() == 0)
			{
				intervals.push_back(interval_t(y, y+1));
				y_offsets.insert(y_offsets.begin() + 1, y_offsets[0]);
				for (size_t j=z + z_interval_idx + 1;j<z_offsets.size();++j) { ++z_offsets[j]; }
				update_index(d);
				recursive_call(intervals[0].index);
				return;
			}
			else if (i0 == i1)
			{
				typename std::vector<interval_t>::const_iterator it = intervals.insert(intervals.begin() + i1, interval_t(y, y+1));
				interval_value_t b = 0;
				interval_value_t y_interval_idx = 0;
				
				if (i0 > 0)
				{
					b = intervals[i0-1].end;
					y_interval_idx = intervals[i0-1].index;
				}
				y_offsets.insert(y_offsets.begin() + b + y_interval_idx + 1, y_offsets[b + y_interval_idx]);
				for (size_t j=z + z_interval_idx + 1;j<z_offsets.size();++j) { ++z_offsets[j]; }
				update_index(d);
				recursive_call(it->index);
				return;
			}
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (intervals[i].contains(y))
				{
					recursive_call(ab_index);
					break;
				}
				else if (y == b) // extend the interval [a,b) to [a,b)U{y} = [a,b+1)
				{                // there may be a an interval [a', b') such that b = a' -> extend [a, b) to [a,b')
					y_offsets.insert(y_offsets.begin() + b + ab_index + 1, y_offsets[b + ab_index]);
					++b;
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + int(i) + 1); // delete [a',b')	
						for (size_t j=z + z_interval_idx + 1;j<z_offsets.size();++j) { --z_offsets[j]; }
					}
					update_index(d);
					recursive_call(ab_index);
					break;
				}
				else if (y == a-1) // extend the interval [a,b) to {y}U[a,b) = [a-1,b)
				{                  // there may be an interval [a',b') such that b' = a -> extend [a',b') to [a', b)
					y_offsets.insert(y_offsets.begin() + a + ab_index, y_offsets[a + ab_index]);
					interval_index_t y_interval_index = ab_index;
					--a;
					if (i0 < i and intervals[i-1].end == a)
					{
						y_interval_index = intervals[i-1].index;
						intervals[i-1].end = b;
						intervals.erase(intervals.begin() + int(i)); // delete [a,b)
						for (size_t j=z + z_interval_idx + 1;j<z_offsets.size();++j) { --z_offsets[j]; }
					}
					update_index(d);
					recursive_call(y_interval_index);
					break;
				}
				else if (y > b) // add singleton {y} = [y,y+1) after [a,b)
				{
					y_offsets.insert(y_offsets.begin() + b + ab_index + 1, y_offsets[b + ab_index]);
					const interval_index_t& y_interval_idx = intervals.insert(intervals.begin() + int(i) + 1, interval_t(y, y+1))->index;
					update_index(d);
					for (size_t j=z + z_interval_idx + 1;j<z_offsets.size();++j) { ++z_offsets[j]; }
					recursive_call(y_interval_idx);
					break;
				}
			}
			assert(y_offsets.size() == nb_cells(d) + 1);
		}
		
		template<std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_interval_rec_final(const std::array<value_t, 2>& _x_interval, const std::array<value_t, Dim-1>& yz_point, const index_t y_interval_idx)
    {
			constexpr size_t d = 0;
			
			using signed_size_t    = std::make_signed_t<size_t>;
			using interval_value_t = typename interval_t::value_t;
			
			//const value_t y = yz_point[d];
			const interval_t x_interval(_x_interval[0], _x_interval[1]);
			
			//std::vector<std::size_t>& y_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			if (Dim > 1) { assert(yz_point[d] + y_interval_idx     < signed_size_t(m_offsets[d].size())); }
			if (Dim > 1) { assert(yz_point[d] + y_interval_idx + 1 < signed_size_t(m_offsets[d].size())); }
			
			const size_t i0 = (Dim > 1) ? m_offsets[d][yz_point[d] + y_interval_idx]     : 0;
			const size_t i1 = (Dim > 1) ? m_offsets[d][yz_point[d] + y_interval_idx + 1] : intervals.size();
			
			assert(i1 <= intervals.size());
			if (intervals.size() == 0)
			{
				intervals.push_back(x_interval);
				if (Dim > 1) { for (size_t j=yz_point[d] + y_interval_idx + 1;j<m_offsets[d].size();++j) { ++m_offsets[d][j]; } }
				return;
			}
			else if (Dim > 1 and i0 == i1)
			{
				intervals.insert(intervals.begin() + signed_size_t(i1), x_interval);
				for (size_t j=yz_point[d] + y_interval_idx + 1;j<m_offsets[d].size();++j) { ++m_offsets[d][j]; }
				return;
			}
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				
				if (b < x_interval.start)
				{
					intervals.insert(intervals.begin() + int(i) + 1, x_interval);
					if (Dim > 1) { for (size_t j=yz_point[d] + y_interval_idx + 1;j<m_offsets[d].size();++j) { ++m_offsets[d][j]; } }
					break;
				}
				else if (a <= x_interval.start)
				{
					b = std::max(b, x_interval.end);
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + signed_size_t(i) + 1);
						if (Dim > 1) { for (size_t j=yz_point[d] + y_interval_idx + 1;j<m_offsets[d].size();++j) { --m_offsets[d][j]; } }
					}
					break;
				}
				else if (a <= x_interval.end)
				{
					const size_t old_size = intervals.size();
					
					b = std::max(b, x_interval.end);
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + signed_size_t(i) + 1);
					}
					a = x_interval.start;
					for (const size_t j : std::views::iota(i0, i) | std::views::reverse)
					{
						if (intervals[j].end < a) { break; }
						if (intervals[j].start <= a) 
						{
							a = intervals[j].start;
							intervals[j].end = intervals[j].start;
							break; 
						}
						else { intervals[j].end = intervals[j].start; }
					}
					const auto it = std::remove_if(intervals.begin() + i0, intervals.begin() + signed_size_t(i), [](const interval_t& interval) -> bool
					{
						return interval.is_empty();
					});
					intervals.erase(it, intervals.begin() + signed_size_t(i));
					const size_t new_size = intervals.size();
					if (Dim > 1) { for (size_t j=yz_point[d] + y_interval_idx + 1;j<m_offsets[d].size();++j) { m_offsets[d][j] -= (old_size - new_size); } }
					break;
				}
			}
		}

    template<std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_interval(const std::array<interval_t, Dim>& interval_nd)
    {
			constexpr size_t d = Dim-1;
			
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			for (size_t i=0;i<Dim;++i)
			{
				if (interval_nd[d].is_empty() or not interval_nd[d].is_valid()) { return; }
			}
			
			const auto recursive_call = [this, d, &interval_nd = std::as_const(interval_nd)](const interval_value_t x_start, const interval_value_t x_end, const index_t x_interval_index) -> void 
			{
				if constexpr (Dim > 1)
				{
					if (d > 1) { for (interval_value_t x=x_start;x!=x_end;++x) { add_interval_rec(interval_nd, d-1, x, x_interval_index);  } }
					else       { for (interval_value_t x=x_start;x!=x_end;++x) { add_interval_rec_final(interval_nd, x, x_interval_index); } }
				}
			};
			
			const interval_t& x_interval  = interval_nd[d];
			
			//std::vector<std::size_t>& x_offsets = m_offsets[d-1];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			if (Dim > 1) { assert(m_offsets[d-1].size() == nb_cells(d) + 1); }
			
			if (intervals.size() == 0)
			{
				intervals.push_back(x_interval);
				if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + 1, x_interval.size(), m_offsets[d-1][0]); }
				update_index(d);
				recursive_call(x_interval.start, x_interval.end, intervals[0].index);
				return;
			}
			for (const size_t i : std::views::iota(0u, intervals.size()) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (b < x_interval.start)
				{
					if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + b + ab_index + 1, x_interval.size(), m_offsets[d-1][b + ab_index]); }
					const interval_index_t& x_interval_idx = intervals.insert(intervals.begin() + int(i) + 1, x_interval)->index;
					update_index(d);
					recursive_call(x_interval.start, x_interval.end, x_interval_idx);
					break;
				}
				else if (a <= x_interval.start)
				{
					if (Dim > 1 and b < x_interval.end) { m_offsets[d-1].insert(m_offsets[d-1].begin() + b + ab_index + 1, x_interval.end - b, m_offsets[d-1][b + ab_index]); }
					b = std::max(b, x_interval.end);
					if (i+1 < intervals.size() and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + int(i) + 1);
					}
					update_index(d);
					recursive_call(x_interval.start, x_interval.end, ab_index);
					break;
				}
				else if (a <= x_interval.end)
				{
					//b = std::max(b, x_interval.end);
					// add offsets corresponding to [b, x_interval.end[
					if (Dim > 1 and b < x_interval.end) { m_offsets[d-1].insert(m_offsets[d-1].begin() + b + ab_index + 1, x_interval.end - b, m_offsets[d-1][b + ab_index]); }
					b = std::max(b, x_interval.end);
					if (i+1 < intervals.size() and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + int(i) + 1);
					}
					// if last interval, add offsets corresponding to [x_interval.start, a[
					if (Dim > 1 and i == 0u) { m_offsets[d-1].insert(m_offsets[d-1].begin() + a + ab_index,  a - x_interval.start, m_offsets[d-1][a + ab_index]); }
					interval_value_t new_a = x_interval.start;
					for (const size_t j : std::views::iota(0u, i) | std::views::reverse)
					{
						if (intervals[j].end < new_a)    
						{ // add offsets corresponding to [new_a, intervals[j+1].start[
							if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + intervals[j+1].start + intervals[j+1].index,  intervals[j+1].start - new_a, m_offsets[d-1][intervals[j+1].start + intervals[j+1].index]); }
							break; 
						}
						if (intervals[j].start <= new_a) 
						{ // add offsets corresponding to [intervals[j].end, intervals[j+1].start[
							if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + intervals[j].end + intervals[j].index + 1, intervals[j+1].start - intervals[j].end, m_offsets[d-1][intervals[j].end + intervals[j].index]); }
							new_a = intervals[j].start; 
							intervals[j].end = intervals[j].start;
							break; 
						}
						else
						{ // add offsets corresponding to [intervals[j].end, intervals[j+1].start[
							if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + intervals[j].end + intervals[j].index + 1, intervals[j+1].start - intervals[j].end, m_offsets[d-1][intervals[j].end + intervals[j].index]); }
							intervals[j].end = intervals[j].start;
							// if last interval, add offsets corresponding to [x_interval.start, intervals[j].start[
							if (Dim > 1 and j == 0u) { m_offsets[d-1].insert(m_offsets[d-1].begin() + intervals[j].start + intervals[j].index, intervals[j].start - x_interval.start, m_offsets[d-1][intervals[j].start + intervals[j].index]); }
						}
					}
					a = new_a;
					const auto it = std::remove_if(intervals.begin(), intervals.begin() + int(i), [](const interval_t& interval) -> bool
					{
						return interval.is_empty();
					});
					// We need to recompute the indexes before removing the empty intervals.
					// 1. Since the intervals to be removed are empty, the indexes are correctly calculated
					// 2. If we where to remove the intervals before storing the index the pointer would be invalidated.
					// Thus we first compute and store the index and then remove the empty intervals. 
					update_index(d);
					const interval_value_t interval_start = a;
					const interval_value_t interval_end = b;
					const interval_index_t interval_index = ab_index; // after erase, ab_index is invalidated.
					intervals.erase(it, intervals.begin() + int(i));
					
					recursive_call(interval_start, interval_end, interval_index);
					break;
				}
			}	
			if (Dim > 1) { assert(m_offsets[d-1].size() == nb_cells(d) + 1); }
		}
    
    template<std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_interval_rec(const std::array<interval_t, Dim>& interval_nd, const size_t d, const typename interval_t::value_t y, const index_t y_interval_idx)
    {
			assert(d > 0);
			
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			const auto recursive_call = [this, d, &interval_nd = std::as_const(interval_nd)](const interval_value_t x_start, const interval_value_t x_end, const index_t x_interval_index) -> void 
			{
				if (d > 1) { for (interval_value_t x=x_start;x!=x_end;++x) { add_interval_rec(interval_nd, d-1, x, x_interval_index);  } }
				else       { for (interval_value_t x=x_start;x!=x_end;++x) { add_interval_rec_final(interval_nd, x, x_interval_index); } }
			};
			
			const interval_t& x_interval = interval_nd[d];
			
			std::vector<std::size_t>& x_offsets = m_offsets[d-1];
			std::vector<std::size_t>& y_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			assert(x_offsets.size() == nb_cells(d) + 1);
			assert(y + y_interval_idx < y_offsets.size());
			assert(y + y_interval_idx + 1 < y_offsets.size());
			
			const size_t i0 = y_offsets[y + y_interval_idx];
			const size_t i1 = y_offsets[y + y_interval_idx + 1];
			
			assert(i1 <= intervals.size());
			
			if (intervals.size() == 0)
			{
				intervals.push_back(x_interval);
				x_offsets.insert(x_offsets.begin() + 1, x_interval.size(), x_offsets[0]);
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				update_index(d);
				recursive_call(x_interval.start, x_interval.end, intervals[0].index);
				return;
			}
			else if (i0 == i1)
			{
				typename std::vector<interval_t>::const_iterator it = intervals.insert(intervals.begin() + i1, x_interval);
				interval_value_t b = 0;
				interval_value_t x_interval_idx = 0;
				
				if (i0 > 0)
				{
					b = intervals[i0-1].end;
					x_interval_idx = intervals[i0-1].index;
				}
				x_offsets.insert(x_offsets.begin() + b + x_interval_idx + 1, x_interval.size(), x_offsets[b + x_interval_idx]);
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				update_index(d);
				recursive_call(x_interval.start, x_interval.end, it->index);
				return;
			}
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (b < x_interval.start)
				{
					x_offsets.insert(x_offsets.begin() + b + ab_index + 1, x_interval.size(), x_offsets[b + ab_index]);
					const interval_index_t& x_interval_idx = intervals.insert(intervals.begin() + int(i) + 1, x_interval)->index;
					for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
					update_index(d);
					recursive_call(x_interval.start, x_interval.end, x_interval_idx);
					break;
				}
				else if (a <= x_interval.start)
				{
					if (b < x_interval.end) { x_offsets.insert(x_offsets.begin() + b + ab_index + 1, x_interval.end - b, x_offsets[b + ab_index]); }
					b = std::max(b, x_interval.end);
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + i + 1);
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					update_index(d);
					recursive_call(x_interval.start, x_interval.end, ab_index);
					break;
				}
				else if (a <= x_interval.end)
				{
					const size_t old_size = intervals.size();
					//b = std::max(b, x_interval.end);
					// add offsets corresponding to [b, x_interval.end[
					if (b < x_interval.end) { x_offsets.insert(x_offsets.begin() + b + ab_index + 1, x_interval.end - b, x_offsets[b + ab_index]); }
					b = std::max(b, x_interval.end);
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + i + 1);
					}
					// if last interval, add offsets corresponding to [x_interval.start, a[
					if (i==i0) { x_offsets.insert(x_offsets.begin() + a + ab_index,  a - x_interval.start, x_offsets[a + ab_index]); }
					interval_value_t new_a = x_interval.start;
					for (const size_t j : std::views::iota(i0, i) | std::views::reverse)
					{
						if (intervals[j].end < new_a)    
						{ // add offsets corresponding to [new_a, intervals[j+1].start[
							x_offsets.insert(x_offsets.begin() + intervals[j+1].start + intervals[j+1].index,  intervals[j+1].start - new_a, x_offsets[intervals[j+1].start + intervals[j+1].index]);
							break; 
						}
						if (intervals[j].start <= new_a) 
						{ // add offsets corresponding to [intervals[j].end, intervals[j+1].start[
							x_offsets.insert(x_offsets.begin() + intervals[j].end + intervals[j].index + 1, intervals[j+1].start - intervals[j].end, x_offsets[intervals[j].end + intervals[j].index]);
							new_a = intervals[j].start; 
							intervals[j].end = intervals[j].start;
							break; 
						}
						else
						{ // add offsets corresponding to [intervals[j].end, intervals[j+1].start[
							x_offsets.insert(x_offsets.begin() + intervals[j].end + intervals[j].index + 1, intervals[j+1].start - intervals[j].end, x_offsets[intervals[j].end + intervals[j].index]);
							intervals[j].end = intervals[j].start;
							// if last interval, add offsets corresponding to [x_interval.start, intervals[j].start[
							if (j == 0u) { x_offsets.insert(x_offsets.begin() + intervals[j].start + intervals[j].index, intervals[j].start - x_interval.start, x_offsets[intervals[j].start + intervals[j].index]); }
						}
					}
					a = new_a;
					const auto it = std::remove_if(intervals.begin() + i0, intervals.begin() + i, [](const interval_t& interval) -> bool
					{
						return interval.is_empty();
					});
					// We need to recompute the indexes before removing the empty intervals.
					// 1. Since the intervals to be removed are empty, the indexes are correctly calculated
					// 2. If we where to remove the intervals before storing the index the pointer would be invalidated.
					// Thus we first compute and store the index and then remove the empty intervals. 
					update_index(d);
					const interval_value_t interval_start = a;
					const interval_value_t interval_end = b;
					const interval_index_t interval_index = ab_index; // after erase, ab_index is invalidated.
					intervals.erase(it, intervals.begin() + i);
					
					const size_t new_size = intervals.size();
					for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { y_offsets[j] -= (old_size - new_size); }
					
					recursive_call(interval_start, interval_end, interval_index);
					break;
				}
			}
			assert(x_offsets.size() == nb_cells(d) + 1);
		}
    
    template<std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_interval_rec_final(const std::array<interval_t, Dim>& interval_nd, const typename interval_t::value_t y, const index_t y_interval_idx)
    {
			constexpr size_t d = 0;
			
			//using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			const interval_t& x_interval = interval_nd[d];
			
			std::vector<std::size_t>& y_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			assert(y + y_interval_idx < y_offsets.size());
			assert(y + y_interval_idx + 1 < y_offsets.size());
			
			const size_t i0 = y_offsets[y + y_interval_idx];
			const size_t i1 = y_offsets[y + y_interval_idx + 1];
			
			assert(i1 <= intervals.size());
			
			if (intervals.size() == 0)
			{
				intervals.push_back(x_interval);
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				update_index(d);
				return;
			}
			else if (i0 == i1)
			{
				intervals.insert(intervals.begin() + i1, x_interval);
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				update_index(d);
				return;
			}
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				
				if (b < x_interval.start)
				{
					intervals.insert(intervals.begin() + int(i) + 1, x_interval);
					update_index(d);
					for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
					break;
				}
				else if (a <= x_interval.start)
				{
					b = std::max(b, x_interval.end);
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + i + 1);
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					update_index(d);
					break;
				}
				else if (a <= x_interval.end)
				{
					const size_t old_size = intervals.size();
					
					b = std::max(b, x_interval.end);
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + i + 1);
					}
					a = x_interval.start;
					for (const size_t j : std::views::iota(i0, i) | std::views::reverse)
					{
						if (intervals[j].end < a) { break; }
						if (intervals[j].start <= a) 
						{
							a = intervals[j].start;
							intervals[j].end = intervals[j].start;
							break; 
						}
						else 
						{ 
							intervals[j].end = intervals[j].start; 
						}
					}
					const auto it = std::remove_if(intervals.begin() + i0, intervals.begin() + i, [](const interval_t& interval) -> bool
					{
						return interval.is_empty();
					});
					intervals.erase(it, intervals.begin() + i);
					update_index(d);
					
					const size_t new_size = intervals.size();
					for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { y_offsets[j] -= (old_size - new_size); }
					break;
				}
			}
		}
    
    template <std::size_t Dim, class TInterval>
		inline void LevelCellArray<Dim, TInterval>::add_point(const indices_t& point)
		{
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			constexpr size_t d = Dim-1;
			
			const auto recursive_call = [this, &point = std::as_const(point)](const interval_index_t x_interval_index) -> void 
			{
				if constexpr (Dim > 1)
				{
					if (d > 1) { add_point_rec(point, d-1, x_interval_index);  }
					else       { add_point_rec_final(point, x_interval_index); }
				}
			};
			
			const value_t x = point[d];
			
			//std::vector<std::size_t>& x_offsets = m_offsets[d-1];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			if (Dim > 1) { assert(m_offsets[d-1].size() == nb_cells(d) + 1); }
			
			if (intervals.size() == 0)
			{
				intervals.push_back( interval_t(x, x+1) );	
				if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + 1, m_offsets[d-1][0]); }
				update_index(d);			
				recursive_call(intervals[0].index);
				return;
			}
			for (const size_t i : std::views::iota(0u, intervals.size()) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (intervals[i].contains(x))
				{
					recursive_call(ab_index);
					break;
				}
				else if (x == b) // extend the interval [a,b) to [a,b)U{x} = [a,b+1)
				{                // there may be a an interval [a', b') such that b = a' -> extend [a, b) to [a,b')
					if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + b + ab_index + 1, m_offsets[d-1][b + ab_index]); }
					++b;
					if (i+1 < intervals.size() and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + int(i) + 1); // delete [a',b')	
					}
					if (Dim > 1) { update_index(d); }
					recursive_call(ab_index);
					break;
				}
				else if (x == a-1) // extend the interval [a,b) to {x}U[a,b) = [a-1,b)
				{                  // there may be an interval [a',b') such that b' = a -> extend [a',b') to [a', b)
					if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + a + ab_index, m_offsets[d-1][a + ab_index]); }
					interval_index_t z_interval_index = ab_index;
					--a;
					if (0 < i and intervals[i-1].end == a)
					{
						z_interval_index = intervals[i-1].index;
						intervals[i-1].end = b;
						intervals.erase(intervals.begin() + int(i)); // delete [a,b)
					}
					if (Dim > 1) { update_index(d); }
					recursive_call( z_interval_index );
					break;
				}
				else if (x > b) // add singleton {x} = [x,x+1) after [a,b)
				{
					if (Dim > 1) { m_offsets[d-1].insert(m_offsets[d-1].begin() + b + ab_index + 1, m_offsets[d-1][b + ab_index]); }
					const interval_index_t& x_interval_idx = intervals.insert(intervals.begin() + int(i) + 1, interval_t(x, x+1))->index;
					if (Dim > 1) { update_index(d); }
					recursive_call(x_interval_idx); // probably possible to avoid recursive call here.
					break;
				}
			}
			
			if (Dim > 1) { assert(m_offsets[d-1].size() == nb_cells(d) + 1); }
		}

		template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_point_rec(const indices_t& point, const size_t d, const index_t y_interval_idx)
    {
			assert(d > 0);
			
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			const auto recursive_call = [this, d, &point = std::as_const(point)](const interval_index_t x_interval_index) -> void 
			{
				if (d > 1) { add_point_rec(point, d-1, x_interval_index);  }
				else       { add_point_rec_final(point, x_interval_index); }
			};
			
			const value_t x = point[d];
			const value_t y = point[d+1];

			std::vector<std::size_t>& x_offsets = m_offsets[d-1];
			std::vector<std::size_t>& y_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			assert(x_offsets.size() == nb_cells(d) + 1);
			assert(y + y_interval_idx < y_offsets.size());
			assert(y + y_interval_idx + 1 < y_offsets.size());
			
			const size_t i0 = y_offsets[y + y_interval_idx];
			const size_t i1 = y_offsets[y + y_interval_idx + 1];
			
			if (intervals.size() == 0)
			{
				intervals.push_back(interval_t(x, x+1));
				x_offsets.insert(x_offsets.begin() + 1, x_offsets[0]);
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				update_index(d);
				recursive_call(intervals[0].index);
				return;
			}
			else if (i0 == i1)
			{
				//const interval_index_t& index = intervals.insert(intervals.begin() + i1, interval_t(x, x+1))->index;
				typename std::vector<interval_t>::const_iterator it = intervals.insert(intervals.begin() + i1, interval_t(x, x+1));
				interval_value_t b = 0;
				interval_value_t x_interval_idx = 0;
				
				if (i0 > 0)
				{
					b = intervals[i0-1].end;
					x_interval_idx = intervals[i0-1].index;
				}
				x_offsets.insert(x_offsets.begin() + b + x_interval_idx + 1, x_offsets[b + x_interval_idx]);
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				update_index(d);
				recursive_call(it->index);
				return;
			}
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (intervals[i].contains(x))
				{
					recursive_call(ab_index);
					break;
				}
				else if (x == b) // extend the interval [a,b) to [a,b)U{x} = [a,b+1)
				{                // there may be a an interval [a', b') such that b = a' -> extend [a, b) to [a,b')
					x_offsets.insert(x_offsets.begin() + b + ab_index + 1, x_offsets[b + ab_index]);
					++b;
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + int(i) + 1); // delete [a',b')	
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					update_index(d);
					recursive_call(ab_index);
					break;
				}
				else if (x == a-1) // extend the interval [a,b) to {x}U[a,b) = [a-1,b)
				{                  // there may be an interval [a',b') such that b' = a -> extend [a',b') to [a', b)
					x_offsets.insert(x_offsets.begin() + a + ab_index, x_offsets[a + ab_index]);
					interval_index_t z_interval_index = ab_index;
					--a;
					if (i0 < i and intervals[i-1].end == a)
					{
						z_interval_index = intervals[i-1].index;
						intervals[i-1].end = b;
						intervals.erase(intervals.begin() + int(i)); // delete [a,b)
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					update_index(d);
					recursive_call(z_interval_index);
					break;
				}
				else if (x > b) // add singleton {x} = [x,x+1) after [a,b)
				{
					x_offsets.insert(x_offsets.begin() + b + ab_index + 1, x_offsets[b + ab_index]);
					const interval_index_t& x_interval_idx = intervals.insert(intervals.begin() + int(i) + 1, interval_t(x, x+1))->index;
					update_index(d);
					for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
					recursive_call(x_interval_idx);
					break;
				}
			}
			assert(x_offsets.size() == nb_cells(d) + 1);
		}
		
		template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::add_point_rec_final(const indices_t& point, const index_t y_interval_idx)
    {			
			constexpr size_t d = 0;
			
			using interval_value_t = typename interval_t::value_t;
			
			const value_t x = point[d];
			const value_t y = point[d+1];
			
			std::vector<std::size_t>& y_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			assert(y + y_interval_idx < y_offsets.size());
			assert(y + y_interval_idx + 1 < y_offsets.size());
			
			const size_t i0 = y_offsets[y + y_interval_idx];
			const size_t i1 = y_offsets[y + y_interval_idx + 1];
			
			if (intervals.size() == 0)
			{
				intervals.push_back(interval_t(x, x+1));
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				return;
			}
			else if (i0 == i1)
			{
				intervals.insert(intervals.begin() + i1, interval_t(x, x+1));
				for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
				return;
			}
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				
				if (intervals[i].contains(x))
				{
					break;
				}
				else if (x == b) // extend the interval [a,b) to [a,b)U{x} = [a,b+1)
				{                // there may be a an interval [a', b') such that b = a' -> extend [a, b) to [a,b')
					++b;
					if (i+1 < i1 and b == intervals[i+1].start)
					{
						b = intervals[i+1].end;
						intervals.erase(intervals.begin() + int(i) + 1); // delete [a',b')	
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					break;
				}
				else if (x == a-1) // extend the interval [a,b) to {x}U[a,b) = [a-1,b)
				{                  // there may be an interval [a',b') such that b' = a -> extend [a',b') to [a', b)
					--a;
					if (i0 < i and intervals[i-1].end == a)
					{
						intervals[i-1].end = b;
						intervals.erase(intervals.begin() + int(i)); // delete [a,b)
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					break;
				}
				else if (x > b) // add singleton {x} = [x,x+1) after [a,b)
				{
					intervals.insert(intervals.begin() + int(i) + 1, interval_t(x, x+1));
					for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
					break;
				}
			}
		}

		template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::remove_point(const indices_t& point)
    {			
			constexpr size_t d = Dim-1;
			
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
		
			const auto recursive_call = [this, &point = std::as_const(point)](const interval_index_t x_interval_index) -> void 
			{
				if constexpr (Dim > 1)
				{
					if (d > 1) { remove_point_rec(point, d-1, x_interval_index);  }
					else       { remove_point_rec_final(point, x_interval_index); }
				}
			};
			
			const value_t x = point[d];
			
			//std::vector<std::size_t>& x_offsets = m_offsets[d-1];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			if (Dim > 1) { assert(m_offsets[d-1].size() == nb_cells(d) + 1); }
			
			
			for (const size_t i : std::views::iota(0u, intervals.size()) | std::views::reverse)
			{		
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (x == a)
				{
					recursive_call(ab_index);
					const bool remove_x = (Dim == 1) or (m_offsets[d-1][x + ab_index] == m_offsets[d-1][x + ab_index + 1]); // thanks lazy eval 
					if (remove_x) // there is no interval for x=a, shrink [a,b) to [a+1,b)
					{
						if (Dim > 1) { m_offsets[d-1].erase(m_offsets[d-1].begin() + x + ab_index); } // known at compile time, do not create branches
						++a;
						if (intervals[i].is_empty())  // remove interval
						{ 
							intervals.erase(intervals.begin() + int(i));
						}
					}
					if (Dim > 1) { update_index(d); }
					break;
				}
				else if (x == b-1)
				{
					recursive_call(ab_index);
					const bool remove_x = (Dim == 1) or (m_offsets[d-1][x + ab_index] == m_offsets[d-1][x + ab_index + 1]); 
					if (remove_x) // there is no interval for x=a, shrink [a,b) to [a,b-1)
					{
						if (Dim > 1) { m_offsets[d-1].erase(m_offsets[d-1].begin() + x + ab_index); } // known at compile time, do not create branches
						--b;
						if (intervals[i].is_empty())  // remove interval
						{ 
							intervals.erase(intervals.begin() + int(i));
						}
					}
					if (Dim > 1) { update_index(d); }
					break;
				}
				else if (intervals[i].contains(x))
				{
					recursive_call(ab_index);
					const bool remove_x = (Dim == 1) or (m_offsets[d-1][x + ab_index] == m_offsets[d-1][x + ab_index + 1]); 
					if (remove_x) // there is no interval for x, split [a,b) to [a, x) and [x+1,b)
					{
						if (Dim > 1) { m_offsets[d-1].erase(m_offsets[d-1].begin() + x + ab_index); } // known at compile time, do not create branches
						
						interval_t new_interval = interval_t(x+1, b);
						b = x;
						intervals.insert(intervals.begin() + int(i) + 1, new_interval);
					}
					if (Dim > 1) { update_index(d); }
					break;
				}
			}
			
			if (Dim > 1) { assert(m_offsets[d-1].size() == nb_cells(d) + 1); }
		}

		template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::remove_point_rec(const indices_t& point, const size_t d, const index_t y_interval_idx)
    {			
			assert(d > 0);
			
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
		
			const auto recursive_call = [this, d, &point = std::as_const(point)]( const interval_index_t x_interval_index) -> void 
			{
				if (d > 1) { remove_point_rec(point, d-1, x_interval_index);  }
				else       { remove_point_rec_final(point, x_interval_index);      }
			};
			
			const value_t x = point[d];
			const value_t y = point[d+1];
			
			std::vector<std::size_t>& x_offsets = m_offsets[d-1];
			std::vector<std::size_t>& y_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			if (Dim > 1) { assert(x_offsets.size() == nb_cells(d) + 1); }
			
			const size_t i0 = y_offsets[y + y_interval_idx];
			const size_t i1 = y_offsets[y + y_interval_idx + 1];
			
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{		
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (x == a)
				{
					recursive_call(ab_index);
					if (x_offsets[x + ab_index] == x_offsets[x + ab_index + 1]) // there is no interval for x=a, shrink [a,b) to [a+1,b)
					{
						x_offsets.erase(x_offsets.begin() + x + ab_index);
						++a;
						if (intervals[i].is_empty())  // remove interval
						{ 
							intervals.erase(intervals.begin() + int(i));
							for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
						}
					}
					update_index(d);
					break;
				}
				else if (x == b-1)
				{
					recursive_call(ab_index);
					if (x_offsets[x + ab_index] == x_offsets[x + ab_index + 1]) // there is no interval for x=a, shrink [a,b) to [a,b-1)
					{
						x_offsets.erase(x_offsets.begin() + x + ab_index);
						--b;
						if (intervals[i].is_empty())  // remove interval
						{ 
							intervals.erase(intervals.begin() + int(i));
							for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
						}
					}
					update_index(d);
					break;
				}
				else if (intervals[i].contains(x))
				{
					recursive_call(ab_index);
					if (x_offsets[x + ab_index] == x_offsets[x + ab_index + 1]) // there is no interval for x, split [a,b) to [a, x) and [x+1,b)
					{
						x_offsets.erase(x_offsets.begin() + x + ab_index);
						
						interval_t new_interval = interval_t(x+1, b);
						b = x;
						intervals.insert(intervals.begin() + int(i) + 1, new_interval);
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
					}
					update_index(d);
					break;
				}
			}
			
			if (Dim > 1) { assert(x_offsets.size() == nb_cells(d) + 1); }
		}
		
		template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::remove_point_rec_final(const indices_t& point, const index_t y_interval_idx)
    {			
			constexpr size_t d = 0;
			
			using interval_index_t = typename interval_t::index_t;
			using interval_value_t = typename interval_t::value_t;
			
			const value_t x = point[d];
			const value_t y = point[d+1];
			
			std::vector<std::size_t>& y_offsets = m_offsets[d];
			
			std::vector<interval_t>& intervals = m_cells[d];
			
			const size_t i0 = y_offsets[y + y_interval_idx];
			const size_t i1 = y_offsets[y + y_interval_idx + 1];
			
			for (const size_t i : std::views::iota(i0, i1) | std::views::reverse)
			{		
				interval_value_t& a = intervals[i].start;
				interval_value_t& b = intervals[i].end;
				interval_index_t& ab_index = intervals[i].index;
				
				if (x == a)
				{
					++a;
					if (intervals[i].is_empty())  // remove interval
					{ 
						intervals.erase(intervals.begin() + int(i));
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					break;
				}
				else if (x == b-1)
				{
					--b;
					if (intervals[i].is_empty())  // remove interval
					{ 
						intervals.erase(intervals.begin() + int(i));
						for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { --y_offsets[j]; }
					}
					break;
				}
				else if (intervals[i].contains(x))
				{					
					interval_t new_interval = interval_t(x+1, b);
					b = x;
					intervals.insert(intervals.begin() + int(i) + 1, new_interval);
					for (size_t j=y + y_interval_idx + 1;j<y_offsets.size();++j) { ++y_offsets[j]; }
					break;
				}
			}
		}

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::begin() -> iterator
    {
        typename iterator::offset_type_iterator offset_index;
        typename iterator::iterator_container current_index;
        typename iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].begin();
        }

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cbegin();
            index[d]        = current_index[d + 1]->start;
        }
        return iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::end() -> iterator
    {
        typename iterator::offset_type_iterator offset_index;
        typename iterator::iterator_container current_index;
        typename iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].end() - 1;
        }
        ++current_index[0];

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cend() - 2;
            index[d]        = current_index[d + 1]->end - 1;
        }

        return iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::cbegin() const -> const_iterator
    {
        typename const_iterator::offset_type_iterator offset_index;
        typename const_iterator::iterator_container current_index;
        typename const_iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].cbegin();
        }

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cbegin();
            index[d]        = current_index[d + 1]->start;
        }
        return const_iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::cend() const -> const_iterator
    {
        typename const_iterator::offset_type_iterator offset_index;
        typename const_iterator::iterator_container current_index;
        typename const_iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].cend() - 1;
        }
        ++current_index[0];

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cend() - 2;
            index[d]        = current_index[d + 1]->end - 1;
        }

        return const_iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::end() const -> const_iterator
    {
        return cend();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rend() -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rbegin() const -> const_reverse_iterator
    {
        return rcbegin();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rend() const -> const_reverse_iterator
    {
        return rcend();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rcbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rcend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    /**
     * Return the x-interval satisfying the input parameters
     *
     * @param interval The desired x-interval.
     * @param index The desired indices for the other dimensions.
     */
    template <std::size_t Dim, class TInterval>
    template <typename... T, typename D>
    inline auto LevelCellArray<Dim, TInterval>::get_interval(const interval_t& interval, T... index) const -> const interval_t&
    {
        auto offset = find(*this, {interval.start, index...});
        return m_cells[0][static_cast<std::size_t>(offset)];
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::get_interval(const interval_t& interval, const coord_type& index) const -> const interval_t&
    {
        all_coord_type point;
        point[0] = interval.start;
        for (std::size_t d = 1; d < dim; ++d)
        {
            point[d] = index[d - 1];
        }
        auto offset = find(*this, point);
        return m_cells[0][static_cast<std::size_t>(offset)];
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::get_interval(const all_coord_type& coord) const -> const interval_t&
    {
        auto offset = find(*this, coord);
        return m_cells[0][static_cast<std::size_t>(offset)];
    }

    template <std::size_t Dim, class TInterval>
    template <typename... T, typename D>
    inline auto LevelCellArray<Dim, TInterval>::get_index(value_t i, T... index) const -> index_t
    {
        return get_interval({i, i + 1}, index...).index + i;
    }

    template <std::size_t Dim, class TInterval>
    template <class E>
    inline auto LevelCellArray<Dim, TInterval>::get_index(value_t i, const xt::xexpression<E>& others) const -> index_t
    {
        return get_interval({i, i + 1}, others).index + i;
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::get_index(const all_coord_type& coord) const -> index_t
    {
        return get_interval(coord).index + coord(0);
    }

    template <std::size_t Dim, class TInterval>
    template <typename... T, typename D>
    inline auto LevelCellArray<Dim, TInterval>::get_cell(value_t i, T... index) const -> cell_t
    {
        return {m_origin_point, m_scaling_factor, m_level, i, xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>{index...}, get_index(i, index...)};
    }

    template <std::size_t Dim, class TInterval>
    template <class E>
    inline auto LevelCellArray<Dim, TInterval>::get_cell(value_t i, const xt::xexpression<E>& others) const -> cell_t
    {
        return {m_origin_point, m_scaling_factor, m_level, i, others, get_index(i, others)};
    }

    template <std::size_t Dim, class TInterval>
    template <class E>
    inline auto LevelCellArray<Dim, TInterval>::get_cell(const xt::xexpression<E>& coord) const -> cell_t
    {
        xt::xtensor_fixed<value_t, xt::xshape<dim>> coord_array = coord;

        auto i      = coord_array[0];
        auto others = xt::view(coord_array, xt::range(1, _));
        return {m_origin_point, m_scaling_factor, m_level, i, others, get_index(i, others)};
    }

    /**
     * Update the index in the x-intervals allowing to navigate in the
     * Field data structure.
     */
    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::update_index()
    {
        std::size_t acc_size = 0;
        for_each_interval(*this,
                          [&](auto, auto& interval, auto)
                          {
                              interval.index = safe_subs<index_t>(acc_size, interval.start);
                              acc_size += interval.size();
                          });
    }
    
    template <std::size_t Dim, class TInterval>
		void LevelCellArray<Dim, TInterval>::update_index(const size_t d)
		{
			if (d < m_cells.size())
			{
				size_t acc_size = 0;
				for (interval_t& interval : m_cells[d])
				{
					interval.index = safe_subs<index_t>(acc_size, interval.start);
					acc_size += interval.size();
				}
			}
		}

    template <std::size_t Dim, class TInterval>
    inline bool LevelCellArray<Dim, TInterval>::empty() const
    {
        return m_cells[0].empty();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::shape() const
    {
        std::array<std::size_t, dim> output;
        for (std::size_t d = 0; d < dim; ++d)
        {
            output[d] = m_cells[d].size();
        }
        return output;
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::nb_intervals() const
    {
        std::size_t s = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            s += m_cells[d].size();
        }
        return s;
    }

    template <std::size_t Dim, class TInterval>
    inline std::size_t LevelCellArray<Dim, TInterval>::nb_cells() const
    {
        auto op = [](std::size_t i, const auto& interval)
        {
            return i + interval.size();
        };

        return std::accumulate(m_cells[0].cbegin(), m_cells[0].cend(), std::size_t(0), op);
    }

    template <std::size_t Dim, class TInterval>
    inline std::size_t LevelCellArray<Dim, TInterval>::level() const
    {
        return m_level;
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::clear()
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            m_cells[d].clear();
        }
    }

    template <std::size_t Dim, class TInterval>
    inline double LevelCellArray<Dim, TInterval>::cell_length() const
    {
        return samurai::cell_length(m_scaling_factor, m_level);
    }

    /**
     * Return the maximum value that can take the end of an interval for each
     * direction.
     */
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::max_indices() const
    {
        std::array<value_t, dim> max;
        for (std::size_t d = 0; d < dim; ++d)
        {
            max[d] = std::max_element(m_cells[d].begin(),
                                      m_cells[d].end(),
                                      [](const auto& a, const auto& b)
                                      {
                                          return (a.end < b.end);
                                      })
                         ->end;
        }
        return max;
    }

    /**
     * Return the minimum value that can take the start of an interval for each
     * direction.
     */
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::min_indices() const
    {
        std::array<value_t, dim> min;
        for (std::size_t d = 0; d < dim; ++d)
        {
            min[d] = std::min_element(m_cells[d].begin(),
                                      m_cells[d].end(),
                                      [](const auto& a, const auto& b)
                                      {
                                          return (a.start < b.start);
                                      })
                         ->start;
        }
        return min;
    }

    /**
     * Return the minimum value that can take the start and
     * the maximum value that can take the end of an interval
     * for each direction.
     */
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::minmax_indices() const
    {
        std::array<std::pair<value_t, value_t>, dim> minmax;
        auto min = min_indices();
        auto max = max_indices();
        for (std::size_t d = 0; d < dim; ++d)
        {
            minmax[d].first  = min[d];
            minmax[d].second = max[d];
        }
        return minmax;
    }

    template <std::size_t Dim, class TInterval>
    inline auto& LevelCellArray<Dim, TInterval>::origin_point() const
    {
        return m_origin_point;
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::set_origin_point(const coords_t& origin_point)
    {
        m_origin_point = origin_point;
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::scaling_factor() const
    {
        return m_scaling_factor;
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::set_scaling_factor(double scaling_factor)
    {
        m_scaling_factor = scaling_factor;
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::operator[](std::size_t d) const -> const std::vector<interval_t>&
    {
        return m_cells[d];
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::operator[](std::size_t d) -> std::vector<interval_t>&
    {
        return m_cells[d];
    }

    template <std::size_t Dim, class TInterval>
    inline const std::vector<std::size_t>& LevelCellArray<Dim, TInterval>::offsets(std::size_t d) const
    {
        assert(d > 0);
        return m_offsets[d - 1];
    }

    template <std::size_t Dim, class TInterval>
    inline std::vector<std::size_t>& LevelCellArray<Dim, TInterval>::offsets(std::size_t d)
    {
        assert(d > 0);
        return m_offsets[d - 1];
    }

    template <std::size_t Dim, class TInterval>
    template <typename TGrid, std::size_t N>
    inline void LevelCellArray<Dim, TInterval>::init_from_level_cell_list(const TGrid& grid,
                                                                          std::array<value_t, dim - 1> index,
                                                                          std::integral_constant<std::size_t, N>)
    {
        // Working interval
        interval_t curr_interval(0, 0, 0);

        // For each position along the Nth dimension
        for (const auto& point : grid)
        {
            // Coordinate along the Nth dimension
            const auto i = point.first;

            // Recursive call on the current position for the (N-1)th dimension
            index[N - 1]                      = i;
            const std::size_t previous_offset = m_cells[N - 1].size();
            init_from_level_cell_list(point.second, index, std::integral_constant<std::size_t, N - 1>{});

            /* Since we move on a sparse storage, each coordinate have non-empty
             * co-dimensions So the question is, are we continuing an existing
             * interval or have we jump to another one.
             *
             * WARNING: we are supposing that the sparse array of dimension
             * dim-1 has no empty entry. Otherwise, we should check that the
             * recursive call has do something by comparing previous_offset
             * with the size of m_cells[N-1].
             */
            if (curr_interval.is_valid())
            {
                // If the coordinate has jump out of the current interval
                if (i > curr_interval.end)
                {
                    // Adding the previous interval...
                    m_cells[N].emplace_back(curr_interval);

                    // ... and creating a new one.
                    curr_interval = interval_t(i, i + 1, static_cast<index_t>(m_offsets[N - 1].size()) - i);
                }
                else
                {
                    // Otherwise, we are just continuing the current interval
                    ++curr_interval.end;
                }
            }
            else
            {
                // If there is no current interval (at the beginning of the
                
                // loop) we create a new one.
                curr_interval = interval_t(i, i + 1, static_cast<index_t>(m_offsets[N - 1].size()) - i);
            }

            // Updating m_offsets (at each iteration since we are always
            // updating an interval)
            m_offsets[N - 1].emplace_back(previous_offset);
        }

        // Adding the working interval if valid
        if (curr_interval.is_valid())
        {
            m_cells[N].emplace_back(curr_interval);
        }
    }

    template <std::size_t Dim, class TInterval>
    template <typename TIntervalList>
    inline void LevelCellArray<Dim, TInterval>::init_from_level_cell_list(const TIntervalList& interval_list,
                                                                          const std::array<value_t, dim - 1>& /* index */,
                                                                          std::integral_constant<std::size_t, 0>)
    {
        // Along the X axis, simply copy the intervals in cells[0]
        std::copy(interval_list.begin(), interval_list.end(), std::back_inserter(m_cells[0]));
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::init_from_box(const Box<value_t, dim>& box)
    {
        auto dimensions = xt::cast<std::size_t>(box.length());
        auto start_pt   = box.min_corner();
        auto end_pt     = box.max_corner();

        std::size_t size = 1;
        for (std::size_t d = dim - 1; d > 0; --d)
        {
            m_offsets[d - 1].resize((dimensions[d] * size) + 1);
            for (std::size_t i = 0; i < (dimensions[d] * size) + 1; ++i)
            {
                m_offsets[d - 1][i] = i;
            }
            m_cells[d].resize(size);
            for (std::size_t i = 0; i < size; ++i)
            {
                m_cells[d][i] = {start_pt[d], end_pt[d], static_cast<index_t>(m_offsets[d - 1][i * dimensions[d]]) - start_pt[d]};
            }
            size *= dimensions[d];
        }

        m_cells[0].resize(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            m_cells[0][i] = {start_pt[0], end_pt[0], static_cast<index_t>(i * dimensions[0]) - start_pt[0]};
        }
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::to_stream(std::ostream& os) const
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            os << fmt::format(disable_color ? fmt::text_style() : fmt::emphasis::bold, "{:>10}", fmt::format("dim {}", d)) << std::endl;

            os << fmt::format("{:>20}", "cells = ");
            for (std::size_t ic = 0; ic < m_cells[d].size(); ++ic)
            {
                os << fmt::format(disable_color ? fmt::text_style() : fmt::emphasis::bold, "{}->", ic);
                os << m_cells[d][ic] << " ";
            }
            os << "\n" << std::endl;

            if (d > 0)
            {
                os << fmt::format("{:>20}", "offsets = ");
                for (std::size_t io = 0; io < m_offsets[d - 1].size(); ++io)
                {
                    os << fmt::format("({}: {}) ", io, m_offsets[d - 1][io]);
                }
                os << std::endl << std::endl;
            }
        }
    }

    template <std::size_t Dim, class TInterval>
    inline bool operator==(const LevelCellArray<Dim, TInterval>& lca_1, const LevelCellArray<Dim, TInterval>& lca_2)
    {
        if (lca_1.level() != lca_2.level())
        {
            return false;
        }

        if (lca_1.shape() != lca_2.shape())
        {
            return false;
        }

        for (std::size_t i = 0; i < Dim; ++i)
        {
            if (lca_1[i] != lca_2[i])
            {
                return false;
            }
        }

        for (std::size_t i = 1; i < Dim; ++i)
        {
            if (lca_1.offsets(i) != lca_2.offsets(i))
            {
                return false;
            }
        }
        return true;
    }

    template <std::size_t Dim, class TInterval>
    inline std::ostream& operator<<(std::ostream& out, const LevelCellArray<Dim, TInterval>& level_cell_array)
    {
        level_cell_array.to_stream(out);
        return out;
    }

    ////////////////////////////////////////////
    // LevelCellArray_iterator implementation //
    ////////////////////////////////////////////

    template <class LCA, bool is_const>
    inline LevelCellArray_iterator<LCA, is_const>::LevelCellArray_iterator(LCA* lca,
                                                                           offset_type_iterator&& offset_index,
                                                                           iterator_container&& current_index,
                                                                           coord_type&& index)
        : p_lca(lca)
        , m_offset_index(std::move(offset_index))
        , m_current_index(std::move(current_index))
        , m_index(std::move(index))
    {
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator++() -> self_type&
    {
        if (m_current_index[0] == (*p_lca)[0].end())
        {
            return *this;
        }
        ++m_current_index[0];

        for (std::size_t d = 0; d < m_current_index.size() - 1; ++d)
        {
            auto dst = static_cast<std::size_t>(
                std::distance((*p_lca)[d].cbegin(), static_cast<const_index_type_iterator>(m_current_index[d])));
            if (dst == *(m_offset_index[d] + 1))
            {
                ++m_offset_index[d];
                ++m_index[d];
                if (m_index[d] == m_current_index[d + 1]->end)
                {
                    ++m_current_index[d + 1];
                    if (m_current_index[d + 1] != (*p_lca)[d + 1].end())
                    {
                        m_index[d] = m_current_index[d + 1]->start;
                    }
                }
            }
            else
            {
                break;
            }
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator--() -> self_type&
    {
        if (m_current_index[0] == (*p_lca)[0].begin())
        {
            --m_current_index[0];
            return *this;
        }
        --m_current_index[0];

        for (std::size_t d = 0; d < m_current_index.size() - 1; ++d)
        {
            auto dst = static_cast<std::size_t>(
                std::distance((*p_lca)[d].cbegin(), static_cast<const_index_type_iterator>(m_current_index[d])));
            if (dst == *m_offset_index[d] - 1)
            {
                --m_offset_index[d];
                if (m_index[d] == m_current_index[d + 1]->start)
                {
                    if (m_current_index[d + 1] != (*p_lca)[d + 1].begin())
                    {
                        --m_current_index[d + 1];
                        m_index[d] = m_current_index[d + 1]->end - 1;
                    }
                }
                else
                {
                    --m_index[d];
                }
            }
            else
            {
                break;
            }
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator+=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            ++(*this);
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator-=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            --(*this);
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_current_index[0] - rhs.m_current_index[0];
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator*() const -> reference
    {
        return *(m_current_index[0]);
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator->() const -> pointer
    {
        return std::addressof(this->operator*());
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::index() const -> const coord_type&
    {
        return m_index;
    }

    template <class LCA, bool is_const>
    inline std::size_t LevelCellArray_iterator<LCA, is_const>::level() const
    {
        return p_lca->level();
    }

    template <class LCA, bool is_const>
    inline bool LevelCellArray_iterator<LCA, is_const>::equal(const self_type& rhs) const
    {
        return p_lca == rhs.p_lca && m_current_index[0] == rhs.m_current_index[0];
    }

    template <class LCA, bool is_const>
    inline bool LevelCellArray_iterator<LCA, is_const>::less_than(const self_type& rhs) const
    {
        return p_lca == rhs.p_lca && m_current_index[0] < rhs.m_current_index[0];
    }

    template <class LCA, bool is_const>
    inline bool operator==(const LevelCellArray_iterator<LCA, is_const>& it1, const LevelCellArray_iterator<LCA, is_const>& it2)
    {
        return it1.equal(it2);
    }

    template <class LCA, bool is_const>
    inline bool operator<(const LevelCellArray_iterator<LCA, is_const>& it1, const LevelCellArray_iterator<LCA, is_const>& it2)
    {
        return it1.less_than(it2);
    }

    template <class LCA, bool is_const>
    inline bool operator==(const std::reverse_iterator<LevelCellArray_iterator<LCA, is_const>>& it1,
                           const std::reverse_iterator<LevelCellArray_iterator<LCA, is_const>>& it2)
    {
        return it1.base().equal(it2.base());
    }
} // namespace samurai
