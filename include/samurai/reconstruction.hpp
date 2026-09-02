#pragma once

#include <array>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "field.hpp"
#include "numeric/prediction.hpp"
#include "prediction_shifts.hpp"
#include "samurai_config.hpp"
#include "subset/node.hpp"
#include "utils.hpp"

/**
 * @file reconstruction.hpp
 *
 * Multiresolution prediction and reconstruction: recovering the value a field
 * would take on a grid finer than the one it is stored on.
 *
 * A cell at level @c l splits into @c 2^dim children at level @c l+1. samurai
 * keeps only the cells actually present in the adapted mesh; the value of an
 * absent finer cell is @e predicted from its coarser neighbours through the
 * wavelet interpolation of half-width @c prediction_stencil_radius (the
 * @ref prediction_coefficients of numeric/prediction_coefficients.hpp). Prediction is linear,
 * so the predicted value is a fixed linear combination of coarse-cell values
 * that depends only on the level gap and on the child position inside its coarse
 * cell, never on the field. Three layers build on that fact:
 *
 *   - @ref prediction_map        the combination itself: a sparse map
 *                                {coarse-cell offset -> weight}.
 *   - @ref prediction "prediction(delta_l, ii)"
 *                                builds and memoises the map predicting child
 *                                @c ii, @c delta_l levels below its coarse cell.
 *   - @ref portion "portion(f, ...)"
 *                                applies such a map to a field @c f, vectorised
 *                                over a whole x-interval of coarse cells.
 *
 * Two higher-level operations consume them: @ref reconstruction projects an
 * adapted field onto a uniform fine mesh (I/O, post-processing), and
 * @ref transfer moves a field from one adapted mesh to another.
 */

namespace samurai
{
    /// Hash a std::array by folding element hashes (boost::hash_combine mix), so
    /// an offset key can index the unordered_map of a @ref prediction_map.
    template <typename T, std::size_t N>
    struct ArrayHash
    {
        std::size_t operator()(const std::array<T, N>& arr) const
        {
            return std::accumulate(arr.begin(),
                                   arr.end(),
                                   std::size_t{0},
                                   [](std::size_t acc, const T& element)
                                   {
                                       return acc ^ (std::hash<T>{}(element) + 0x9e3779b9 + (acc << 6) + (acc >> 2));
                                   });
        }
    };

    /**
     * Sparse linear combination of cell values. @c coeff maps a @c dim-D integer
     * offset @c k (relative to a reference cell) to its weight, so the map stands
     * for  @c sum_k coeff[k] * value(reference + k).
     *
     * The class is the algebra of such combinations: @c += / @c -= merge or cancel
     * the weights of matching offsets, @c *= and the scalar @c += scale or shift
     * every weight. Because prediction is linear, summing the maps of several
     * children yields the map of their sum in one object (used by the slice form
     * of @ref get_prediction). The single-offset constructor is the identity
     * combination @c value(reference + k) (weight 1).
     */
    template <std::size_t dim, class index_t = default_config::value_t>
    class prediction_map
    {
      public:

        using key_t = std::array<index_t, dim>;

        prediction_map() = default;

        prediction_map(const key_t& k)
        {
            coeff[k] = 1.;
        }

        /// Weight of offset @c k, inserted at 0 if absent (so it can be accumulated into).
        double& operator()(const key_t& k)
        {
            auto it = coeff.find(k);
            if (it == coeff.end())
            {
                coeff[k] = 0.;
            }
            return coeff[k];
        }

        double& get(std::array<index_t, 1> index)
        {
            return (*this)(index[0]);
        }

        double& get(std::array<index_t, 2> index)
        {
            return (*this)(index[0], index[1]);
        }

        double& get(std::array<index_t, 3> index)
        {
            return (*this)(index[0], index[1], index[2]);
        }

        prediction_map& operator+=(const prediction_map& p)
        {
            for (const auto& c : p.coeff)
            {
                (*this)(c.first) += c.second;
            }
            return *this;
        }

        prediction_map& operator-=(const prediction_map& p)
        {
            for (const auto& c : p.coeff)
            {
                (*this)(c.first) -= c.second;
            }
            return *this;
        }

        prediction_map& operator*=(const double d)
        {
            for (auto& c : coeff)
            {
                c.second *= d;
            }
            return *this;
        }

        prediction_map& operator+=(const double d)
        {
            for (auto& c : coeff)
            {
                c.second += d;
            }
            return *this;
        }

        void remove_small_entries(double tol = 1e-15)
        {
            for (auto it = coeff.begin(); it != coeff.end();)
            {
                if (std::abs(it->second) < tol)
                {
                    it = coeff.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

        /// Print each `offset: weight` entry, one per line (debugging).
        void to_stream(std::ostream& out) const
        {
            for (const auto& c : coeff)
            {
                out << fmt::format("({}):  {}", c.first, c.second) << std::endl;
            }
        }

        std::unordered_map<std::array<index_t, dim>, double, ArrayHash<index_t, dim>> coeff;
    };

    template <std::size_t dim, class index_t>
    auto operator+(const prediction_map<dim, index_t>& p1, const prediction_map<dim, index_t>& p2)
    {
        prediction_map<dim, index_t> that{p1};
        that += p2;
        return that;
    }

    template <std::size_t dim, class index_t>
    auto operator+(const double d, const prediction_map<dim, index_t>& p)
    {
        prediction_map<dim, index_t> that{p};
        that += d;
        return that;
    }

    template <std::size_t dim, class index_t>
    auto operator-(const prediction_map<dim, index_t>& p1, const prediction_map<dim, index_t>& p2)
    {
        prediction_map<dim, index_t> that{p1};
        that -= p2;
        return that;
    }

    template <std::size_t dim, class index_t>
    auto operator*(const double d, const prediction_map<dim, index_t>& p)
    {
        prediction_map<dim, index_t> that{p};
        that *= d;
        return that;
    }

    template <std::size_t dim, class index_t>
    SAMURAI_INLINE std::ostream& operator<<(std::ostream& out, const prediction_map<dim, index_t>& pred)
    {
        pred.to_stream(out);
        return out;
    }

    namespace detail
    {
        /// Shift @c parent_indices by the stencil offset of a tensor-product node: per
        /// direction, @c loop_indices runs over @c [0, 2*order+1) and @c loop_index - order
        /// is the signed offset around the parent (0 at the stencil centre).
        template <std::size_t... Is>
        auto compute_new_indices(auto order, const auto& parent_indices, const auto& loop_indices, std::index_sequence<Is...>)
        {
            return std::make_tuple(
                (std::get<Is>(parent_indices) + static_cast<default_config::value_t>(std::get<Is>(loop_indices) - order))...);
        }

        auto compute_new_indices(auto order, const auto& parent_indices, const auto& loop_indices)
        {
            return compute_new_indices(order,
                                       parent_indices,
                                       loop_indices,
                                       std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(parent_indices)>>>{});
        }

        /// Shift @c parent_indices by the stencil offset of a tensor-product node whose stencil
        /// starts @c starts[d] cells from the parent along @c d (@c -order when centred, more
        /// when shifted inward): @c loop_indices runs over @c [0, 2*order+1).
        template <std::size_t... Is>
        auto compute_shifted_indices(const auto& starts, const auto& parent_indices, const auto& loop_indices, std::index_sequence<Is...>)
        {
            return std::make_tuple((std::get<Is>(parent_indices) + static_cast<default_config::value_t>(starts[Is])
                                    + static_cast<default_config::value_t>(std::get<Is>(loop_indices)))...);
        }

        auto compute_shifted_indices(const auto& starts, const auto& parent_indices, const auto& loop_indices)
        {
            return compute_shifted_indices(starts,
                                           parent_indices,
                                           loop_indices,
                                           std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(parent_indices)>>>{});
        }

        /**
         * The shift of the one-level prediction stencil around @a parent, a cell @a gap levels
         * below the cell a @ref PredictionPositionClass describes, read off that class: the
         * most centred candidate whose whole box the domain holds - the rule of
         * prediction_shifts.hpp, asked of the class instead of the mesh.
         */
        template <std::size_t radius, std::size_t dim, int reach, class value_t>
        PredictionStencilShift<dim>
        class_shift(const PredictionPositionClass<reach, dim>& cls, std::size_t gap, const std::array<value_t, dim>& parent)
        {
            constexpr int r             = static_cast<int>(radius);
            constexpr std::size_t nodes = ipow(2 * radius + 1, dim);
            const auto& candidates      = shift_search<radius, dim>();

            PredictionStencilShift<dim> best;
            best.fits = false;
            for (std::size_t c = 0; c < candidates.size() && !best.fits; ++c)
            {
                const auto& shift = candidates[c];
                bool admissible   = true;
                for (std::size_t n = 0; n < nodes && admissible; ++n)
                {
                    const auto k = nth_offset<dim>(n, r);
                    std::array<value_t, dim> q;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        q[d] = parent[d] + static_cast<value_t>(shift[d] + k[d]);
                    }
                    admissible = cls.contains(gap, q);
                }
                if (admissible)
                {
                    best.shift = shift;
                    best.fits  = true;
                }
            }
            return best;
        }

        /// Tensor-product weight of a stencil node: the product over the directions of the
        /// per-direction interpolation coefficients @c interp_coeffs[d][indices[d]].
        template <std::size_t... Is>
        auto compute_coeff(const auto& interp_coeffs, const auto& indices, std::index_sequence<Is...>)
        {
            return (std::get<Is>(interp_coeffs)[std::get<Is>(indices)] * ...);
        }

        auto compute_coeff(const auto& interp_coeffs, const auto& indices)
        {
            return compute_coeff(interp_coeffs, indices, std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(interp_coeffs)>>>{});
        }

        /**
         * Cartesian-product loop: call @c func once per node of the box
         * @c [start, end) (one range per direction), passing the @c dim indices of
         * the node. @c compute_idx_func maps each raw counter to the index handed to
         * @c func. The convenience overloads below take a tuple of coefficient arrays
         * (range @c [0, size)) or of intervals (range @c [start, end)).
         *
         * @note The counter is @c std::size_t, so the ranges must be non-negative.
         *       A signed range (e.g. sub-cell indices shifted below 0) needs its own
         *       loop, see @ref accumulate_slice.
         */
        template <class FuncIdx, class Func, std::size_t d>
        void multi_dim_loop(auto& current_index,
                            const auto& start,
                            const auto& end,
                            Func&& func2apply,
                            FuncIdx&& compute_idx_func,
                            std::integral_constant<std::size_t, d>)
        {
            if constexpr (d == 0)
            {
                std::apply(
                    [&](auto... indices)
                    {
                        func2apply(indices...);
                    },
                    current_index);
            }
            else
            {
                for (std::size_t i = std::get<d - 1>(start); i < std::get<d - 1>(end); ++i)
                {
                    std::get<d - 1>(current_index) = compute_idx_func(i);
                    multi_dim_loop(current_index,
                                   start,
                                   end,
                                   std::forward<Func>(func2apply),
                                   std::forward<FuncIdx>(compute_idx_func),
                                   std::integral_constant<std::size_t, d - 1>{});
                }
            }
        }

        template <class FuncIdx, class Func>
        void multi_dim_loop(
            const auto& start,
            const auto& end,
            Func&& func2apply,
            FuncIdx&& compute_idx_func =
                [](auto i)
            {
                return i;
            })
            requires(std::tuple_size_v<std::decay_t<decltype(start)>> == std::tuple_size_v<std::decay_t<decltype(end)>>)
        {
            constexpr std::size_t num_dims = std::tuple_size_v<std::decay_t<decltype(start)>>;

            auto current_index = []<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return std::make_tuple(((void)Is, std::size_t{0})...);
            }(std::make_index_sequence<num_dims>{});
            multi_dim_loop(current_index,
                           start,
                           end,
                           std::forward<Func>(func2apply),
                           std::forward<FuncIdx>(compute_idx_func),
                           std::integral_constant<std::size_t, num_dims>{});
        }

        template <class Func>
        void multi_dim_loop(const auto& start, const auto& end, Func&& func2apply)
        {
            auto compute_idx_func = [](auto i)
            {
                return i;
            };
            multi_dim_loop(start, end, std::forward<Func>(func2apply), compute_idx_func);
        }

        template <typename Func, class... T>
            requires(std::same_as<std::decay_t<T>, std::array<typename std::decay_t<T>::value_type, std::tuple_size_v<std::decay_t<T>>>>
                     && ...)
        void multi_dim_loop(const std::tuple<T...>& coeff_arrays, Func&& func)
        {
            auto start = []<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return std::make_tuple(((void)Is, std::size_t{0})...);
            }(std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(coeff_arrays)>>>{});

            auto end = std::apply(
                [](auto&... coeff_arrays)
                {
                    return std::make_tuple(coeff_arrays.size()...);
                },
                coeff_arrays);

            multi_dim_loop(start, end, std::forward<Func>(func));
        }

        template <class Func, class... T>
            requires(std::same_as<std::decay_t<T>, Interval<typename std::decay_t<T>::value_t, typename std::decay_t<T>::index_t>> && ...)
        void multi_dim_loop(const std::tuple<T...>& interval_arrays, Func&& func)
        {
            auto [start, end] = std::apply(
                [](auto&... intervals)
                {
                    return std::make_pair(std::make_tuple(intervals.start...), std::make_tuple(intervals.end...));
                },
                interval_arrays);

            multi_dim_loop(start, end, std::forward<Func>(func));
        }

    }

    /**
     * Prediction stencil of a single fine child, as a @ref prediction_map.
     *
     * Returns the linear combination of coarse-cell values that predicts the child
     * whose integer position is @c indices, sitting @c level levels below a
     * reference coarse cell placed at the origin; the map offsets are in coarse-cell
     * units, relative to that reference. For @c indices in @c [0, 2^level)^dim the
     * child lies inside the reference cell; outside that box the stencil reaches into
     * the neighbouring coarse cells (this is how the LBM stream expresses a shift by
     * crossing coarse-cell boundaries).
     *
     * @tparam order  half-width of the wavelet interpolation, i.e.
     *                @c prediction_stencil_radius (uses @ref prediction_coefficients).
     * @param  cls    where the reference cell sits relative to the domain, see
     *                @ref PredictionPositionClass. Near a boundary the one-level stencils the
     *                map composes are shifted inward so that they read only cells the domain
     *                has, exactly as @c prediction_op does level by level; the class carries
     *                what that decision needs, so the map depends on it and on nothing else
     *                about the mesh. @ref PredictionPositionClass::interior gives the centred
     *                family everywhere, which is the map away from every boundary.
     * @param  level  the level gap @c delta_l (NOT an absolute level). @c 0 is the
     *                identity map @c {indices: 1} (the cell itself).
     *
     * Built by recursion on the gap: the child's parent is @c indices>>1 one level up,
     * and the odd/even parity @c indices&1 picks the interpolation coefficients per direction.
     * The map is the parent's stencil, weighted by the parent's own coefficient, plus the
     * tensor-product interpolation correction of the other stencil nodes, each itself a
     * @c level-1 stencil, so composing the one-level wavelet prediction @c level times. Every
     * intermediate map is memoised in a static cache keyed by @c (order, level, cls, indices).
     * The nodes are visited in the same order as before the stencils could shift, and a
     * centred parent carries the coefficient 1 exactly, so away from every boundary the maps
     * are bit for bit what they were.
     */
    template <std::size_t order = 1, class... index_t>
        requires(sizeof...(index_t) >= 1)
    auto&
    prediction(const PredictionPositionClass<prediction_class_reach<order>, sizeof...(index_t)>& cls, std::size_t level, index_t... indices)
    {
        static constexpr std::size_t dim = sizeof...(index_t);
        using class_t                    = PredictionPositionClass<prediction_class_reach<order>, dim>;
        using value_t                    = default_config::value_t;

        static std::unordered_map<std::tuple<std::size_t, std::size_t, class_t, index_t...>, prediction_map<dim, value_t>> values;

        auto key  = std::make_tuple(order, level, cls, indices...);
        auto iter = values.find(key);

        if (iter != values.end())
        {
            return iter->second;
        }

        if (level == 0)
        {
            values[key] = prediction_map<dim, value_t>{{indices...}};
            return values[key];
        }

        auto parent_indices = std::make_tuple((indices >> 1)...);
        auto parities       = std::make_tuple(static_cast<std::size_t>(indices & 1)...);

        // The stencil around the parent, which lives level - 1 gaps below the reference cell.
        const auto parent_array = std::apply(
            [](auto... p)
            {
                return std::array<value_t, dim>{static_cast<value_t>(p)...};
            },
            parent_indices);
        const auto shift = detail::class_shift<order, dim>(cls, level - 1, parent_array);

        std::array<int, dim> starts{};
        std::array<std::size_t, dim> parent_node{};
        auto interp_coeff_values = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            return std::make_tuple(
                [&]
                {
                    const auto& c   = prediction_coefficients<order>(std::get<Is>(parities), shift.fits ? shift.shift[Is] : 0);
                    starts[Is]      = c.start;
                    parent_node[Is] = static_cast<std::size_t>(-c.start);
                    return c.c;
                }()...);
        }(std::make_index_sequence<dim>{});

        // The parent's own contribution first: its map, weighted by the product of its
        // coefficients - exactly 1 for a centred stencil.
        double parent_weight = 1.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            parent_weight *= std::apply(
                [&](const auto&... c)
                {
                    std::size_t dd = 0;
                    double out     = 0.;
                    ((dd++ == d ? (out = c[parent_node[d]], 0) : 0), ...);
                    return out;
                },
                interp_coeff_values);
        }

        std::apply(
            [&](auto... parent_values)
            {
                values[key] = prediction<order, index_t...>(cls, level - 1, parent_values...);
            },
            parent_indices);
        if (parent_weight != 1.)
        {
            values[key] *= parent_weight;
        }

        detail::multi_dim_loop(interp_coeff_values,
                               [&](auto... loop_indices)
                               {
                                   std::size_t dd           = 0;
                                   const bool is_not_parent = ((loop_indices != parent_node[dd++]) || ...);

                                   if (is_not_parent)
                                   {
                                       double c = detail::compute_coeff(interp_coeff_values, std::make_tuple(loop_indices...));
                                       std::apply(
                                           [&](auto... offsets)
                                           {
                                               values[key] += c * prediction<order, index_t...>(cls, level - 1, offsets...);
                                           },
                                           detail::compute_shifted_indices(starts, parent_indices, std::make_tuple(loop_indices...)));
                                   }
                               });
        return values[key];
    }

    /// The map of a child of a reference cell away from every boundary: the centred family.
    template <std::size_t order = 1, class... index_t>
        requires(sizeof...(index_t) >= 1)
    auto& prediction(std::size_t level, index_t... indices)
    {
        static constexpr std::size_t dim = sizeof...(index_t);
        static const auto interior       = PredictionPositionClass<prediction_class_reach<order>, dim>::interior();
        return prediction<order, index_t...>(interior, level, indices...);
    }

    /**
     * Subset operator used by @ref reconstruction. On the intersection of a coarse
     * level with the uniform reconstruction level, it writes every fine child of the
     * coarse cells: for each of the @c 2^{delta_l.dim} sub-positions it applies the
     * @ref prediction stencil of that child. The whole coarse x-interval is handled at
     * once by addressing its fine cells with a strided interval (@c i_f.step), so the
     * prediction of a given sub-position is broadcast over the interval. @c delta_l == 0
     * (source already at the reconstruction level) is a plain copy. One overload per
     * dimension, same logic with one more nested sub-position loop.
     */
    template <std::size_t dim, class TInterval>
    class reconstruction_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(reconstruction_op_)

        // The runs of the interval over which the position class is constant, each with its
        // class: the stencils of the children of a coarse cell depend on where that cell sits
        // relative to the domain, and on a box domain the whole interval is one run away from
        // the boundary.
        template <std::size_t radius, class T2, class Func>
        SAMURAI_INLINE void for_each_class_run(const T2& src, Func&& f) const
        {
            for_each_prediction_position_run<prediction_class_reach<radius>>(src.mesh(), level, i, index, std::forward<Func>(f));
        }

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<1>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        {
            using index_t                                   = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_stencil_radius = T2::mesh_t::config_t::prediction_stencil_radius;

            std::size_t delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i) = src(level, i);
            }
            else
            {
                index_t nb_cells = 1 << delta_l;
                for_each_class_run<prediction_stencil_radius>(
                    src,
                    [&](const auto& run, const auto& cls)
                    {
                        for (index_t ii = 0; ii < nb_cells; ++ii)
                        {
                            const auto& pred = prediction<prediction_stencil_radius, index_t>(cls, delta_l, ii);
                            for (const auto& kv : pred.coeff)
                            {
                                auto i_f = (run << delta_l) + ii;
                                i_f.step = nb_cells;
                                dest(reconstruct_level, i_f) += kv.second * src(level, run + kv.first[0]);
                            }
                        }
                    });
            }
        }

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<2>, std::size_t& reconstruct_level, T1& dest, const T2& src) const

        {
            using index_t                                   = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_stencil_radius = T2::mesh_t::config_t::prediction_stencil_radius;

            std::size_t delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i, j) = src(level, i, j);
            }
            else
            {
                index_t nb_cells = 1 << delta_l;
                for_each_class_run<prediction_stencil_radius>(
                    src,
                    [&](const auto& run, const auto& cls)
                    {
                        for (index_t jj = 0; jj < nb_cells; ++jj)
                        {
                            auto j_f = (j << delta_l) + jj;
                            for (index_t ii = 0; ii < nb_cells; ++ii)
                            {
                                const auto& pred = prediction<prediction_stencil_radius, index_t>(cls, delta_l, ii, jj);
                                auto i_f         = (run << delta_l) + ii;
                                i_f.step         = nb_cells;

                                for (const auto& kv : pred.coeff)
                                {
                                    dest(reconstruct_level, i_f, j_f) += kv.second * src(level, run + kv.first[0], j + kv.first[1]);
                                }
                            }
                        }
                    });
            }
        }

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<3>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        {
            using index_t                                   = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_stencil_radius = T2::mesh_t::config_t::prediction_stencil_radius;

            std::size_t delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i, j, k) = src(level, i, j, k);
            }
            else
            {
                index_t nb_cells = 1 << delta_l;
                for_each_class_run<prediction_stencil_radius>(
                    src,
                    [&](const auto& run, const auto& cls)
                    {
                        for (index_t kk = 0; kk < nb_cells; ++kk)
                        {
                            auto k_f = (k << delta_l) + kk;
                            for (index_t jj = 0; jj < nb_cells; ++jj)
                            {
                                auto j_f = (j << delta_l) + jj;
                                for (index_t ii = 0; ii < nb_cells; ++ii)
                                {
                                    const auto& pred = prediction<prediction_stencil_radius, index_t>(cls, delta_l, ii, jj, kk);
                                    auto i_f         = (run << delta_l) + ii;
                                    i_f.step         = nb_cells;

                                    for (const auto& kv : pred.coeff)
                                    {
                                        dest(reconstruct_level,
                                             i_f,
                                             j_f,
                                             k_f) += kv.second * src(level, run + kv.first[0], j + kv.first[1], k + kv.first[2]);
                                    }
                                }
                            }
                        }
                    });
            }
        }
    };

    /// Wrap @ref reconstruction_op_ as a subset operator (see @c apply_op).
    template <class T1, class T2>
    SAMURAI_INLINE auto make_reconstruction(std::size_t& reconstruct_level, T1&& reconstruct_field, T2&& field)
    {
        return make_field_operator_function<reconstruction_op_>(reconstruct_level,
                                                                std::forward<T1>(reconstruct_field),
                                                                std::forward<T2>(field));
    }

    /**
     * Reconstruct an adapted field onto the uniform grid at the domain (finest) level
     * and return it as a new field on that grid. Every coarse cell is expanded into its
     * fine children by @ref reconstruction_op_. Mainly for I/O and post-processing, where
     * a single-resolution image of the solution is wanted.
     *
     * Requires at least two boundary ghosts (the prediction stencil reaches two coarse
     * cells); throws otherwise. Ghosts are refreshed first via @c update_ghost_mr_if_needed.
     */
    template <class Field>
    auto reconstruction(Field& field)
    {
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        using ca_type   = typename mesh_t::ca_type;

        // 2r: the composed prediction map reaches 2r coarse cells, and a stencil clamped at a
        // boundary reaches 2r inward.
        if (field.mesh().max_stencil_radius() < 2 * static_cast<int>(mesh_t::config_t::prediction_stencil_radius))
        {
            throw std::runtime_error("The reconstruction function requires at least 2 * prediction_stencil_radius ghosts on the "
                                     "boundary.\nTo fix this issue, remove mesh_config.disable_minimal_ghost_width().");
        }

        update_ghost_mr_if_needed(field);

        auto make_field_like = [](const std::string& name, auto& mesh)
        {
            if constexpr (Field::is_scalar)
            {
                return make_scalar_field<typename Field::value_type>(name, mesh);
            }
            else
            {
                return make_vector_field<typename Field::value_type, Field::n_comp>(name, mesh);
            }
        };

        auto& mesh = field.mesh();
        ca_type reconstruct_mesh;
        std::size_t reconstruct_level       = mesh.domain().level();
        reconstruct_mesh[reconstruct_level] = mesh.domain();
        reconstruct_mesh.update_index();

        auto m                 = holder(reconstruct_mesh);
        auto reconstruct_field = make_field_like(field.name(), m);
        reconstruct_field.fill(0.);

        std::size_t min_level = mesh[mesh_id_t::cells].min_level();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto set = intersection(mesh[mesh_id_t::cells][level], reconstruct_mesh[reconstruct_level]).on(level);
            set.apply_op(make_reconstruction(reconstruct_level, reconstruct_field, field));
        }
        return reconstruct_field;
    }

    namespace detail
    {
        /**
         * Sum into @a out the @ref prediction stencils of every child of a signed box
         * @c [start, end) of child indices, one @ref prediction call per child.
         * Compile-time recursion over the directions with a @e signed @c value_t counter:
         * child indices may be negative (e.g. the LBM donor slice shifted by @c -c),
         * which the @c std::size_t @ref multi_dim_loop cannot iterate.
         */
        template <std::size_t prediction_stencil_radius, class value_t, std::size_t dim, class PMap, class Class, class Prefix>
        void accumulate_slice(PMap& out,
                              const Class& cls,
                              std::size_t delta_l,
                              const std::array<value_t, dim>& start,
                              const std::array<value_t, dim>& end,
                              const Prefix& prefix)
        {
            constexpr std::size_t d = std::tuple_size_v<Prefix>;
            if constexpr (d == dim)
            {
                out += std::apply(
                    [&](auto... idx)
                    {
                        return prediction<prediction_stencil_radius, value_t>(cls, delta_l, idx...);
                    },
                    prefix);
            }
            else
            {
                for (value_t k = start[d]; k < end[d]; ++k)
                {
                    accumulate_slice<prediction_stencil_radius, value_t, dim>(out,
                                                                              cls,
                                                                              delta_l,
                                                                              start,
                                                                              end,
                                                                              std::tuple_cat(prefix, std::make_tuple(k)));
                }
            }
        }

        /// The position class the composed maps of a field of this radius are keyed on.
        template <std::size_t prediction_stencil_radius, std::size_t dim>
        using prediction_class_t = PredictionPositionClass<prediction_class_reach<prediction_stencil_radius>, dim>;

        /**
         * Prediction stencil to apply for a @ref portion request, dispatched on the type
         * of the child index @a ii. Two forms:
         *
         *  - @b scalar (@c ii all @c value_t): the stencil of that single child, i.e.
         *    @ref prediction "prediction(delta_l, ii)". The scalar overload just forwards
         *    to the already-cached @ref prediction and ignores @a level.
         *
         *  - @b slice (@c ii all @c interval_t, this overload): each direction is a range
         *    of children, so @a ii describes a whole box of them. Reconstruction is linear,
         *    so the box's stencil is the sum of its children's stencils; that sum is built
         *    once (via @ref accumulate_slice) and cached, keyed by @c (radius, level, ii).
         *    A later @ref portion then reconstructs and sums the whole box in one pass over
         *    a small, gap-independent stencil, instead of one call per child. This is what
         *    makes the LBM stream cheap (see @c LBMScheme::portion_column); a nonlinear user
         *    such as the FV flux must instead recompute each child with the scalar form.
         *
         * @param cls    where the coarse cell sits relative to the domain, which the maps
         *               depend on near a boundary (see @ref PredictionPositionClass).
         * @param level  coarse level where the field is read; only a cache discriminator,
         *               the stencil itself depends on @c delta_l, @a cls and @a ii alone.
         */
        template <std::size_t prediction_stencil_radius, class Field, class... index_t>
            requires(Field::dim == sizeof...(index_t) && (std::same_as<typename Field::interval_t, index_t> && ...))
        decltype(auto) get_prediction(const prediction_class_t<prediction_stencil_radius, Field::dim>& cls,
                                      std::size_t level,
                                      std::size_t delta_l,
                                      const std::tuple<index_t...>& ii)
        {
            static constexpr std::size_t dim = Field::dim;
            using value_t                    = typename Field::interval_t::value_t;
            using class_t                    = prediction_class_t<prediction_stencil_radius, dim>;
            static std::unordered_map<std::tuple<std::size_t, std::size_t, class_t, index_t...>, prediction_map<dim, value_t>> values;

            auto& map = std::apply(
                [&](auto&... index) -> auto&
                {
                    return values[{prediction_stencil_radius, level, cls, index...}];
                },
                ii);

            if (map.coeff.empty())
            {
                const auto start = std::apply(
                    [](const auto&... iv)
                    {
                        return std::array<value_t, dim>{iv.start...};
                    },
                    ii);
                const auto end = std::apply(
                    [](const auto&... iv)
                    {
                        return std::array<value_t, dim>{iv.end...};
                    },
                    ii);
                accumulate_slice<prediction_stencil_radius, value_t, dim>(map, cls, delta_l, start, end, std::tuple<>{});
            }

            return map;
        }

        /// Scalar form of @ref get_prediction: the stencil of the single child @a ii.
        template <std::size_t prediction_stencil_radius, class Field, class... index_t>
            requires((std::same_as<typename Field::interval_t::value_t, index_t> && ...))
        decltype(auto) get_prediction(const prediction_class_t<prediction_stencil_radius, Field::dim>& cls,
                                      std::size_t,
                                      std::size_t delta_l,
                                      const std::tuple<index_t...>& ii)
        {
            using value_t = typename Field::interval_t::value_t;
            return std::apply(
                [&](const auto&... index) -> auto&
                {
                    return prediction<prediction_stencil_radius, value_t>(cls, delta_l, index...);
                },
                ii);
        }

        /**
         * Apply the @ref get_prediction stencil of the child(ren) @a ii to a field, over a
         * whole coarse cell / interval: @c result = sum_k weight_k * get_f(level, i + offset_k).
         * @a get_f reads the field (a component, the full vector, ...); @a i is the coarse
         * location, its first entry an interval so the whole row is reconstructed at once, its
         * remaining entries the transverse coarse indices. @a ii selects the child (scalar) or
         * the child box (interval slice, summed by @ref get_prediction).
         */
        template <std::size_t prediction_stencil_radius, class Field, class Func, class... index_t, class... cell_index_t>
            requires(Field::dim == sizeof...(index_t) + 1 && Field::dim == sizeof...(cell_index_t)
                     && ((std::same_as<typename Field::interval_t, cell_index_t> && ...)
                         || (std::same_as<typename Field::interval_t::value_t, cell_index_t> && ...))
                     && (std::same_as<typename Field::interval_t::value_t, index_t> && ...))
        void portion_impl(auto& result,
                          Func&& get_f,
                          const Field& f,
                          std::size_t level,
                          std::size_t delta_l,
                          const std::tuple<typename Field::interval_t, index_t...>& i,
                          const std::tuple<cell_index_t...>& ii)
        {
            using result_t            = std::decay_t<decltype(result)>;
            using interval_t          = typename Field::interval_t;
            using value_t             = typename interval_t::value_t;
            constexpr std::size_t dim = Field::dim;

            if constexpr (std::is_same_v<result_t, double>)
            {
                result = 0.;
            }
            else
            {
                result.fill(0.);
            }

            // The stencils depend on where the coarse cells sit relative to the domain, so the
            // row is reconstructed run by run of constant position class; away from every
            // boundary the whole row is one run.
            const auto& row = std::get<0>(i);
            xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> index;
            [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                ((index[Is] = std::get<Is + 1>(i)), ...);
            }(std::make_index_sequence<dim - 1>{});

            for_each_prediction_position_run<prediction_class_reach<prediction_stencil_radius>>(
                f.mesh(),
                level,
                row,
                index,
                [&](const interval_t& run, const auto& cls)
                {
                    const auto& pred = get_prediction<prediction_stencil_radius, Field>(cls, level, delta_l, ii);

                    auto i_run         = i;
                    std::get<0>(i_run) = run;

                    for (const auto& kv : pred.coeff)
                    {
                        std::apply(
                            [&](auto... indices)
                            {
                                if constexpr (std::is_same_v<result_t, double>)
                                {
                                    result += kv.second * get_f(level, indices...)[0];
                                }
                                else if (run.start == row.start && run.end == row.end)
                                {
                                    result += kv.second * get_f(level, indices...);
                                }
                                else
                                {
                                    xt::view(result, xt::range(run.start - row.start, run.end - row.start)) += kv.second
                                                                                                             * get_f(level, indices...);
                                }
                            },
                            detail::compute_new_indices(0, i_run, kv.first));
                    }
                });
        }
    }

    /**
     * Reconstructed value of the child(ren) @a ii of the coarse cell(s) @a i, for a field
     * @a f stored @a delta_l levels coarser than those children.
     *
     * @param f       the field, read at @a level.
     * @param element (per-component overloads only) reconstruct just this component.
     * @param level   coarse level where @a f is read.
     * @param delta_l level gap between @a f and the children (0 = same level = plain read).
     * @param i       coarse location: its first entry is an interval (the reconstruction is
     *                vectorised over that whole row of coarse cells), the rest are the
     *                transverse coarse indices.
     * @param ii      one child index per direction: a @c value_t picks a single child, an
     *                @c interval_t sums the whole child slice at once (see @ref get_prediction).
     *
     * The @c (result, ...) overloads accumulate into a caller-provided buffer; the returning
     * overloads allocate a zeroed result first. The stencil half-width defaults to the mesh's
     * @c prediction_stencil_radius, or is given explicitly as the first template argument.
     */
    template <class Field, class... index_t, class... cell_index_t>
    void portion(auto& result,
                 const Field& f,
                 std::size_t element,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        auto get_f = [&](std::size_t level, const auto&... indices)
        {
            return f(element, level, indices...);
        };
        detail::portion_impl<Field::mesh_t::config_t::prediction_stencil_radius, Field>(result, get_f, f, level, delta_l, i, ii);
    }

    template <class Field, class... index_t, class... cell_index_t>
    auto portion(const Field& f,
                 std::size_t element,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        auto result = std::apply(
            [&](const auto&... indices)
            {
                return zeros_like(f(element, level, indices...));
            },
            i);
        portion(result, f, element, level, delta_l, i, ii);
        return result;
    }

    /// Whole-field @ref portion (no @c element): reconstructs every component at once.
    template <std::size_t prediction_stencil_radius, class Field, class... index_t, class... cell_index_t>
    void portion(auto& result,
                 const Field& f,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        auto get_f = [&](std::size_t level, const auto&... indices)
        {
            return f(level, indices...);
        };

        detail::portion_impl<prediction_stencil_radius, Field>(result, get_f, f, level, delta_l, i, ii);
    }

    template <class Field, class... index_t, class... cell_index_t>
    void portion(auto& result,
                 const Field& f,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        portion<Field::mesh_t::config_t::prediction_stencil_radius>(result, f, level, delta_l, i, ii);
    }

    template <std::size_t prediction_stencil_radius, class Field, class... index_t, class... cell_index_t>
    auto portion(const Field& f,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        auto result = std::apply(
            [&](const auto&... indices)
            {
                return zeros_like(f(level, indices...));
            },
            i);
        portion<prediction_stencil_radius>(result, f, level, delta_l, i, ii);
        return result;
    }

    template <class Field, class... index_t, class... cell_index_t>
    auto portion(const Field& f,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        return portion<Field::mesh_t::config_t::prediction_stencil_radius>(f, level, delta_l, i, ii);
    }

    namespace detail
    {
        /// Turn a coarse-cell index array into the @a i tuple @ref portion expects: the x index
        /// becomes the degenerate interval @c [x, x+1), the transverse indices are kept scalar.
        template <std::size_t dim, class interval_t>
        auto extract_src_tuple(const auto& src_indices)
        {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return std::make_tuple(interval_t{src_indices[0], src_indices[0] + 1}, ((void)Is, src_indices[Is + 1])...);
            }(std::make_index_sequence<dim - 1>{});
        }

        /// Turn a child index array into the @a ii tuple @ref portion expects (kept scalar,
        /// one child per direction).
        template <std::size_t dim>
        auto extract_dst_tuple([[maybe_unused]] auto delta_l, const auto& dst_indices)
        {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                // assert((dst_indices[Is] <= (1 << delta_l)) && ...); // doesn't compile on linux

                return std::make_tuple(((void)Is, dst_indices[Is])...);
            }(std::make_index_sequence<dim>{});
        }
    }

    /// Single-cell @ref portion taking plain index arrays (a coarse cell and one of its
    /// children) instead of the interval tuples; convenience for @ref transfer. Reconstructs
    /// component 0 for a vector field.
    template <class Field>
    void portion(auto& result,
                 const Field& f,
                 std::size_t level,
                 std::size_t delta_l,
                 const typename Field::cell_t::indices_t& src_indices,
                 const typename Field::cell_t::indices_t& dst_indices)
    {
        static constexpr std::size_t dim = Field::dim;
        using interval_t                 = typename Field::interval_t;
        static_assert(dim <= 3, "Not implemented for dim > 3");

        auto get_f = [&](std::size_t level, const auto&... indices)
        {
            if constexpr (Field::is_scalar)
            {
                return f(level, indices...);
            }
            else
            {
                return xt::view(f(level, indices...), 0);
            }
        };

        auto src_tuple = detail::extract_src_tuple<dim, interval_t>(src_indices);
        auto dst_tuple = detail::extract_dst_tuple<dim>(delta_l, dst_indices);

        detail::portion_impl<Field::mesh_t::config_t::prediction_stencil_radius, Field>(result, get_f, f, level, delta_l, src_tuple, dst_tuple);
    }

    /**
     * Copy a field from one adapted mesh onto another (of the same domain), resampling where
     * the two meshes differ in resolution. For each destination level, a cell is filled from
     * the source by one of three cases: an exact copy where the source has the same cell; a
     * projection (average of the @c 2^{shift.dim} finer source cells) where the source is finer;
     * or a @ref portion prediction where the source is coarser. Requires at least two boundary
     * ghosts on the source (prediction stencil); throws otherwise.
     */
    template <class Field_src, class Field_dst>
    void transfer(Field_src& field_src, Field_dst& field_dst)
    {
        static constexpr std::size_t dim = Field_src::dim;
        using mesh_id_t                  = typename Field_src::mesh_t::mesh_id_t;
        using interval_t                 = typename Field_src::mesh_t::interval_t;
        using size_type                  = typename Field_src::size_type;
        using value_t                    = typename interval_t::value_t;
        auto& mesh_src                   = field_src.mesh();
        auto& mesh_dst                   = field_dst.mesh();

        if (field_src.mesh().max_stencil_radius()
            < 2 * static_cast<int>(std::decay_t<decltype(mesh_src)>::config_t::prediction_stencil_radius))
        {
            throw std::runtime_error("The transfer function requires at least 2 * prediction_stencil_radius ghosts on the boundary.\nTo "
                                     "fix this issue, remove mesh_config.disable_minimal_ghost_width().");
        }

        update_ghost_mr_if_needed(field_src);

        field_dst.fill(0.);

        for (std::size_t level_dst = mesh_dst.min_level(); level_dst <= mesh_dst.max_level(); ++level_dst)
        {
            // Case 1: same cell on both meshes -> copy.
            auto same_cell = intersection(mesh_dst[mesh_id_t::cells][level_dst], mesh_src[mesh_id_t::cells][level_dst]);
            same_cell(
                [&](const auto& i, const auto& index)
                {
                    field_dst(level_dst, i, index) = field_src(level_dst, i, index);
                });

            // Case 2: source finer -> average its children onto the destination cell.
            for (std::size_t level_src = level_dst + 1; level_src <= mesh_src.max_level(); ++level_src)
            {
                auto proj_cell = intersection(mesh_dst[mesh_id_t::cells][level_dst], mesh_src[mesh_id_t::cells][level_src]).on(level_src);

                proj_cell(
                    [&](const auto& i, const auto& index)
                    {
                        std::size_t shift = level_src - level_dst;

                        auto src = field_src(level_src, i, index);
                        auto dst = field_dst(level_dst, i >> shift, index >> shift);
                        for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                        {
                            auto i_dst = static_cast<size_type>(((i.start + ii) >> static_cast<value_t>(shift))
                                                                - (i.start >> static_cast<value_t>(shift)));
#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
                            static_assert(sizeof(Field_src) == 0,
                                          "transfer() is not implemented with the Eigen field container (SAMURAI_FIELD_CONTAINER_EIGEN3) "
                                          "for scalar and vectorial fields. Use the xtensor field container (default) instead.");
                        // In the lid-driven-cavity demo, the following line of code does not compile with Eigen.
#endif
                            view(dst, i_dst) += view(src, static_cast<size_type>(ii)) / (1 << shift * dim);
                        }
                    });
            }

            // Case 3: source coarser -> predict each destination child from it (see @ref portion).
            for (std::size_t level_src = mesh_src.min_level(); level_src < level_dst; ++level_src)
            {
                auto pred_cell = intersection(mesh_dst[mesh_id_t::cells][level_dst], mesh_src[mesh_id_t::cells][level_src]).on(level_dst);

                pred_cell(
                    [&](const auto& i, const auto& index)
                    {
                        auto shift = level_dst - level_src;
                        if constexpr (dim == 1)
                        {
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto dst   = field_dst(level_dst, interval_t{i.start + ii, i.start + ii + 1});
                                auto i_src = (i.start + static_cast<value_t>(ii)) >> shift;
                                portion(dst,
                                        field_src,
                                        level_src,
                                        shift,
                                        std::make_tuple(interval_t{i_src, i_src + 1}),
                                        std::make_tuple(i.start + ii - (i_src << static_cast<value_t>(shift))));
                            }
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j = index[0];
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto dst   = field_dst(level_dst, interval_t{i.start + ii, i.start + ii + 1}, j);
                                auto i_src = (i.start + static_cast<value_t>(ii)) >> shift;
                                auto j_src = j >> shift;
                                portion(dst,
                                        field_src,
                                        level_src,
                                        shift,
                                        std::make_tuple(interval_t{i_src, i_src + 1}, j_src),
                                        std::make_tuple(i.start + ii - (i_src << static_cast<value_t>(shift)), j - (j_src << shift)));
                            }
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j = index[0];
                            auto k = index[1];
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto dst   = field_dst(level_dst, interval_t{i.start + ii, i.start + ii + 1}, j, k);
                                auto i_src = (i.start + static_cast<value_t>(ii)) >> shift;
                                auto j_src = j >> shift;
                                auto k_src = k >> shift;
                                portion(dst,
                                        field_src,
                                        level_src,
                                        shift,
                                        std::make_tuple(interval_t{i_src, i_src + 1}, j_src, k_src),
                                        std::make_tuple(i.start + ii - (i_src << static_cast<value_t>(shift)),
                                                        j - (j_src << shift),
                                                        k - (k_src << shift)));
                            }
                        }
                    });
            }
        }
    }
}
