#pragma once

#include <array>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tuple>
#include <unordered_map>

#include "field.hpp"
#include "numeric/prediction.hpp"
#include "samurai_config.hpp"
#include "subset/node.hpp"
#include "utils.hpp"

namespace samurai
{
    // Hash function for std::array
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

        void to_stream(std::ostream& out) const
        {
            for (const auto& c : coeff)
            {
                for (const auto& i : c.first)
                {
                }
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
    inline std::ostream& operator<<(std::ostream& out, const prediction_map<dim, index_t>& pred)
    {
        pred.to_stream(out);
        return out;
    }

    namespace detail
    {
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

        template <std::size_t... Is>
        auto compute_coeff(const auto& interp_coeffs, const auto& indices, std::index_sequence<Is...>)
        {
            return (std::get<Is>(interp_coeffs)[std::get<Is>(indices)] * ...);
        }

        auto compute_coeff(const auto& interp_coeffs, const auto& indices)
        {
            return compute_coeff(interp_coeffs, indices, std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(interp_coeffs)>>>{});
        }

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

    template <std::size_t order = 1, class... index_t>
    auto& prediction(std::size_t level, index_t... indices)
    {
        static constexpr std::size_t dim = sizeof...(index_t);

        static std::unordered_map<std::tuple<std::size_t, std::size_t, index_t...>, prediction_map<dim, default_config::value_t>> values;

        auto key  = std::make_tuple(order, level, indices...);
        auto iter = values.find(key);

        if (iter != values.end())
        {
            return iter->second;
        }

        if (level == 0)
        {
            values[key] = prediction_map<dim, default_config::value_t>{{indices...}};
            return values[key];
        }

        auto parent_indices = std::make_tuple((indices >> 1)...);
        auto signs          = std::make_tuple(((indices & 1) ? -1. : 1.)...);

        std::apply(
            [&](auto... parent_values)
            {
                values[key] = prediction<order, index_t...>(level - 1, parent_values...);
            },
            parent_indices);

        auto interp_coeff_values = std::apply(
            [&](auto... sign_values)
            {
                return std::make_tuple(interp_coeffs<2 * order + 1>(sign_values)...);
            },
            signs);

        detail::multi_dim_loop(interp_coeff_values,
                               [&](auto... loop_indices)
                               {
                                   bool is_not_center = ((loop_indices != order) || ...);

                                   if (is_not_center)
                                   {
                                       double c = detail::compute_coeff(interp_coeff_values, std::make_tuple(loop_indices...));
                                       std::apply(
                                           [&](auto... offsets)
                                           {
                                               values[key] += c * prediction<order, index_t...>(level - 1, offsets...);
                                           },
                                           detail::compute_new_indices(order, parent_indices, std::make_tuple(loop_indices...)));
                                   }
                               });
        return values[key];
    }

    template <std::size_t dim, class TInterval>
    class reconstruction_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(reconstruction_op_)

        // template <std::size_t d, class T1, class T2>
        // inline void operator()(Dim<d>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        // {
        //     using index_t                          = typename T2::interval_t::value_t;
        //     constexpr std::size_t prediction_order = T2::mesh_t::config::prediction_order;

        //     std::size_t delta_l = reconstruct_level - level;
        //     if (delta_l == 0)
        //     {
        //         dest(level, i, index) = src(level, i, index);
        //     }
        //     else
        //     {
        //         index_t nb_cells = 1 << delta_l;
        //         detail::multi_dim_loop(interp_coeff_values,
        //                                  [&](auto&... out_indices)
        //                                  {
        //                                      auto& pred = prediction<prediction_order, index_t>(delta_l, out_indices...);
        //                                      for (const auto& kv : pred.coeff)
        //                                      {
        //                                          std::apply(
        //                                              [&](auto&... in_indices)
        //                                              {
        //                                                  dest(reconstruct_level, out_indices...) += kv.second * src(level,
        //                                                  in_indices...);
        //                                              },
        //                                              detail::compute_new_indices(0, i, kv.first));
        //                                      }
        //                                  });
        //     }
        // }

        template <class T1, class T2>
        inline void operator()(Dim<1>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        {
            using index_t                          = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_order = T2::mesh_t::config::prediction_order;

            std::size_t delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i) = src(level, i);
            }
            else
            {
                index_t nb_cells = 1 << delta_l;
                for (index_t ii = 0; ii < nb_cells; ++ii)
                {
                    auto pred = prediction<prediction_order, index_t>(delta_l, ii);
                    for (const auto& kv : pred.coeff)
                    {
                        auto i_f = (i << delta_l) + ii;
                        i_f.step = nb_cells;
                        dest(reconstruct_level, i_f) += kv.second * src(level, i + kv.first[0]);
                    }
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<2>, std::size_t& reconstruct_level, T1& dest, const T2& src) const

        {
            using index_t                          = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_order = T2::mesh_t::config::prediction_order;

            std::size_t delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i, j) = src(level, i, j);
            }
            else
            {
                index_t nb_cells = 1 << delta_l;
                for (index_t jj = 0; jj < nb_cells; ++jj)
                {
                    auto j_f = (j << delta_l) + jj;
                    for (index_t ii = 0; ii < nb_cells; ++ii)
                    {
                        const auto& pred = prediction<prediction_order, index_t>(delta_l, ii, jj);
                        auto i_f         = (i << delta_l) + ii;
                        i_f.step         = nb_cells;

                        for (const auto& kv : pred.coeff)
                        {
                            dest(reconstruct_level, i_f, j_f) += kv.second * src(level, i + kv.first[0], j + kv.first[1]);
                        }
                    }
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<3>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        {
            using index_t                          = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_order = T2::mesh_t::config::prediction_order;

            std::size_t delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i, j, k) = src(level, i, j, k);
            }
            else
            {
                index_t nb_cells = 1 << delta_l;
                for (index_t kk = 0; kk < nb_cells; ++kk)
                {
                    auto k_f = (k << delta_l) + kk;
                    for (index_t jj = 0; jj < nb_cells; ++jj)
                    {
                        auto j_f = (j << delta_l) + jj;
                        for (index_t ii = 0; ii < nb_cells; ++ii)
                        {
                            const auto& pred = prediction<prediction_order, index_t>(delta_l, ii, jj, kk);
                            auto i_f         = (i << delta_l) + ii;
                            i_f.step         = nb_cells;

                            for (const auto& kv : pred.coeff)
                            {
                                dest(reconstruct_level, i_f, j_f, k_f) += kv.second
                                                                        * src(level, i + kv.first[0], j + kv.first[1], k + kv.first[2]);
                            }
                        }
                    }
                }
            }
        }
    };

    template <class T1, class T2>
    inline auto make_reconstruction(std::size_t& reconstruct_level, T1&& reconstruct_field, T2&& field)
    {
        return make_field_operator_function<reconstruction_op_>(reconstruct_level,
                                                                std::forward<T1>(reconstruct_field),
                                                                std::forward<T2>(field));
    }

    template <class Field>
    auto reconstruction(Field& field)
    {
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        using ca_type   = typename mesh_t::ca_type;

        if (field.mesh().max_stencil_radius() < 2)
        {
            std::cerr << "The reconstruction function requires at least 2 ghosts on the boundary.\nTo fix this issue, set mesh_config.max_stencil_radius(2) or mesh_config.max_stencil_size(4)."
                      << std::endl;
            exit(EXIT_FAILURE);
        }

        update_ghost_mr_if_needed(field);

        auto make_field_like = [](std::string const& name, auto& mesh)
        {
            if constexpr (Field::is_scalar)
            {
                return make_scalar_field<typename Field::value_type>(name, mesh);
            }
            else
            {
                return make_vector_field<typename Field::value_type, Field::n_comp, detail::is_soa_v<Field>>(name, mesh);
            }
        };

        auto& mesh = field.mesh();
        ca_type reconstruct_mesh;
        std::size_t reconstruct_level       = mesh.domain().level();
        reconstruct_mesh[reconstruct_level] = mesh.domain();
        reconstruct_mesh.update_index();

        auto m = holder(reconstruct_mesh);
        // auto reconstruct_field = make_field<typename Field::value_type, Field::n_comp, detail::is_soa_v<Field>>(field.name(), m);
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
        template <std::size_t prediction_order, class Field, class... index_t>
            requires(Field::dim == sizeof...(index_t) + 1 && (std::same_as<typename Field::interval_t, index_t> && ...))
        decltype(auto) get_prediction(std::size_t level, std::size_t delta_l, const std::tuple<index_t...>& ii)
        {
            static constexpr std::size_t dim = Field::dim;
            using value_t                    = typename Field::interval_t::value_t;
            static std::unordered_map<std::tuple<std::size_t, std::size_t, typename Field::interval_t, index_t...>, prediction_map<dim, value_t>>
                values;

            auto iter = std::apply(
                [&](auto&... index)
                {
                    return values.find({prediction_order, level, index...});
                },
                ii);

            if (iter == values.end())
            {
                multi_dim_loop(
                    ii,
                    [&](auto... ii_)
                    {
                        std::apply(
                            [&](auto&... index)
                            {
                                values[{prediction_order, level, index...}] += prediction<prediction_order, value_t>(delta_l, ii_...);
                            },
                            ii);
                    });
            }

            return std::apply(
                [&](auto&... index) -> auto&
                {
                    return values[{prediction_order, level, index...}];
                },
                ii);
        }

        template <std::size_t prediction_order, class Field, class... index_t>
            requires((std::same_as<typename Field::interval_t::value_t, index_t> && ...))
        decltype(auto) get_prediction(std::size_t, std::size_t delta_l, const std::tuple<index_t...>& ii)
        {
            using value_t = typename Field::interval_t::value_t;
            return std::apply(
                [delta_l](const auto&... index) -> auto&
                {
                    return prediction<prediction_order, value_t>(delta_l, index...);
                },
                ii);
        }

        template <std::size_t prediction_order, class Field, class Func, class... index_t, class... cell_index_t>
            requires(Field::dim == sizeof...(index_t) + 1 && Field::dim == sizeof...(cell_index_t)
                     && ((std::same_as<typename Field::interval_t, cell_index_t> && ...)
                         || (std::same_as<typename Field::interval_t::value_t, cell_index_t> && ...))
                     && (std::same_as<typename Field::interval_t::value_t, index_t> && ...))
        void portion_impl(auto& result,
                          Func&& get_f,
                          std::size_t level,
                          std::size_t delta_l,
                          const std::tuple<typename Field::interval_t, index_t...>& i,
                          const std::tuple<cell_index_t...>& ii)
        {
            using result_t = std::decay_t<decltype(result)>;

            const auto& pred = get_prediction<prediction_order, Field>(level, delta_l, ii);

            if constexpr (std::is_same_v<result_t, double>)
            {
                result = 0.;
            }
            else
            {
                result.fill(0.);
            }

            for (const auto& kv : pred.coeff)
            {
                std::apply(
                    [&](auto... indices)
                    {
                        if constexpr (std::is_same_v<result_t, double>)
                        {
                            result += kv.second * get_f(level, indices...)[0];
                        }
                        else
                        {
                            result += kv.second * get_f(level, indices...);
                        }
                    },
                    detail::compute_new_indices(0, i, kv.first));
            }
        }
    }

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
        detail::portion_impl<Field::mesh_t::config::prediction_order, Field>(result, f, get_f, level, delta_l, i, ii);
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
        portion(result, f, level, delta_l, i, ii);
        return result;
    }

    template <std::size_t prediction_order, class Field, class... index_t, class... cell_index_t>
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

        detail::portion_impl<prediction_order, Field>(result, get_f, level, delta_l, i, ii);
    }

    template <class Field, class... index_t, class... cell_index_t>
    void portion(auto& result,
                 const Field& f,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        portion<Field::mesh_t::config::prediction_order>(result, f, level, delta_l, i, ii);
    }

    template <std::size_t prediction_order, class Field, class... index_t, class... cell_index_t>
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
        portion<prediction_order>(result, f, level, delta_l, i, ii);
        return result;
    }

    template <class Field, class... index_t, class... cell_index_t>
    auto portion(const Field& f,
                 std::size_t level,
                 std::size_t delta_l,
                 const std::tuple<typename Field::interval_t, index_t...>& i,
                 const std::tuple<cell_index_t...>& ii)
    {
        return portion<Field::mesh_t::config::prediction_order>(f, level, delta_l, i, ii);
    }

    namespace detail
    {
        template <std::size_t dim, class interval_t>
        auto extract_src_tuple(const auto& src_indices)
        {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return std::make_tuple(interval_t{src_indices[0], src_indices[0] + 1}, ((void)Is, src_indices[Is + 1])...);
            }(std::make_index_sequence<dim - 1>{});
        }

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

    // N-D
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
                if constexpr (Field::is_soa)
                {
                    return xt::view(f(level, indices...), xt::all(), 0);
                }
                else
                {
                    return xt::view(f(level, indices...), 0);
                }
            }
        };

        auto src_tuple = detail::extract_src_tuple<dim, interval_t>(src_indices);
        auto dst_tuple = detail::extract_dst_tuple<dim>(delta_l, dst_indices);

        detail::portion_impl<Field::mesh_t::config::prediction_order, Field>(result, get_f, level, delta_l, src_tuple, dst_tuple);
    }

    template <class Field_src, class Field_dst>
    void transfer(Field_src& field_src, Field_dst& field_dst)
    {
        static constexpr std::size_t dim = Field_src::dim;
        using mesh_id_t                  = typename Field_src::mesh_t::mesh_id_t;
        using interval_t                 = typename Field_src::interval_t;
        using size_type                  = typename Field_src::inner_types::size_type;
        using value_t                    = typename interval_t::value_t;
        auto& mesh_src                   = field_src.mesh();
        auto& mesh_dst                   = field_dst.mesh();

        if (field_src.mesh().max_stencil_radius() < 2)
        {
            std::cerr << "The transfert function requires at least 2 ghosts on the boundary.\nTo fix this issue, set mesh_config.max_stencil_radius(2) or mesh_config.max_stencil_size(4)."
                      << std::endl;
            exit(EXIT_FAILURE);
        }

        update_ghost_mr_if_needed(field_src);

        field_dst.fill(0.);

        for (std::size_t level_dst = mesh_dst.min_level(); level_dst <= mesh_dst.max_level(); ++level_dst)
        {
            auto same_cell = intersection(mesh_dst[mesh_id_t::cells][level_dst], mesh_src[mesh_id_t::cells][level_dst]);
            same_cell(
                [&](const auto& i, const auto& index)
                {
                    field_dst(level_dst, i, index) = field_src(level_dst, i, index);
                });

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
                            if constexpr (detail::is_soa_v<Field_src> && !Field_src::is_scalar)
                            {
                                view(dst, placeholders::all(), i_dst) += view(src, placeholders::all(), static_cast<size_type>(ii))
                                                                       / (1 << shift * dim);
                            }
                            else
                            {
#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
                                static_assert(detail::is_soa_v<Field_src> && !Field_src::is_scalar,
                                              "transfer() is not implemented with Eigen for scalar fields and vectorial fields in AOS.");
                            // In the lid-driven-cavity demo, the following line of code does not compile with Eigen.
#else
                                static_assert(Field_src::inner_types::static_layout == layout_type::row_major,
                                              "transfer() is not implemented when the xtensor within a field is col-major.");
                            // In the lid-driven-cavity demo, the following line of code crashes at execution in col_major.
#endif
                                view(dst, i_dst) += view(src, static_cast<size_type>(ii)) / (1 << shift * dim);
                            }
                        }
                    });
            }

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
