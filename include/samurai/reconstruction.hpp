#pragma once

#include "field.hpp"
#include "numeric/prediction.hpp"
#include "samurai_config.hpp"
#include "subset/subset_op.hpp"
#include <array>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tuple>

namespace samurai
{
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

        std::map<std::array<index_t, dim>, double> coeff;
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

    template <std::size_t order = 1, class index_t = default_config::value_t>
    auto prediction(std::size_t level, index_t i) -> prediction_map<1, index_t>
    {
        static std::map<std::tuple<std::size_t, std::size_t, index_t>, prediction_map<1, index_t>> values;

        if (level == 0)
        {
            return prediction_map<1, index_t>{{i}};
        }

        auto iter = values.find({order, level, i});

        if (iter == values.end())
        {
            int ig      = i >> 1;
            double sign = (i & 1) ? -1. : 1.;

            values[{order, level, i}] = prediction<order, index_t>(level - 1, ig);

            auto interp = interp_coeffs<2 * order + 1>(sign);
            for (std::size_t ci = 0; ci < interp.size(); ++ci)
            {
                if (ci != order)
                {
                    values[{order, level, i}] += interp[ci] * prediction<order, index_t>(level - 1, ig + static_cast<index_t>(ci - order));
                }
            }
            return values[{order, level, i}];
        }
        return iter->second;
    }

    template <std::size_t order = 1, class index_t = default_config::value_t>
    auto prediction(std::size_t level, index_t i, index_t j) -> prediction_map<2, index_t>
    {
        static std::map<std::tuple<std::size_t, std::size_t, index_t, index_t>, prediction_map<2, index_t>> values;

        if (level == 0)
        {
            return prediction_map<2, index_t>{
                {i, j}
            };
        }

        auto iter = values.find({order, level, i, j});

        if (iter == values.end())
        {
            int ig       = i >> 1;
            int jg       = j >> 1;
            double isign = (i & 1) ? -1. : 1.;
            double jsign = (j & 1) ? -1. : 1.;

            values[{order, level, i, j}] = prediction<order, index_t>(level - 1, ig, jg);

            auto interpx = interp_coeffs<2 * order + 1>(isign);
            auto interpy = interp_coeffs<2 * order + 1>(jsign);

            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
            {
                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                {
                    if (ci != order || cj != order)
                    {
                        values[{order, level, i, j}] += interpx[ci] * interpy[cj]
                                                      * prediction<order, index_t>(level - 1,
                                                                                   ig + static_cast<index_t>(ci - order),
                                                                                   jg + static_cast<index_t>(cj - order));
                    }
                }
            }
            return values[{order, level, i, j}];
        }
        return iter->second;
    }

    template <std::size_t order = 1, class index_t = default_config::value_t>
    auto prediction(std::size_t level, index_t i, index_t j, index_t k) -> prediction_map<3, index_t>
    {
        static std::map<std::tuple<std::size_t, std::size_t, index_t, index_t, index_t>, prediction_map<3, index_t>> values;

        if (level == 0)
        {
            return prediction_map<3, index_t>{
                {i, j, k}
            };
        }

        auto iter = values.find({order, level, i, j, k});

        if (iter == values.end())
        {
            int ig       = i >> 1;
            int jg       = j >> 1;
            int kg       = k >> 1;
            double isign = (i & 1) ? -1. : 1.;
            double jsign = (j & 1) ? -1. : 1.;
            double ksign = (k & 1) ? -1. : 1.;

            values[{order, level, i, j, k}] = prediction<order, index_t>(level - 1, ig, jg, kg);

            auto interpx = interp_coeffs<2 * order + 1>(isign);
            auto interpy = interp_coeffs<2 * order + 1>(jsign);
            auto interpz = interp_coeffs<2 * order + 1>(ksign);

            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
            {
                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                {
                    for (std::size_t ck = 0; ck < interpz.size(); ++ck)
                    {
                        if (ci != order || cj != order || ck != order)
                        {
                            values[{order, level, i, j, k}] += interpx[ci] * interpy[cj] * interpz[ck]
                                                             * prediction<order, index_t>(level - 1,
                                                                                          ig + static_cast<index_t>(ci - order),
                                                                                          jg + static_cast<index_t>(cj - order),
                                                                                          kg + static_cast<index_t>(ck - order));
                        }
                    }
                }
            }
            return values[{order, level, i, j, k}];
        }
        return iter->second;
    }

    template <class TInterval>
    class reconstruction_op_ : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(reconstruction_op_)

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
                        auto pred = prediction<prediction_order, index_t>(delta_l, ii, jj);
                        auto i_f  = (i << delta_l) + ii;
                        i_f.step  = nb_cells;

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
                            auto pred = prediction<prediction_order, index_t>(delta_l, ii, jj, kk);
                            auto i_f  = (i << delta_l) + ii;
                            i_f.step  = nb_cells;

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
    auto reconstruction(const Field& field)
    {
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        using ca_type   = typename mesh_t::ca_type;

        auto& mesh = field.mesh();
        ca_type reconstruct_mesh;
        std::size_t reconstruct_level       = mesh.domain().level();
        reconstruct_mesh[reconstruct_level] = mesh.domain();
        reconstruct_mesh.update_index();

        auto m                 = holder(reconstruct_mesh);
        auto reconstruct_field = make_field<typename Field::value_type, Field::size, Field::is_soa>(field.name(), m);
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
        // 1D portion
        template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
        auto
        portion_impl(const Field& f, std::size_t element, std::size_t level, const typename Field::interval_t& i, std::size_t delta_l, index_t ii)
        {
            auto pred = prediction<prediction_order, index_t>(delta_l, ii);

            auto result = xt::zeros_like(f(element, level, i));

            for (const auto& kv : pred.coeff)
            {
                // cppcheck-suppress useStlAlgorithm
                result += kv.second * f(element, level, i + kv.first[0]);
            }
            return result;
        }

        template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
        auto portion_impl(const Field& f, std::size_t level, const typename Field::interval_t& i, std::size_t delta_l, index_t ii)
        {
            auto pred = prediction<prediction_order, index_t>(delta_l, ii);

            auto result = xt::zeros_like(f(level, i));

            for (const auto& kv : pred.coeff)
            {
                // cppcheck-suppress useStlAlgorithm
                result += kv.second * f(level, i + kv.first[0]);
            }
            return result;
        }

        // 2D portion
        template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
        auto portion_impl(const Field& f,
                          std::size_t element,
                          std::size_t level,
                          const typename Field::interval_t& i,
                          index_t j,
                          std::size_t delta_l,
                          index_t ii,
                          index_t jj)
        {
            auto pred = prediction<prediction_order, index_t>(delta_l, ii, jj);

            auto result = xt::zeros_like(f(element, level, i, j));

            for (const auto& kv : pred.coeff)
            {
                // cppcheck-suppress useStlAlgorithm
                result += kv.second * f(element, level, i + kv.first[0], j + kv.first[1]);
            }
            return result;
        }

        template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
        auto
        portion_impl(const Field& f, std::size_t level, const typename Field::interval_t& i, index_t j, std::size_t delta_l, index_t ii, index_t jj)
        {
            auto pred = prediction<prediction_order, index_t>(delta_l, ii, jj);

            auto result = xt::zeros_like(f(level, i, j));

            for (const auto& kv : pred.coeff)
            {
                // cppcheck-suppress useStlAlgorithm
                result += kv.second * f(level, i + kv.first[0], j + kv.first[1]);
            }
            return result;
        }

        // 3D portion
        template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
        auto portion_impl(const Field& f,
                          std::size_t element,
                          std::size_t level,
                          const typename Field::interval_t& i,
                          index_t j,
                          index_t k,
                          std::size_t delta_l,
                          index_t ii,
                          index_t jj,
                          index_t kk)
        {
            auto pred = prediction<prediction_order, index_t>(delta_l, ii, jj, kk);

            auto result = xt::zeros_like(f(element, level, i, j, k));

            for (const auto& kv : pred.coeff)
            {
                // cppcheck-suppress useStlAlgorithm
                result += kv.second * f(element, level, i + kv.first[0], j + kv.first[1], k + kv.first[2]);
            }
            return result;
        }

        template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
        auto portion_impl(const Field& f,
                          std::size_t level,
                          const typename Field::interval_t& i,
                          index_t j,
                          index_t k,
                          std::size_t delta_l,
                          index_t ii,
                          index_t jj,
                          index_t kk)
        {
            auto pred = prediction<prediction_order, index_t>(delta_l, ii, jj, kk);

            auto result = xt::zeros_like(f(level, i, j, k));

            for (const auto& kv : pred.coeff)
            {
                // cppcheck-suppress useStlAlgorithm
                result += kv.second * f(level, i + kv.first[0], j + kv.first[1], k + kv.first[2]);
            }
            return result;
        }
    }

    // 1D
    template <class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f, std::size_t element, std::size_t level, const typename Field::interval_t& i, std::size_t delta_l, index_t ii)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, element, level, i, delta_l, ii);
    }

    template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f, std::size_t element, std::size_t level, const typename Field::interval_t& i, std::size_t delta_l, index_t ii)
    {
        return detail::portion_impl<prediction_order>(f, element, level, i, delta_l, ii);
    }

    template <class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f, std::size_t level, const typename Field::interval_t& i, std::size_t delta_l, index_t ii)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, level, i, delta_l, ii);
    }

    template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f, std::size_t level, const typename Field::interval_t& i, std::size_t delta_l, index_t ii)
    {
        return detail::portion_impl<prediction_order>(f, level, i, delta_l, ii);
    }

    // 2D
    template <class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f,
                 std::size_t element,
                 std::size_t level,
                 const typename Field::interval_t& i,
                 index_t j,
                 std::size_t delta_l,
                 index_t ii,
                 index_t jj)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, element, level, i, j, delta_l, ii, jj);
    }

    template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f,
                 std::size_t element,
                 std::size_t level,
                 const typename Field::interval_t& i,
                 index_t j,
                 std::size_t delta_l,
                 index_t ii,
                 index_t jj)
    {
        return detail::portion_impl<prediction_order>(f, element, level, i, j, delta_l, ii, jj);
    }

    template <class Field, class index_t = typename Field::interval_t::value_t>
    auto
    portion(const Field& f, std::size_t level, const typename Field::interval_t& i, index_t j, std::size_t delta_l, index_t ii, index_t jj)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, level, i, j, delta_l, ii, jj);
    }

    template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
    auto
    portion(const Field& f, std::size_t level, const typename Field::interval_t& i, index_t j, std::size_t delta_l, index_t ii, index_t jj)
    {
        return detail::portion_impl<prediction_order>(f, level, i, j, delta_l, ii, jj);
    }

    // 3D
    template <class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f,
                 std::size_t element,
                 std::size_t level,
                 const typename Field::interval_t& i,
                 typename Field::interval_t::value_t j,
                 typename Field::interval_t::value_t k,
                 std::size_t delta_l,
                 std::size_t ii,
                 std::size_t jj,
                 std::size_t kk)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, element, level, i, j, k, delta_l, ii, jj, kk);
    }

    template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f,
                 std::size_t element,
                 std::size_t level,
                 const typename Field::interval_t& i,
                 typename Field::interval_t::value_t j,
                 typename Field::interval_t::value_t k,
                 std::size_t delta_l,
                 std::size_t ii,
                 std::size_t jj,
                 std::size_t kk)
    {
        return detail::portion_impl<prediction_order>(f, element, level, i, j, k, delta_l, ii, jj, kk);
    }

    template <class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f,
                 std::size_t level,
                 const typename Field::interval_t& i,
                 index_t j,
                 index_t k,
                 std::size_t delta_l,
                 index_t ii,
                 index_t jj,
                 index_t kk)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, level, i, j, k, delta_l, ii, jj, kk);
    }

    template <std::size_t prediction_order, class Field, class index_t = typename Field::interval_t::value_t>
    auto portion(const Field& f,
                 std::size_t level,
                 const typename Field::interval_t& i,
                 index_t j,
                 index_t k,
                 std::size_t delta_l,
                 index_t ii,
                 index_t jj,
                 index_t kk)
    {
        return detail::portion_impl<prediction_order>(f, level, i, j, k, delta_l, ii, jj, kk);
    }

    template <class Field_src, class Field_dst>
    void transfer(Field_src& field_src, Field_dst& field_dst)
    {
        static constexpr std::size_t dim = Field_src::dim;
        using mesh_id_t                  = typename Field_src::mesh_t::mesh_id_t;
        using interval_t                 = typename Field_src::interval_t;
        using value_t                    = typename interval_t::value_t;
        auto& mesh_src                   = field_src.mesh();
        auto& mesh_dst                   = field_dst.mesh();

        field_dst.fill(0.);

        for (std::size_t level_dst = mesh_dst.min_level(); level_dst <= mesh_dst.max_level(); ++level_dst)
        {
            auto same_cell = intersection(mesh_dst[mesh_id_t::cells][level_dst], mesh_src[mesh_id_t::cells][level_dst]);
            same_cell(
                [&](const auto& i, const auto& index)
                {
                    if constexpr (dim == 1)
                    {
                        field_dst(level_dst, i) = field_src(level_dst, i);
                    }
                    else if constexpr (dim == 2)
                    {
                        auto j                     = index[0];
                        field_dst(level_dst, i, j) = field_src(level_dst, i, j);
                    }
                    else if constexpr (dim == 3)
                    {
                        auto j                        = index[0];
                        auto k                        = index[1];
                        field_dst(level_dst, i, j, k) = field_src(level_dst, i, j, k);
                    }
                });

            for (std::size_t level_src = level_dst + 1; level_src <= mesh_src.max_level(); ++level_src)
            {
                auto proj_cell = intersection(mesh_dst[mesh_id_t::cells][level_dst], mesh_src[mesh_id_t::cells][level_src]).on(level_src);

                proj_cell(
                    [&](const auto& i, const auto& index)
                    {
                        std::size_t shift = level_src - level_dst;

                        if constexpr (dim == 1)
                        {
                            auto src = field_src(level_src, i);
                            auto dst = field_dst(level_dst, i >> shift);
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto i_dst = static_cast<std::size_t>(((i.start + ii) >> static_cast<value_t>(shift))
                                                                      - (i.start >> static_cast<value_t>(shift)));
                                dst(i_dst) += src(ii) / (1 << shift);
                            }
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j   = index[0];
                            auto src = field_src(level_src, i, j);
                            auto dst = field_dst(level_dst, i >> shift, j >> shift);
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto i_dst = static_cast<std::size_t>(((i.start + ii) >> static_cast<value_t>(shift))
                                                                      - (i.start >> static_cast<value_t>(shift)));
                                dst(i_dst) += src(ii) / (1 << (shift * dim));
                            }
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j   = index[0];
                            auto k   = index[1];
                            auto src = field_src(level_src, i, j, k);
                            auto dst = field_dst(level_dst, i >> shift, j >> shift, k >> shift);
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto i_dst = static_cast<std::size_t>(((i.start + ii) >> static_cast<value_t>(shift))
                                                                      - (i.start >> static_cast<value_t>(shift)));
                                dst(i_dst) += src(ii) / (1 << (shift * dim));
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
                            auto dst = field_dst(level_dst, i);
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto i_src = (i.start + static_cast<value_t>(ii)) >> shift;
                                dst(ii)    = portion(field_src,
                                                  level_src,
                                                  interval_t{i_src, i_src + 1},
                                                  shift,
                                                  i.start + ii - (i_src << static_cast<value_t>(shift)))[0];
                            }
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j   = index[0];
                            auto dst = field_dst(level_dst, i, j);
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto i_src = (i.start + static_cast<value_t>(ii)) >> shift;
                                auto j_src = j >> shift;
                                dst(ii)    = portion(field_src,
                                                  level_src,
                                                  interval_t{i_src, i_src + 1},
                                                  j_src,
                                                  shift,
                                                  i.start + ii - (i_src << static_cast<value_t>(shift)),
                                                  j - (j_src << shift))[0];
                            }
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j   = index[0];
                            auto k   = index[1];
                            auto dst = field_dst(level_dst, i, j, k);
                            for (value_t ii = 0; ii < static_cast<value_t>(i.size()); ++ii)
                            {
                                auto i_src = (i.start + static_cast<value_t>(ii)) >> shift;
                                auto j_src = j >> shift;
                                auto k_src = k >> shift;
                                dst(ii)    = portion(field_src,
                                                  level_src,
                                                  interval_t{i_src, i_src + 1},
                                                  j_src,
                                                  k_src,
                                                  shift,
                                                  i.start + ii - (i_src << static_cast<value_t>(shift)),
                                                  j - (j_src << shift),
                                                  k - (k_src << shift))[0];
                            }
                        }
                    });
            }
        }
    }

}
