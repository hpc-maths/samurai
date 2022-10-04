#pragma once

#include <array>
#include <tuple>
#include <fmt/format.h>
#include "samurai_config.hpp"

namespace samurai
{
    template<std::size_t dim, class index_t=samurai::default_config::value_t>
    class prediction_map
    {
        public:
            using key_t = std::array<index_t, dim>;

            prediction_map() = default;

            prediction_map(const key_t & k)
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
                for(auto& c: p.coeff)
                {
                    auto& cc = (*this)(c.first);
                    cc += c.second;
                }
                return *this;
            }

            prediction_map& operator-=(const prediction_map& p)
            {
                for(auto& c: p.coeff)
                {
                    auto& cc = (*this)(c.first);
                    cc -= c.second;
                }
                return *this;
            }

            prediction_map& operator*=(const double d)
            {
                std::size_t index=0;
                for(auto& c: coeff)
                {
                    c.second *= d;
                }
                return *this;
            }

            prediction_map& operator+=(const double d)
            {
                for(auto& c: coeff)
                {
                    c.second += d;
                }
                return *this;
            }

            void to_stream(std::ostream& out) const
            {
                for(auto &c: coeff)
                {
                    for(auto& i: c.first)
                    {

                    }
                    out << fmt::format("({}):  {}", c.first[0], c.second) << std::endl;
                }
            }

            std::map<std::array<index_t, dim>, double> coeff;
    };

    template<std::size_t dim, class index_t>
    auto operator+(const prediction_map<dim, index_t> &p1, const prediction_map<dim, index_t> &p2)
    {
        prediction_map<dim, index_t> that{p1};
        that += p2;
        return that;
    }

    template<std::size_t dim, class index_t>
    auto operator+(const double d, const prediction_map<dim, index_t> &p)
    {
        prediction_map<dim, index_t> that{p};
        that += d;
        return that;
    }

    template<std::size_t dim, class index_t>
    auto operator-(const prediction_map<dim, index_t> &p1, const prediction_map<dim, index_t> &p2)
    {
        prediction_map<dim, index_t> that{p1};
        that -= p2;
        return that;
    }

    template<std::size_t dim, class index_t>
    auto operator*(const double d, const prediction_map<dim, index_t> &p)
    {
        prediction_map<dim, index_t> that{p};
        that *= d;
        return that;
    }

    template<std::size_t dim, class index_t>
    inline std::ostream &operator<<(std::ostream &out, const prediction_map<dim, index_t> &pred)
    {
        pred.to_stream(out);
        return out;
    }

    template <std::size_t order=1, class index_t=samurai::default_config::value_t>
    auto prediction(std::size_t level, index_t i, bool reset=false)
    {
        static std::map<std::tuple<std::size_t, index_t>, prediction_map<1, index_t>> values;

        if (reset)
        {
            values.clear();
        }

        if (level == 0)
        {
            return prediction_map<1, index_t>{{i}};
        }

        auto iter = values.find({level, i});

        if (iter == values.end())
        {
            int ig = i >> 1;
            double sign = (i & 1)? -1.: 1.;

            values[{level, i}] = prediction<order, index_t>(level-1, ig);

            auto interp = interp_coeffs<2*order+1>(sign);
            for(std::size_t ci=0; ci<interp.size(); ++ci)
            {
                if (ci != order)
                {
                    values[{level, i}] += interp[ci]*prediction<order, index_t>(level-1, ig + ci - order);
                }
            }
            return values[{level, i}];
        }
        else
        {
            return iter->second;
        }
    }

    template <std::size_t order=1, class index_t=samurai::default_config::value_t>
    auto prediction_2d(std::size_t level, index_t i, index_t j, bool reset=false)
    {
        static std::map<std::tuple<std::size_t, index_t, index_t>, prediction_map<2, index_t>> values;

        if (reset)
        {
            values.clear();
        }

        if (level == 0)
        {
            return prediction_map<2, index_t>{{i, j}};
        }

        auto iter = values.find({level, i, j});

        if (iter == values.end())
        {
            int ig = i >> 1;
            int jg = j >> 1;
            double isign = (i & 1)? -1.: 1.;
            double jsign = (j & 1)? -1.: 1.;

            values[{level, i, j}] = prediction_2d<order, index_t>(level-1, ig, jg);

            auto interpx = interp_coeffs<2*order+1>(isign);
            auto interpy = interp_coeffs<2*order+1>(jsign);

            for(std::size_t ci=0; ci<interpx.size(); ++ci)
            {
                for(std::size_t cj=0; cj<interpy.size(); ++cj)
                {
                    if (ci != order || cj != order)
                    {
                        values[{level, i, j}] += interpx[ci]*interpy[cj]*prediction_2d<order, index_t>(level-1, ig + ci - order, jg + cj - order);
                    }
                }
            }
            return values[{level, i, j}];
        }
        else
        {
            return iter->second;
        }
    }

    template <std::size_t order=1, class index_t=samurai::default_config::value_t>
    auto prediction_3d(std::size_t level, index_t i, index_t j, index_t k, bool reset=false)
    {
        static std::map<std::tuple<std::size_t, index_t, index_t, index_t>, prediction_map<3, index_t>> values;

        if (reset)
        {
            values.clear();
        }

        if (level == 0)
        {
            return prediction_map<3, index_t>{{i, j, k}};
        }

        auto iter = values.find({level, i, j, k});

        if (iter == values.end())
        {
            int ig = i >> 1;
            int jg = j >> 1;
            int kg = k >> 1;
            double isign = (i & 1)? -1.: 1.;
            double jsign = (j & 1)? -1.: 1.;
            double ksign = (k & 1)? -1.: 1.;

            values[{level, i, j, k}] = prediction_3d<order, index_t>(level-1, ig, jg, kg);

            auto interpx = interp_coeffs<2*order+1>(isign);
            auto interpy = interp_coeffs<2*order+1>(jsign);
            auto interpz = interp_coeffs<2*order+1>(ksign);

            for(std::size_t ci=0; ci<interpx.size(); ++ci)
            {
                for(std::size_t cj=0; cj<interpy.size(); ++cj)
                {
                    for(std::size_t ck=0; ck<interpz.size(); ++ck)
                    {
                       if (ci != order || cj != order || ck != order)
                        {
                            values[{level, i, j, k}] += interpx[ci]*interpy[cj]*interpz[ck]*prediction_3d<order, index_t>(level-1, ig + ci - order, jg + cj - order, kg + ck - order);
                        }
                    }
                }
            }
            return values[{level, i, j, k}];
        }
        else
        {
            return iter->second;
        }
    }

    template<class TInterval>
    class reconstruction_op_: public field_operator_base<TInterval>
    {
    public:
        INIT_OPERATOR(reconstruction_op_)

        template<class T1, class T2>
        inline void operator()(Dim<1>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        {
            using index_t = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_order = T2::mesh_t::config::prediction_order;

            int delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i) = src(level, i);
            }
            else
            {
                std::size_t nb_cells = 1<<delta_l;
                for (int ii = 0; ii < nb_cells; ++ii)
                {
                    auto pred = prediction<prediction_order, index_t>(delta_l, ii);
                    for(auto& kv: pred.coeff)
                    {
                        auto i_f = (i<<delta_l) + ii;
                        i_f.step = nb_cells;
                        dest(reconstruct_level, i_f) += kv.second*src(level, i + kv.first[0]);
                    }
                }
            }
        }

        template<class T1, class T2>
        inline void operator()(Dim<2>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        {
            using index_t = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_order = T2::mesh_t::config::prediction_order;

            int delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i, j) = src(level, i, j);
            }
            else
            {
                std::size_t nb_cells = 1<<delta_l;
                for (int jj = 0; jj < nb_cells; ++jj)
                {
                    auto j_f = (j<<delta_l) + jj;
                    for (int ii = 0; ii < nb_cells; ++ii)
                    {
                        auto pred = prediction_2d<prediction_order, index_t>(delta_l, ii, jj);
                        auto i_f = (i<<delta_l) + ii;
                        i_f.step = nb_cells;

                        for(auto& kv: pred.coeff)
                        {
                            dest(reconstruct_level, i_f, j_f) += kv.second*src(level, i + kv.first[0], j + kv.first[1]);
                        }
                    }
                }
            }
        }

        template<class T1, class T2>
        inline void operator()(Dim<3>, std::size_t& reconstruct_level, T1& dest, const T2& src) const
        {
            using index_t = typename T2::interval_t::value_t;
            constexpr std::size_t prediction_order = T2::mesh_t::config::prediction_order;

            int delta_l = reconstruct_level - level;
            if (delta_l == 0)
            {
                dest(level, i, j, k) = src(level, i, j, k);
            }
            else
            {
                std::size_t nb_cells = 1<<delta_l;
                for (int kk = 0; kk < nb_cells; ++kk)
                {
                    auto k_f = (k<<delta_l) + kk;
                    for (int jj = 0; jj < nb_cells; ++jj)
                    {
                        auto j_f = (j<<delta_l) + jj;
                        for (int ii = 0; ii < nb_cells; ++ii)
                        {
                            auto pred = prediction_3d<prediction_order, index_t>(delta_l, ii, jj, kk);
                            auto i_f = (i<<delta_l) + ii;
                            i_f.step = nb_cells;

                            for(auto& kv: pred.coeff)
                            {
                                dest(reconstruct_level, i_f, j_f, k_f) += kv.second*src(level, i + kv.first[0], j + kv.first[1], k + kv.first[2]);
                            }
                        }
                    }
                }
            }
        }

    };

    template<class T1, class T2>
    inline auto make_reconstruction(std::size_t& reconstruct_level, T1&& reconstruct_field, T2&& field)
    {
        return make_field_operator_function<reconstruction_op_>(reconstruct_level, std::forward<T1>(reconstruct_field), std::forward<T2>(field));
    }

    template <class Field>
    auto reconstruction(const Field& field)
    {
        using mesh_t = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        using ca_type = typename mesh_t::ca_type;
        constexpr std::size_t prediction_order = mesh_t::config::prediction_order;
        using index_t = typename Field::interval_t::value_t;
        auto mesh = field.mesh();
        ca_type reconstruct_mesh;
        std::size_t reconstruct_level = mesh.domain().level();
        reconstruct_mesh[reconstruct_level] = mesh.domain();
        reconstruct_mesh.update_index();

        auto m = holder(reconstruct_mesh);
        auto reconstruct_field = make_field<typename Field::value_type, Field::size>(field.name(), m);
        reconstruct_field.fill(0.);

        std::size_t min_level = mesh[mesh_id_t::cells].min_level();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level = min_level; level <= max_level; ++level)
        {
            auto set = samurai::intersection(mesh[mesh_id_t::cells][level], reconstruct_mesh[reconstruct_level])
                      .on(level);
            set.apply_op(make_reconstruction(reconstruct_level, reconstruct_field, field));
        }
        return reconstruct_field;
    }

    template <class Field, class Interval>
    auto flux(const Field& f, std::size_t level, const Interval& i, const std::array<int, 1>& stencil, std::size_t dest_level)
    {
        using value_t = typename Field::value_type;
        using mesh_t = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        using ca_type = typename mesh_t::ca_type;
        constexpr std::size_t prediction_order = mesh_t::config::prediction_order;
        using index_t = typename Field::interval_t::value_t;

        if (level == dest_level)
        {
            return f(level, i + stencil[0]);
        }
        else
        {
            std::size_t delta_l = dest_level - level;
            index_t i_pred = 0;

            if (stencil[0] > 0)
            {
                i_pred = stencil[0] - 1;
            }
            else if (stencil[0] < 0)
            {
                i_pred = (1 << delta_l) + stencil[0] - 1;
            }

            auto pred = prediction<prediction_order, index_t>(delta_l, i_pred);

            xt::xtensor<value_t, 1> result = xt::zeros<value_t>(i.size());

            for(auto& kv: pred.coeff)
            {
                result += kv.second*f(level, i + stencil[0] + kv.first[0]);
            }
            return result;
        }
    }

    namespace detail
    {
        template <std::size_t prediction_order, class Field, class Interval>
        auto portion_impl(const Field& f, std::size_t element, std::size_t level, const Interval& i, std::size_t delta_l, std::size_t index)
        {
            using index_t = typename Field::interval_t::value_t;

            auto pred = prediction<prediction_order, index_t>(delta_l, index);

            auto result = xt::zeros_like(f(element, level, i));

            for(auto& kv: pred.coeff)
            {
                result += kv.second*f(element, level, i + kv.first[0]);
            }
            return result;
        }

        template <std::size_t prediction_order, class Field, class Interval>
        auto portion_impl(const Field& f, std::size_t level, const Interval& i, std::size_t delta_l, std::size_t index)
        {
            using index_t = typename Field::interval_t::value_t;

            auto pred = prediction<prediction_order, index_t>(delta_l, index);

            auto result = xt::zeros_like(f(level, i));

            for(auto& kv: pred.coeff)
            {
                result += kv.second*f(level, i + kv.first[0]);
            }
            return result;
        }
    }

    template <class Field, class Interval>
    auto portion(const Field& f, std::size_t element, std::size_t level, const Interval& i, std::size_t delta_l, std::size_t index)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, element, level, i, delta_l, index);
    }

    template <std::size_t prediction_order, class Field, class Interval>
    auto portion(const Field& f, std::size_t element, std::size_t level, const Interval& i, std::size_t delta_l, std::size_t index)
    {
        return detail::portion_impl<prediction_order>(f, element, level, i, delta_l, index);
    }

    template <class Field, class Interval>
    auto portion(const Field& f, std::size_t level, const Interval& i, std::size_t delta_l, std::size_t index)
    {
        return detail::portion_impl<Field::mesh_t::config::prediction_order>(f, level, i, delta_l, index);
    }

    template <std::size_t prediction_order, class Field, class Interval>
    auto portion(const Field& f, std::size_t level, const Interval& i, std::size_t delta_l, std::size_t index)
    {
        return detail::portion_impl<prediction_order>(f, level, i, delta_l, index);
    }
}