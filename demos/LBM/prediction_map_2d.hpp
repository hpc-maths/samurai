// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once
#include <map>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsemantic.hpp>

template <class index_t>
class prediction_map
{
  public:

    prediction_map() = default;

    prediction_map(index_t i, index_t j)
    {
        coeff[{i, j}] = 1.;
    }

    double& operator()(index_t i, index_t j)
    {
        auto it = coeff.find({i, j});
        if (it == coeff.end())
        {
            coeff[{i, j}] = 0.;
        }
        return coeff[{i, j}];
    }

    prediction_map<index_t>& operator+=(const prediction_map<index_t> p)
    {
        for (auto& c : p.coeff)
        {
            auto& cc = (*this)(std::get<0>(c.first), std::get<1>(c.first));
            cc += c.second;
        }
        return *this;
    }

    prediction_map<index_t>& operator-=(const prediction_map<index_t> p)
    {
        for (auto& c : p.coeff)
        {
            auto& cc = (*this)(std::get<0>(c.first), std::get<1>(c.first));
            cc -= c.second;
        }
        return *this;
    }

    prediction_map<index_t>& operator*=(const double d)
    {
        std::size_t index = 0;
        for (auto& c : coeff)
        {
            c.second *= d;
        }
        return *this;
    }

    prediction_map<index_t>& operator+=(const double d)
    {
        for (auto& c : coeff)
        {
            c.second += d;
        }
        return *this;
    }

    void to_stream(std::ostream& out) const
    {
        for (auto& c : coeff)
        {
            std::cout << "( " << std::get<0>(c.first) << ", " << std::get<1>(c.first) << " ): " << c.second << "\n";
        }
    }

    // private:
    std::map<std::tuple<index_t, index_t>, double> coeff;
};

template <class index_t>
auto operator+(const prediction_map<index_t>& p1, const prediction_map<index_t>& p2)
{
    prediction_map<index_t> that{p1};
    that += p2;
    return that;
}

template <class index_t>
auto operator+(const double d, const prediction_map<index_t>& p)
{
    prediction_map<index_t> that{p};
    that += d;
    return that;
}

template <class index_t>
auto operator-(const prediction_map<index_t>& p1, const prediction_map<index_t>& p2)
{
    prediction_map<index_t> that{p1};
    that -= p2;
    return that;
}

template <class index_t>
auto operator*(const double d, const prediction_map<index_t>& p)
{
    prediction_map<index_t> that{p};
    that *= d;
    return that;
}

template <class index_t>
inline std::ostream& operator<<(std::ostream& out, const prediction_map<index_t>& pred)
{
    pred.to_stream(out);
    return out;
}

template <class index_t>
auto prediction(std::size_t level, const index_t& i, const index_t j, bool reset = false)
{
    static std::map<std::tuple<std::size_t, index_t, index_t>, prediction_map<index_t>> values;

    if (reset)
    {
        values.clear();
    }

    if (level == 0)
    {
        return prediction_map<index_t>{i, j};
    }

    auto iter = values.find({level, i, j});

    if (iter == values.end())
    {
        auto ig     = i >> 1;
        auto jg     = j >> 1;
        double d_x  = (i & 1) ? -1. / 8 : 1. / 8;
        double d_xy = ((i + j) & 1) ? -1. / 64 : 1. / 64;
        double d_y  = (j & 1) ? -1. / 8 : 1. / 8;

        return values[{level, i, j}] = prediction(level - 1, ig, jg)
                                     - d_x * (prediction(level - 1, ig + 1, jg) - prediction(level - 1, ig - 1, jg))
                                     - d_y * (prediction(level - 1, ig, jg + 1) - prediction(level - 1, ig, jg - 1))
                                     - d_xy
                                           * (prediction(level - 1, ig + 1, jg + 1) - prediction(level - 1, ig + 1, jg - 1)
                                              - prediction(level - 1, ig - 1, jg + 1) + prediction(level - 1, ig - 1, jg + 1));
    }
    else
    {
        return iter->second;
    }
}
