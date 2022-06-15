// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once
#include <map>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsemantic.hpp>

class prediction_map
{
    using index_t = int;
    public:
        prediction_map() = default;

        prediction_map(index_t i)
        {
            coeff[i] = 1.;
        }

        double& operator()(index_t i)
        {
            auto it = coeff.find(i);
            if (it == coeff.end())
            {
                coeff[i] = 0.;
            }
            return coeff[i];
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
                std::cout << "( " << c.first << ", ): " << c.second << "\n";
            }
        }
    // private:
        std::map<int, double> coeff;
};

auto operator+(const prediction_map &p1, const prediction_map &p2)
{
    prediction_map that{p1};
    that += p2;
    return that;
}

auto operator+(const double d, const prediction_map &p)
{
    prediction_map that{p};
    that += d;
    return that;
}

auto operator-(const prediction_map &p1, const prediction_map &p2)
{
    prediction_map that{p1};
    that -= p2;
    return that;
}

auto operator*(const double d, const prediction_map &p)
{
    prediction_map that{p};
    that *= d;
    return that;
}

inline std::ostream &operator<<(std::ostream &out, const prediction_map &pred)
{
    pred.to_stream(out);
    return out;
}

// template<class index_t>
// auto prediction(std::size_t level, const index_t &i, bool reset=false)
// {
//     static std::map<std::tuple<std::size_t, index_t>, prediction_map> values;

//     if (reset)
//     {
//         values.clear();
//     }

//     if (level == 0)
//     {
//         return prediction_map{i};
//     }

//     auto iter = values.find({level, i});

//     if (iter == values.end())
//     {
//         auto ig = i >> 1;
//         double d_x = (i & 1)? -1./8: 1./8;

//         return values[{level, i}] = prediction(level-1, ig) - d_x * (prediction(level-1, ig+1)
//                                                                    - prediction(level-1, ig-1));
//     }
//     else
//     {
//         return iter->second;
//     }
// }

auto prediction(std::size_t level, const int &i, bool reset=false)
{
    static std::map<std::tuple<std::size_t, int>, prediction_map> values;

    if (reset)
    {
        values.clear();
    }

    if (level == 0)
    {
        return prediction_map{i};
    }

    auto iter = values.find({level, i});

    if (iter == values.end())
    {
        int ig = i >> 1;
        double sign = (i & 1)? -1.: 1.;

        std::cout << fmt::format("construct ({}, {}) with ig = {}: ", level, i, ig) << std::endl;
        values[{level, i}] = prediction(level-1, ig);
        int s = 1;
        for (auto c: samurai::coeffs<2>())
        {
            values[{level, i}] += c*sign * (prediction(level-1, ig + s)
                                          - prediction(level-1, ig - s));
            s++;
        }
        for (auto& v: values[{level, i}].coeff)
        {
            std::cout << fmt::format("[{}, {}] ", v.first, v.second);
        }
        std::cout << std::endl;
        return values[{level, i}];
        // double d_x = (i & 1)? -1./8: 1./8;
        // return values[{level, i}] = prediction(level-1, ig) - d_x * (prediction(level-1, ig+1)
        //                                                            - prediction(level-1, ig-1));
    }
    else
    {
        return iter->second;
    }
}