// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <CLI/CLI.hpp>

namespace samurai
{
    class mra_config
    {
      public:

        auto& epsilon(double eps)
        {
            m_epsilon = eps;
            return *this;
        }

        auto& epsilon() const
        {
            return m_epsilon;
        }

        auto& regularity(double reg)
        {
            m_regularity = reg;
            return *this;
        }

        auto& regularity() const
        {
            return m_regularity;
        }

        auto& relative_detail(bool rel)
        {
            m_rel_detail = rel;
            return *this;
        }

        auto& relative_detail() const
        {
            return m_rel_detail;
        }

        void parse_args()
        {
            if (args::epsilon != std::numeric_limits<double>::infinity())
            {
                m_epsilon = args::epsilon;
            }
            if (args::regularity != std::numeric_limits<double>::infinity())
            {
                m_regularity = args::regularity;
            }
            if (args::rel_detail)
            {
                m_rel_detail = true;
            }
        }

      private:

        double m_epsilon    = 1e-4;
        double m_regularity = 1.;
        bool m_rel_detail   = false;
    };
}
