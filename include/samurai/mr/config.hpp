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

        void init_options(CLI::App& app_)
        {
            app_.add_option("--mr-eps", m_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
                ->capture_default_str()
                ->group("Multiresolution");
            app_.add_option("--mr-reg", m_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
                ->capture_default_str()
                ->group("Multiresolution");
            app_.add_flag("--mr-rel-detail", m_rel_detail, "Use relative detail instead of absolute detail")->group("Multiresolution");
        }

      private:

        double m_epsilon    = 1e-4;
        double m_regularity = 1.;
        bool m_rel_detail   = false;
    };
}
