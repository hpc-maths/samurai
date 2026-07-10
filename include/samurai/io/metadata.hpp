// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <string>

#include <highfive/H5Easy.hpp>
#include <pugixml.hpp>

namespace samurai
{
    /**
     * Facade used to inject user-defined metadata into the HDF5 file and, when
     * available, the XDMF document produced by samurai::save / samurai::dump.
     *
     * The XDMF node is optional: when it is null (e.g. a checkpoint written with
     * samurai::dump, which has no XDMF), the XDMF-related operations become
     * no-ops and only the HDF5 file is written. A single class therefore serves
     * both the visualization path (save, with XDMF) and the restart path (dump,
     * HDF5 only).
     */
    class MetadataWriter
    {
      public:

        /// Constructor for the HDF5-only case (no XDMF document).
        explicit MetadataWriter(HighFive::File& file)
            : m_file(file)
        {
        }

        /// Constructor for the visualization case (HDF5 + XDMF <Domain> node).
        MetadataWriter(HighFive::File& file, const pugi::xml_node& domain)
            : m_file(file)
            , m_domain(domain)
        {
        }

        /// Write (or overwrite) an arbitrary attribute on the root group of the
        /// HDF5 file. Works for scalars, std::string and std::vector<T>.
        template <class T>
        MetadataWriter& attribute(const std::string& name, const T& value)
        {
            if (m_file.hasAttribute(name))
            {
                m_file.getAttribute(name).write(value);
            }
            else
            {
                m_file.createAttribute(name, value);
            }
            return *this;
        }

        /// Store the simulation time as the HDF5 attribute "time" and, when an
        /// XDMF document is attached, add a <Time> node to every top-level grid
        /// (so that ParaView/VisIt expose a time cursor).
        MetadataWriter& time(double t)
        {
            attribute("time", t);
            for (pugi::xml_node grid : m_domain.children("Grid"))
            {
                auto time_node                      = grid.prepend_child("Time");
                time_node.append_attribute("Value") = t;
            }
            return *this;
        }

        /// Add a human-readable <Information> node under the XDMF <Domain>.
        /// No-op when no XDMF document is attached.
        MetadataWriter& information(const std::string& name, const std::string& value)
        {
            auto info                      = m_domain.append_child("Information");
            info.append_attribute("Name")  = name.c_str();
            info.append_attribute("Value") = value.c_str();
            return *this;
        }

      private:

        HighFive::File& m_file; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        pugi::xml_node m_domain{};
    };

    /**
     * Facade used to read user-defined metadata back from an HDF5 file produced
     * by samurai::save / samurai::dump.
     */
    class MetadataReader
    {
      public:

        explicit MetadataReader(const HighFive::File& file)
            : m_file(file)
        {
        }

        /// Whether an attribute with the given name exists on the root group.
        bool has(const std::string& name) const
        {
            return m_file.hasAttribute(name);
        }

        /// Read an attribute from the root group of the HDF5 file.
        template <class T>
        T attribute(const std::string& name) const
        {
            return m_file.getAttribute(name).read<T>();
        }

        /// Convenience accessor for the simulation time.
        double time() const
        {
            return attribute<double>("time");
        }

      private:

        const HighFive::File& m_file; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    };
}
