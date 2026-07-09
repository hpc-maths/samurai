#include <algorithm>
#include <regex>
#include <sstream>

#include <gtest/gtest.h>

#include <samurai/version.hpp>

namespace samurai
{
    TEST(version, format)
    {
        // version() must always be a "major.minor.patch" triplet.
        EXPECT_TRUE(std::regex_match(version(), std::regex(R"(\d+\.\d+\.\d+)")));
    }

    TEST(version, dependencies)
    {
        const auto deps = dependencies_info();

        // fmt is a mandatory dependency, so it is always reported...
        const auto has = [&](std::string_view name)
        {
            return std::any_of(deps.begin(),
                               deps.end(),
                               [&](const dependency_info& d)
                               {
                                   return d.name == name;
                               });
        };
        EXPECT_TRUE(has("fmt"));

        // ...and every reported version is a non-empty string.
        for (const auto& d : deps)
        {
            EXPECT_FALSE(d.version.empty()) << "empty version for " << d.name;
        }
    }

    TEST(version, build)
    {
        const auto build = build_info();

        const auto value_of = [&](std::string_view name) -> std::string
        {
            const auto it = std::find_if(build.begin(),
                                         build.end(),
                                         [&](const dependency_info& b)
                                         {
                                             return b.name == name;
                                         });
            return it == build.end() ? std::string{} : it->version;
        };

        // The MPI feature must reflect the compile-time configuration.
#ifdef SAMURAI_WITH_MPI
        EXPECT_EQ(value_of("MPI"), "ON");
#else
        EXPECT_EQ(value_of("MPI"), "OFF");
#endif
    }

    TEST(version, print)
    {
        std::ostringstream os;
        print_info(os);
        const auto out = os.str();

        EXPECT_NE(out.find("samurai"), std::string::npos);
        EXPECT_NE(out.find(version()), std::string::npos);
        EXPECT_NE(out.find("Dependencies"), std::string::npos);
        EXPECT_NE(out.find("Build configuration"), std::string::npos);
    }
}
