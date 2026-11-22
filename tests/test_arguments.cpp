#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/arguments.hpp>
#include <samurai/samurai.hpp>

namespace samurai_test
{
    // Defined in arguments_other_translation_unit.cpp.
    std::size_t max_level_in_other_translation_unit();
    int finer_level_flux_in_other_translation_unit();
    const CLI::App* app_in_other_translation_unit();
}

namespace samurai
{
    namespace
    {
        // Parses the command line made of args, the program name excluded.
        void parse(App& app, std::vector<std::string> args)
        {
            args.insert(args.begin(), "program");
            std::vector<char*> argv;
            for (auto& arg : args)
            {
                argv.push_back(arg.data());
            }
            app.parse(static_cast<int>(argv.size()), argv.data());
        }
    }

    // A point option reads one value per coordinate, as for the corners of a box:
    // --max-corner 4 5 gives the point (4, 5).
    TEST(arguments, point_option_reads_one_value_per_coordinate)
    {
        App app;
        xt::xtensor_fixed<double, xt::xshape<1>> point_1d = {0.};
        xt::xtensor_fixed<double, xt::xshape<2>> point_2d = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<3>> point_3d = {0., 0., 0.};
        xt::xtensor_fixed<int, xt::xshape<2>> integers    = {0, 0};
        app.add_option("--point-1d", point_1d);
        app.add_option("--point-2d", point_2d);
        app.add_option("--point-3d", point_3d);
        app.add_option("--integers", integers);

        parse(app, {"--point-1d", "1.5", "--point-2d", "4", "5", "--point-3d", "-1", "2", "3.5", "--integers", "7", "8"});

        EXPECT_EQ(point_1d, (xt::xtensor_fixed<double, xt::xshape<1>>{1.5}));
        EXPECT_EQ(point_2d, (xt::xtensor_fixed<double, xt::xshape<2>>{4., 5.}));
        EXPECT_EQ(point_3d, (xt::xtensor_fixed<double, xt::xshape<3>>{-1., 2., 3.5}));
        EXPECT_EQ(integers, (xt::xtensor_fixed<int, xt::xshape<2>>{7, 8}));
    }

    TEST(arguments, point_option_keeps_its_default_value)
    {
        App app;
        xt::xtensor_fixed<double, xt::xshape<2>> point = {0.5, 1.};
        auto* option                                   = app.add_option("--point", point)->capture_default_str();

        parse(app, {});

        EXPECT_EQ(point, (xt::xtensor_fixed<double, xt::xshape<2>>{0.5, 1.}));
        EXPECT_EQ(option->get_default_str(), "[0.5,1]");
    }

    TEST(arguments, point_option_needs_one_value_per_coordinate)
    {
        xt::xtensor_fixed<double, xt::xshape<2>> point = {0., 0.};

        App too_few;
        too_few.add_option("--point", point);
        EXPECT_THROW(parse(too_few, {"--point", "4"}), CLI::ArgumentMismatch);

        App too_many;
        too_many.add_option("--point", point);
        EXPECT_THROW(parse(too_many, {"--point", "4", "5", "6"}), CLI::ParseError);
    }

    // The command-line options are one set of variables for the whole program: what
    // the translation unit that parses the command line reads must be seen by the others.
    TEST(arguments, command_line_options_are_shared_between_translation_units)
    {
        const auto max_level        = args::max_level;
        const auto finer_level_flux = args::finer_level_flux;

        App cli;
        std::vector<std::string> options = {"program", "--max-level", "7", "--finer-level-flux", "1"};
        std::vector<char*> argv;
        for (auto& option : options)
        {
            argv.push_back(option.data());
        }
        int argc    = static_cast<int>(argv.size());
        char** data = argv.data();
        read_samurai_arguments(cli, argc, data);

        EXPECT_EQ(args::max_level, 7U);
        EXPECT_EQ(samurai_test::max_level_in_other_translation_unit(), 7U);
        EXPECT_EQ(samurai_test::finer_level_flux_in_other_translation_unit(), 1);
        EXPECT_EQ(samurai_test::app_in_other_translation_unit(), &app);

        args::max_level        = max_level;
        args::finer_level_flux = finer_level_flux;
    }
}
