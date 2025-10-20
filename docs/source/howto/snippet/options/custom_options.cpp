#include <iostream>

#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    auto& app = samurai::initialize("Add my options", argc, argv);

    int my_option = 42;
    bool my_flag  = false;

    app.add_option("--my-option", my_option, "An example of custom option")->capture_default_str()->group("Custom");
    app.add_flag("--my-flag", my_flag, "An example of custom flag")->group("Custom");

    SAMURAI_PARSE(argc, argv);

    std::cout << "my-option = " << my_option << std::endl;
    std::cout << "my-flag = " << std::boolalpha << my_flag << std::endl;

    samurai::finalize();
    return 0;
}
