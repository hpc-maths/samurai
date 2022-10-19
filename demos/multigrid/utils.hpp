#pragma once
#include <string>
#include <iostream>

enum DirichletEnforcement : int
{
    Penalization,
    Elimination,
    OnesOnDiagonal
};

void error(std::string msg)
{
    std::string beginRed = "\033[1;31m";
    std::string endColor = "\033[0m";
    std::cout << beginRed << "Error: " << msg << endColor << std::endl;
}
void fatal_error(std::string msg)
{
    std::string beginRed = "\033[1;31m";
    std::string endColor = "\033[0m";
    std::cout << beginRed << "Error: " << msg << endColor << std::endl;
    std::cout << "------------------------- FAILURE -------------------------" << std::endl;
    assert(false);
    exit(EXIT_FAILURE);
}
void warning(std::string msg)
{
    std::string beginYellow = "\033[1;33m";
    std::string endColor = "\033[0m";
    std::cout << beginYellow << "Warning: " << msg << endColor << std::endl;
}