cmake_minimum_required(VERSION 3.16)
project(MySamuraiProject)

find_package(samurai REQUIRED)

add_executable(my_executable main.cpp)
target_link_libraries(my_executable PRIVATE samurai::samurai)