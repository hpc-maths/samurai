find_package(CGAL)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

add_executable(apple-from_obj main.cpp)
target_link_libraries(apple-from_obj samurai CGAL::CGAL Qt5::Core Qt5::Widgets)
