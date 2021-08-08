set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(SFML 2 REQUIRED COMPONENTS network audio graphics window system)

add_executable(
    catdog          main.cpp
    nmath.h         nmath.cpp
    network.h       network.cpp
    graphics.h      graphics.cpp
    utils.h         utils.cpp
)

target_link_libraries(
    catdog
    PUBLIC
        Eigen3::Eigen
        ${SFML_LIBRARIES} ${SFML_DEPENDENCIES}
    PRIVATE
        Threads::Threads
)
