#include <iostream>

#include "integrator.h"
#include "util.h"

//  cmake --build build && cd build && ./gdopt_experimental && cd ..

int main() {
    Integrator spectral = Integrator();
    std::cout << sizeof(gNumber) << " bytes\n";
    return 0;
}