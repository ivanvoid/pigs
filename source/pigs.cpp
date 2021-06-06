#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iostream>
#include <vector>
#include <cmath>        /* abs */
#include <math.h>       /* sqrt, pow, exp */

double rgb_pos_kernel(
    uint8_t R1, uint8_t G1, uint8_t B1,
    uint8_t R2, uint8_t G2, uint8_t B2,
    float x1, float y1, float x2, float y2, 
    double beta){
    // x and y must be devided by 2*N (where N is length of vector) 
    double position = std::abs((x1 - x2) + (y1 - y2));
    
    double l2 = sqrt(
        pow((R1 - R2 + position), 2) +
        pow((G1 - G2 + position), 2) +
        pow((B1 - B2 + position), 2)
        );
    
    double result = exp(-1 * beta * l2);
//     return result;
    return (result < 1e-7) ? 0 : result;
}

PYBIND11_MODULE(_pigs, m) {
    m.doc() = "Lib for faster computation of pigs"; // optional module docstring

    m.def("rgb_pos_kernel", py::vectorize(rgb_pos_kernel), "Kernel computation");
    
}