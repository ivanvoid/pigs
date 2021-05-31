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
    if(result < 1e-9)
        return 0;
    return result;
}


//     rgb1,rgb2,pos1,pos2,N,beta):
//     pos = abs((pos1[0] - pos2[0]) + (pos1[1] - pos2[1])) / (2*N) 
//     l2 = np.sqrt((rgb1[0] - rgb2[0] + pos)**2 + 
//                  (rgb1[1] - rgb2[1] + pos)**2 + 
//                  (rgb1[2] - rgb2[2] + pos)**2)
//     result = np.exp(-beta * l2)

//     return result

int compute_gram(py::array_t<double> input, py::array_t<int> locations){
    // Get arrays info
    // input.shape = (N,3)
    py::buffer_info buffer_input    = input.request();
    py::buffer_info locations_input = locations.request();
//     int N = buffer_input.shape[0];

    // Get pointers to arrays
    double *ptr_input  = static_cast<double *>(buffer_input.ptr);
//     int *ptr_locations = static_cast<int *>(locations_input.ptr);

    
    // Iterate through array
    int k = 0;
    for(int i=0; i < buffer_input.shape[0]; i++){
//         for(int i=0; j < buffer_input.size; j++){
        std::cout << i << ' ' 
            << ptr_input[i] << '\n';
//             << ptr_input[ii] << ' '
//             << ptr_input[iii] << ' '
//             << "\n";
        k++;
//         }
    }
    std::cout << k;
    
    return 1;
}

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_pigs, m) {
    m.doc() = "Lib for faster computation of pigs"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    
//     m.def("compute_gram", &compute_gram, "Gram matrix computation");
    
    m.def("rgb_pos_kernel", py::vectorize(rgb_pos_kernel), "Kernel computation");
    
}