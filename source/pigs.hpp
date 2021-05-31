#pragma once
#include <iostream>
#include <string>


class PIGS{
public:
    // Constructors
    PIGS(); //string filepath
    PIGS(PIGS const & other);
    PIGS(PIGS && other);
    ~PIGS() = default;
    
    // fuctions
    void compute_gram();
private:
    // Vars
};

