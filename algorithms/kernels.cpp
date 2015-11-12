#ifndef QUACODE_USE_CUDA
#include <iostream>

#include <algorithms/kernels.hh>


void foo() {
    std::cerr << "Foo on CPU." << std::endl;
}

bool evaluateCstrs(Constraint ** cstrs, size_t nbCstrs, const int * candidat) {
    for (size_t i = 0; i < nbCstrs; ++i) {
        if (!cstrs[i]->evaluate(candidat)) return false;
    }

    return true;
}

#endif
