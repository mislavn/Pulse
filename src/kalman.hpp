#pragma once

#include "linalg.h"

// M x N matrix
template<class T, unsigned int M, unsigned int N>

struct KalmanFilter {
public:
    KalmanFilter() {
        //TODO init xH
    }

    void update(linalg::vec<T, M> x0) {
    }

private:
    linalg::mat<T, M, N> A;
    linalg::mat<T, M, N> B; // output matrix
    linalg::mat<T, M, N> P; // estimation error
    linalg::mat<T, M, N> Q; // process noise
    linalg::mat<T, M, N> R; // measurement noise
    linalg::mat<T, M, N> K;
    linalg::mat<T, M, N> P0;

    linalg::vec<T, M> I;
    linalg::vec<T, M> xH;
    linalg::vec<T, M> xH_next;
};
