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
        xH_next = A * xH;
        P = A * P * linalg::transpose(A) + Q;
        K = P * linalg::transpose(B) * linalg::inverse(B * P * linalg::transpose(B) + R);
        xH_next += K * (x0 - B * xH_next);
        P = (I - K * B) * P;
        xH = xH_next;
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
