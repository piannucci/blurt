#ifndef KALMAN_H
#define KALMAN_H
#include "blurt.h"

class KalmanPilotTracker {
private:
    std::vector<float> P, x, Q, R;
public:
    KalmanPilotTracker(float uncertainty, float var_n);
    void update(complex pilot, complex &u);
};

#endif
