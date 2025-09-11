#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "MDSim.h"

static const double SIGMA = 0.025;
static const double SIGMA_6 = std::pow(SIGMA, 6.0);

vec f(vec r) {
    double r2 = r * r, r4 = r2 * r2, r8 = r4 * r4;
    return 24.0 * SIGMA_6 * ((2 * SIGMA_6) / (r8 * r4 * r2) - 1.0 / r8) * r;
}
double pot(vec r) {
    double r2 = r * r, r4 = r2 * r2;
    return 4.0 * SIGMA_6 * (SIGMA_6 / (r4 * r4 * r4) - 1.0 / (r4 * r2));
}

static double run_once(int N, int steps) {
    double dt = 1e-3, nue = 1.0, T0 = 1.0;
    MDSim sim(2, dt, f, pot, /*r_inter*/0.15, /*r_verlet*/0.22, /*histRes*/10);

    for (int i = 0; i < N; i++) sim.particles->addParticle(new MDParticle(2));
    sim.getThermostat()->setT(T0);
    sim.getThermostat()->setNue(nue);
    double arr1[2] = { 1.0,1.0 };
    sim.initSim(true, vec(arr1, 2), 0, 0, 400, 0.15);

    // warm-up (few steps)
    for (int s = 0; s < 5; s++) {
        sim.benchRefresh(false, false);
        sim.benchForce();
        sim.velocityVerletStep(false);
    }

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    double pot_total = 0.0;
    for (int s = 0; s < steps; s++) {
        sim.benchRefresh(false, false);     // builds list
        pot_total += sim.benchForce();     // CPU in Release-CPU, CUDA in Release-GPU
        sim.velocityVerletStep(false);     // integrate using computed forces
    }
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

#ifdef USE_CUDA
    std::cout << "[MODE] GPU ";
#else
    std::cout << "[MODE] CPU ";
#endif
    std::cout << "N=" << N << " steps=" << steps << " time_ms=" << ms << " pot=" << pot_total << "\n";
    return ms;
}

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::atoi(argv[1]) : 1024);
    int steps = (argc > 2 ? std::atoi(argv[2]) : 200);
    run_once(N, steps);
    return 0;
}
