#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "MDSim.h"
#include <iomanip>

// ---- Plimpton (1995) LJ reduced units ----
static const double SIGMA = 1.0;
static const double EPS = 1.0;
static const double RHO = 0.8442;         // density
static const double T0 = 0.72;           // target temperature
static const double DT = 0.00462;        // time step
static const double RC = 2.5 * SIGMA;    // force cutoff
static const double RS = 2.8 * SIGMA;    // neighbor-list radius (skin)
static const int    REBUILD_EVERY = 20;     // neighbor rebuild cadence

// LJ force/potential in reduced units (m=1,kB=1)
vec f(vec r) {
    double r2 = r * r; if (r2 == 0.0) return vec(0, r.get_dim());
    double inv2 = 1.0 / r2;
    double sr2 = (SIGMA * SIGMA) * inv2;        // (σ/r)^2
    double sr6 = sr2 * sr2 * sr2;                 // (σ/r)^6
    double sr12 = sr6 * sr6;                     // (σ/r)^12
    double fsc = 24.0 * EPS * (2.0 * sr12 - sr6) * inv2; // (|F|/r) factor
    return fsc * r;                             // vector force
}
double pot(vec r) {
    double r2 = r * r; if (r2 == 0.0) return 0.0;
    double inv2 = 1.0 / r2;
    double sr2 = (SIGMA * SIGMA) * inv2;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;
    return 4.0 * EPS * (sr12 - sr6);
}
// Safer versions without r14:
vec f_safe(vec r) {
    double r2 = r * r; if (r2 == 0) return vec(0, r.get_dim());
    double inv2 = 1.0 / r2; double inv6 = inv2 * inv2 * inv2; double inv12 = inv6 * inv6;
    double fsc = 24.0 * (2.0 * inv12 - inv6) * inv2; // (|F|/r) factor
    return fsc * r;
}
double pot_safe(vec r) {
    double r2 = r * r; if (r2 == 0) return 0.0;
    double inv2 = 1.0 / r2; double inv6 = inv2 * inv2 * inv2; double inv12 = inv6 * inv6;
    return 4.0 * (inv12 - inv6);
}

// ---- FCC initializer ----
static double gL = 0.0; static int gNc = 0; // box length, cells per side
static vec r0_fcc(int idx) {
    static const double off[4][3] = { {0,0,0},{0.5,0.5,0},{0.5,0,0.5},{0,0.5,0.5} };
    int cell = idx / 4, b = idx % 4;
    int ix = cell % gNc;
    int iy = (cell / gNc) % gNc;
    int iz = cell / (gNc * gNc);
    double a = gL / gNc; // lattice constant
    vec r(0, 3);
    r[1] = (ix + off[b][0]) * a;
    r[2] = (iy + off[b][1]) * a;
    r[3] = (iz + off[b][2]) * a;
    return r;
}

// Maxwell–Boltzmann velocities (zero COM, rescaled to T0)
static void init_velocities_MB(MDSim& sim, int N, int dim) {
    std::mt19937 rng(12345);
    std::normal_distribution<double> gauss(0.0, 1.0);
    vec v_com(0, dim);
    // assign random
    for (MDParticleListEntry* e = sim.particles->getFirst(); e; e = e->getNext()) {
        MDParticle* p = e->getThis();
        for (int k = 1; k <= dim; k++) p->v[k] = gauss(rng);
        v_com += p->v;
    }
    v_com *= (1.0 / N);
    // remove COM
    for (MDParticleListEntry* e = sim.particles->getFirst(); e; e = e->getNext())
        e->getThis()->v -= v_com;
    // scale to target temperature
    double ke = 0.0;
    for (MDParticleListEntry* e = sim.particles->getFirst(); e; e = e->getNext())
        ke += e->getThis()->v * e->getThis()->v;
    ke *= 0.5; // m=1
    double dof = 3.0 * N - 3.0;          // remove 3 COM dof
    double ke_target = 0.5 * dof * T0; // equipartition
    double s = std::sqrt(ke_target / ke);
    for (MDParticleListEntry* e = sim.particles->getFirst(); e; e = e->getNext())
        e->getThis()->v *= s;
}

static double kinetic_energy(const MDSim& sim) {
    double ke = 0.0;
    for (MDParticleListEntry* e = sim.particles->getFirst(); e; e = e->getNext()) {
        const MDParticle* p = e->getThis();
        ke += p->v * p->v;
    }
    return 0.5 * ke;
}

// ---- Benchmark driver (3D, NVE, rebuild every 20 steps) ----
static double run_plimpton(int N, int steps) {
    // Box size from density and FCC fill
    gL = std::cbrt(N / RHO);
    gNc = (int)std::ceil(std::cbrt(N / 4.0)); // 4 atoms per FCC cell

    double boxArr[3] = { gL, gL, gL };
    MDSim sim(3, DT, f, pot, /*r_inter*/RC, /*r_verlet*/RS, /*verletUpdate*/0);
    for (int i = 0; i < N; i++) sim.particles->addParticle(new MDParticle(3));
    sim.getThermostat()->setT(T0);
    sim.getThermostat()->setNue(0.0); // NVE (no thermostat during time stepping)
    sim.initSim(true, vec(boxArr, 3), r0_fcc, /*v0*/nullptr, /*histRes*/100, /*histLen*/RC);
    init_velocities_MB(sim, N, 3);

    // Initial neighbor build once
    sim.benchRefresh(false, false);

    // Warm-up
    for (int s = 0; s < 5; s++) {
        sim.benchForce();
        sim.velocityVerletStep(false);
    }

    // Energy at start (optional drift check)
    double ke0 = kinetic_energy(sim);
    double pot0 = sim.benchForce(); // uses current neighbors
    double e0 = ke0 + pot0;

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    double pot_last = 0.0;
    for (int s = 0; s < steps; ++s) {
        if (s % REBUILD_EVERY == 0) sim.benchRefresh(false, false); // rebuild with RS
        pot_last = sim.benchForce();     // compute forces at RC
        sim.velocityVerletStep(false);   // integrate (half-kick, drift, half-kick)
    }
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    double ke1 = kinetic_energy(sim);
    double e1 = ke1 + pot_last;
    double drift_pct = (e1 != 0.0 ? 100.0 * (e1 - e0) / std::abs(e0) : 0.0);

#ifdef USE_CUDA
    std::cout << "[MODE] GPU ";
#else
    std::cout << "[MODE] CPU ";
#endif
    std::cout << "N=" << N << " steps=" << steps
        << " L=" << gL << " dt=" << DT << " rc=" << RC << " rs=" << RS
        << " time_ms=" << ms
        << " E_drift%=" << drift_pct << "\n";

    std::cout << "CSV,"
#ifdef USE_CUDA
#ifdef MD_MIXED_FP32
        << "GPU-FP32forces"
#else
        << "GPU-FP64"
#endif
#else
        << "CPU"
#endif
        << ',' << N << ',' << steps << ',' << gL << ',' << DT << ','
        << RC << ',' << RS << ',' << ms << ',' << drift_pct << "\n";

    return ms;
}

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::atoi(argv[1]) : 4000);
    int steps = (argc > 2 ? std::atoi(argv[2]) : 2000);
    return (int)run_plimpton(N, steps);
}

// This is a 3D NVE run, FCC positions at density ρ, 
// MB velocities at T=0.72 (zero COM & rescaled), rc=2.5σ, 
// neighbor rebuild every 20 steps with rs=2.8σ. It prints 
// time and energy drift %.