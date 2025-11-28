/* src/svcj_engine.h */
#ifndef SVCJ_ENGINE_H
#define SVCJ_ENGINE_H

typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double rho;
    double lambda;
    double mu_j;
    double sigma_j;
} SVCJParams;

// Main Optimizer Entry Point
SVCJParams optimize_core(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int n_opts,
                         double S0, double r, int mode);

#endif