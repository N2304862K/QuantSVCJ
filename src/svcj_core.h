/* src/svcj_core.h */
#ifndef SVCJ_CORE_H
#define SVCJ_CORE_H

typedef struct {
    double kappa;    // Mean reversion speed
    double theta;    // Long run variance
    double sigma_v;  // Vol of Vol
    double rho;      // Correlation
    double lambda;   // Jump Intensity
    double mu_j;     // Mean Jump size
    double sigma_j;  // Jump size std dev
} SVCJParams;

// Modes: 1=History Only, 2=Options Only, 3=Joint
SVCJParams optimize_svcj(double* returns, int n_ret, double dt, 
                         double* strikes, double* prices, double* T, int n_opts, 
                         double S0, double r, int mode);

#endif