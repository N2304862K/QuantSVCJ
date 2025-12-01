#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// --- Config ---
#define MAX_ITER 50
#define DT (1.0/252.0)
#define JITTER 1e-8

// --- Structures ---
typedef struct {
    double mu;
    double kappa;
    double theta;
    double sigma_v;
    double rho;
    double lambda_j;
    double mu_j;
    double sigma_j;
} SVCJParams;

// --- Prototypes ---
void clean_returns(double* returns, int n);
void check_constraints(SVCJParams* p);
double run_filter(double* returns, int n, SVCJParams* p, double* out_spot, double* out_jump);
void fit_history(double* returns, int n, SVCJParams* p);
void calibrate_options(double s0, double* strikes, double* expiries, int* types, double* prices, int n_opts, SVCJParams* p);

#endif