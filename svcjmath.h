#ifndef SVCJMATH_H
#define SVCJMATH_H

// 9 Parameters: mu, kappa, theta, sigma_v, rho, lambda, mu_j, sigma_j, spot_vol, jump_prob
#define NUM_PARAMS 10 

// Core Estimation Function for Matrix/Rolling (History Only)
// Returns number of windows processed
int svcj_matrix_fit(const double* ret_matrix, int n_time, int n_assets, 
                    int window, int step, double* output_buffer);

// Core Estimation for Snapshot (History + Options)
void svcj_snapshot_fit(const double* returns, int n_ret, 
                       const double* strikes, const double* prices, const double* T, int n_opts,
                       double S0, double r, 
                       double* out_params);

#endif