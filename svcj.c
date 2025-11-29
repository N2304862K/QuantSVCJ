/* svcj.c */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "svcj.h"

#define PI 3.14159265358979323846
#define MAX_ITER 100
#define MIN_VOL 1e-5

// --- 1. Characteristic Function (Heston + Merton Jump) ---
// Used for both Option Pricing and potentially advanced filtering
double complex svcj_cf(double complex u, double T, double r, double S0, double V0, SVCJParams p) {
    double complex xi = p.kappa - p.sigma_v * p.rho * u * I;
    double complex d = csqrt(xi * xi + p.sigma_v * p.sigma_v * (u * u + u * I));
    double complex g = (xi - d) / (xi + d);
    
    // Heston Part
    double complex A = (p.kappa * p.theta / (p.sigma_v * p.sigma_v)) * 
                       ((xi - d) * T - 2.0 * clog((1.0 - g * cexp(-d * T)) / (1.0 - g)));
    double complex B = ((xi - d) / (p.sigma_v * p.sigma_v)) * 
                       ((1.0 - cexp(-d * T)) / (1.0 - g * cexp(-d * T)));
    
    // Jump Part (Merton)
    // Drift correction for jump: lambda * k, where k = E[e^J]-1
    double complex jump_drift_correction = -p.lambda * (exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0) * I * u * T;
    
    double complex jump_part = p.lambda * T * (cexp(I * u * p.mu_j - 0.5 * p.sigma_j * p.sigma_j * u * u) - 1.0);
    
    // Combine: Risk Neutral Drift (r) + Heston + Jump - JumpDrift
    return cexp(I * u * (log(S0) + r * T) + A + B * V0 + jump_part + jump_drift_correction);
}

// --- 2. Option Pricing (Fourier Integration) ---
// Computes Call Price using Carr-Madan formulation (Damped)
double price_option(double S0, double K, double T, double r, double V0, SVCJParams p) {
    double alpha = 1.5; // Damping factor
    double k_log = log(K);
    
    // Integration Setup
    double limit = 200.0;
    int N = 200; 
    double eta = limit / N;
    double complex sum = 0.0 + 0.0 * I;
    
    // Trapezoidal Integration
    for (int j = 0; j < N; j++) {
        double v = j * eta;
        double weight = (j == 0) ? 0.5 : 1.0;
        
        // Carr-Madan Form: psi(v) = exp(-rT) * phi(v - (alpha+1)i) / (alpha^2 + alpha - v^2 + i(2alpha+1)v)
        double complex u = v - (alpha + 1.0) * I;
        double complex num = svcj_cf(u, T, r, S0, V0, p); 
        double complex denom = (alpha + I * v) * (alpha + 1.0 + I * v);
        
        sum += weight * cexp(-I * v * k_log) * num / denom;
    }
    
    double price = (exp(-alpha * k_log) / PI) * creal(sum * eta);
    price *= exp(-r*T); // Discounting

    // Put-Call Parity handling is done in Python or wrapper if needed, logic here assumes Call for calibration
    return (price > 0.0) ? price : 1e-8;
}

// --- 3. Unscented Kalman Filter (UKF) / Robust Filter ---
// Returns negative log likelihood
// Also populates final_state if provided: [SpotVol, JumpProb]
double run_filter(double* returns, int n, double dt, SVCJParams p, double* final_state) {
    double v = p.theta; // Initialize at long-run mean
    double log_lik = 0.0;
    
    // Jump Compensator for Physical Measure
    double jump_mean = p.lambda * (exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0);
    
    double current_jump_prob = 0.0;

    for (int t = 0; t < n; t++) {
        // A. Time Update (Prediction)
        // Discretized Heston Expectation
        double v_pred = v + p.kappa * (p.theta - v) * dt;
        if (v_pred < MIN_VOL) v_pred = MIN_VOL;
        
        // Expected Return (Drift)
        // Physical drift approximation ~ (r or mu) - 0.5*v - jump_mean
        // We assume neutral drift = 0 for filtering residuals, or small physical drift.
        // For robustness, we treat drift as negligible daily or constant.
        double mu_pred = (0.0 - 0.5 * v_pred - jump_mean) * dt; 
        
        // B. Measurement Update
        double y = returns[t];
        double innovation = y - mu_pred;
        
        // Total Variance = Diffusive Variance + Jump Variance
        double var_diff = v_pred * dt;
        double var_jump = p.lambda * dt * (p.mu_j * p.mu_j + p.sigma_j * p.sigma_j);
        double var_tot = var_diff + var_jump;
        
        // Likelihood (Gaussian Mixture Approximation)
        double pdf = (1.0 / sqrt(2.0 * PI * var_tot)) * exp(-0.5 * (innovation * innovation) / var_tot);
        log_lik += log((pdf > 1e-12) ? pdf : 1e-12);
        
        // C. State Update & Jump Detection
        // Calculate posterior probability of jump given observation
        // P(Jump | y) propto P(y | Jump) * P(Jump)
        // Simplified heuristic: Mahalanobis distance relative to diffusive vol
        double diff_std = sqrt(var_diff);
        double z_score = fabs(innovation) / (diff_std + 1e-9);
        
        // Sigmoid-like activation for jump probability
        current_jump_prob = 0.0;
        if (z_score > 3.0) {
            current_jump_prob = 1.0; // High confidence of jump
        } else if (z_score > 2.0) {
            current_jump_prob = (z_score - 2.0); // Linear ramp 2.0 to 3.0
        }

        if (current_jump_prob > 0.5) {
            // If Jump: Volatility doesn't spike due to the large return.
            // We trust the prediction or revert slightly.
            v = v_pred;
        } else {
            // If Diffusive: Update Volatility based on correlation rho
            // Heston correlation update: dv ~ rho * dS
            double z_innov = innovation / sqrt(var_tot);
            v = v_pred + p.sigma_v * sqrt(v_pred * dt) * p.rho * z_innov; 
        }

        // Feller / Positivity constraint
        if (v < MIN_VOL) v = MIN_VOL;
    }

    // Output latent states if requested
    if (final_state != NULL) {
        final_state[0] = v;                 // Spot Vol
        final_state[1] = current_jump_prob; // Last Jump Prob
    }

    return -log_lik; // Minimize NLL
}

// --- 4. Optimizer ---
// Coordinate Descent with constraints
SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int n_opts,
                         double S0, double r, int mode) {
    
    // Initial Guesses (Standard Market Params)
    SVCJParams p = {2.0, 0.04, 0.4, -0.7, 0.5, -0.05, 0.1}; 
    SVCJParams best_p = p;
    double best_err = 1e12;
    
    // Adaptive Steps
    double steps[] = {0.2, 0.005, 0.05, 0.05, 0.1, 0.01, 0.01};
    
    // Iteration limit based on complexity
    int iterations = (mode == 1) ? 40 : 80; 

    for (int iter = 0; iter < iterations; iter++) {
        int improved = 0;
        
        for (int i = 0; i < 7; i++) {
            double* ptr = NULL;
            switch(i) {
                case 0: ptr=&p.kappa; break; case 1: ptr=&p.theta; break;
                case 2: ptr=&p.sigma_v; break; case 3: ptr=&p.rho; break;
                case 4: ptr=&p.lambda; break; case 5: ptr=&p.mu_j; break;
                case 6: ptr=&p.sigma_j; break;
            }
            double val = *ptr;
            
            // Search Neighbors
            for (int dir = -1; dir <= 1; dir += 2) {
                *ptr = val + dir * steps[i];
                
                // --- Feller & Domain Constraints ---
                if (p.theta < 0.001) p.theta = 0.001;
                if (p.kappa < 0.1) p.kappa = 0.1;
                if (p.sigma_v < 0.01) p.sigma_v = 0.01;
                
                // Feller: 2*k*theta > sigma_v^2 (Soft constraint via penalty or hard reset)
                // We allow violation but prefer satisfaction.
                
                if (p.rho < -0.99) p.rho = -0.99;
                if (p.rho > 0.99) p.rho = 0.99;
                if (p.lambda < 0.0) p.lambda = 0.0;
                if (p.sigma_j < 0.001) p.sigma_j = 0.001;

                double err = 0.0;
                
                // 1. History Error (NLL)
                if (n_ret > 0) {
                    err += run_filter(returns, n_ret, dt, p, NULL);
                }
                
                // 2. Option Error (SSE)
                if (mode == 1 && n_opts > 0) {
                    double sse = 0.0;
                    for (int o = 0; o < n_opts; o++) {
                        // Assuming S0/K logic is normalized or consistent
                        // Pricing uses theta as V0 proxy for long-term calibration
                        double mdl = price_option(S0, strikes[o], T_exp[o], r, p.theta, p);
                        double diff = mdl - prices[o];
                        sse += diff * diff;
                    }
                    // Balance NLL and SSE magnitude
                    // NLL ~ -1000s, SSE ~ 10s. Weight SSE heavily.
                    err += sse * 5000.0; 
                }
                
                if (err < best_err) {
                    best_err = err;
                    best_p = p;
                    improved = 1;
                } else {
                    *ptr = val; // Revert
                }
            }
        }
        
        // Decay steps if no improvement
        if (!improved) {
            for(int k=0; k<7; k++) steps[k] *= 0.6;
        }
    }
    
    // Final Pass: Get Spot Vol and Jump Prob
    double final_state[2];
    if (n_ret > 0) {
        run_filter(returns, n_ret, dt, best_p, final_state);
    } else {
        final_state[0] = best_p.theta;
        final_state[1] = 0.0;
    }
    
    SVCJResult res;
    res.p = best_p;
    res.spot_vol = final_state[0];
    res.jump_prob = final_state[1];
    res.error = best_err;
    
    return res;
}