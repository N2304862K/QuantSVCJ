/* src/svcj_engine.c */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "svcj_engine.h"

#define PI 3.14159265358979323846

// --- 1. Characteristic Function (Heston + Merton Jump) ---
double complex svcj_cf(double complex u, double T, double r, double S0, double V0, SVCJParams p) {
    // Heston Dynamics
    double complex xi = p.kappa - p.sigma_v * p.rho * u * I;
    double complex d = csqrt(xi * xi + p.sigma_v * p.sigma_v * (u * u + u * I));
    double complex g = (xi - d) / (xi + d);
    
    double complex A = (p.kappa * p.theta / (p.sigma_v * p.sigma_v)) * 
                       ((xi - d) * T - 2.0 * clog((1.0 - g * cexp(-d * T)) / (1.0 - g)));
    double complex B = ((xi - d) / (p.sigma_v * p.sigma_v)) * 
                       ((1.0 - cexp(-d * T)) / (1.0 - g * cexp(-d * T)));
    
    // Jump Component (Merton)
    // Compensator drift adjustment handled here or in pricing drift
    double complex jump_part = p.lambda * T * (cexp(I * u * p.mu_j - 0.5 * p.sigma_j * p.sigma_j * u * u) - 1.0 - I * u * (cexp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0));

    return cexp(I * u * (log(S0) + r * T) + A + B * V0 + jump_part);
}

// --- 2. Option Pricing (Fourier Integration) ---
// Using simplified integration suitable for calibration speed
double price_option_c(double S0, double K, double T, double r, double V0, int is_call, SVCJParams p) {
    // Carr-Madan style damping
    double alpha = 1.5; 
    double k_log = log(K);
    
    // Integration parameters
    double limit = 100.0; // Integration limit
    int N = 100; // Number of steps (Discretization)
    double eta = limit / N;
    
    double complex sum = 0.0 + 0.0 * I;
    
    for (int j = 0; j < N; j++) {
        double v = j * eta;
        double weight = (j == 0) ? 0.5 : 1.0; // Trapezoidal rule
        
        // Modified CF for damping
        double complex u = v - (alpha + 1.0) * I;
        double complex phi = svcj_cf(u, T, r, S0, V0, p);
        
        // Denominator
        double complex denom = (alpha + I * v) * (alpha + 1.0 + I * v);
        
        sum += weight * cexp(-I * v * k_log) * phi / denom;
    }
    
    double price = (exp(-alpha * k_log) / PI) * creal(sum * eta);
    // Real world: Discount factor is in CF or applied here. Assuming CF handles risk-neutral drift.
    price *= exp(-r*T); 

    if (is_call == 0) {
        // Put-Call Parity: P = C - S + K*exp(-rT)
        price = price - S0 + K * exp(-r * T);
    }
    
    return (price > 0.0) ? price : 1e-5;
}

// --- 3. UKF Likelihood (Historical Data) ---
double calculate_log_likelihood(double* returns, int n, double dt, SVCJParams p) {
    double v = p.theta; // Start at mean
    double log_lik = 0.0;
    
    // Drift compensation for jumps
    double jump_drift = p.lambda * (exp(p.mu_j + 0.5*p.sigma_j*p.sigma_j) - 1.0);
    
    for (int t = 0; t < n; t++) {
        // Heston Euler Discretization
        double v_prev = v;
        double v_dt = v_prev * dt;
        if(v_dt < 0) v_dt = 1e-8;

        // Expected diffusive drift
        double mu = (0.0 - 0.5 * v_prev - jump_drift) * dt;
        double sigma2_tot = v_dt + (p.lambda * dt * (p.mu_j*p.mu_j + p.sigma_j*p.sigma_j));
        
        double error = returns[t] - mu;
        double pdf = (1.0 / sqrt(2.0 * PI * sigma2_tot)) * exp(-(error*error)/(2.0*sigma2_tot));
        
        log_lik += log((pdf > 1e-10) ? pdf : 1e-10);
        
        // Filter Step (Simplified Unscented Update logic)
        // If error is large, attribute to Jump, don't spike Volatility
        double jump_threshold = 3.0 * sqrt(v_dt);
        
        if (fabs(error) < jump_threshold) {
             // Diffusive update (correlation)
             double z = error / sqrt(sigma2_tot);
             v += p.kappa * (p.theta - v_prev) * dt + p.sigma_v * sqrt(v_dt) * p.rho * z;
        } else {
             // Jump: Vol reverts to mean
             v += p.kappa * (p.theta - v_prev) * dt;
        }
        
        if (v < 1e-4) v = 1e-4; // Feller / Positivity
    }
    return -log_lik; // Return Negative LL for Minimization
}

// --- 4. Optimization Routine (Coordinate Descent) ---
// Mode 0: History Only, Mode 1: Joint (History + Options)
SVCJParams optimize_core(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int n_opts,
                         double S0, double r, int mode) {

    SVCJParams p = {1.5, 0.04, 0.3, -0.5, 0.1, -0.05, 0.1}; // Initial Guess
    SVCJParams best_p = p;
    double best_err = 1e9;
    
    double steps[] = {0.2, 0.01, 0.05, 0.05, 0.05, 0.01, 0.01};
    int max_iter = (mode == 1) ? 30 : 50; // Fewer iters for joint as pricing is expensive

    for (int k = 0; k < max_iter; k++) {
        int improved = 0;
        for (int i = 0; i < 7; i++) {
            double* val = NULL;
            switch(i) {
                case 0: val=&p.kappa; break; case 1: val=&p.theta; break;
                case 2: val=&p.sigma_v; break; case 3: val=&p.rho; break;
                case 4: val=&p.lambda; break; case 5: val=&p.mu_j; break;
                case 6: val=&p.sigma_j; break;
            }
            double old_val = *val;
            
            // Try +Step, Try -Step
            for (int dir = -1; dir <= 1; dir += 2) {
                *val = old_val + dir * steps[i];
                
                // Constraints
                if (p.theta < 0) p.theta = 1e-4;
                if (p.sigma_v < 0) p.sigma_v = 1e-4;
                if (p.lambda < 0) p.lambda = 0;
                if (p.rho < -0.99) p.rho = -0.99; 
                if (p.rho > 0.99) p.rho = 0.99;
                
                double err = 0;
                
                // History Error
                if (n_ret > 0) {
                     err += calculate_log_likelihood(returns, n_ret, dt, p);
                }
                
                // Option Error
                if (mode == 1 && n_opts > 0) {
                    double opt_sse = 0;
                    for(int o=0; o<n_opts; o++) {
                        // Assuming all calls for simplicity in this demo, logic can be added for puts
                        double model_price = price_option_c(S0, strikes[o], T_exp[o], r, p.theta, 1, p);
                        double diff = model_price - prices[o];
                        opt_sse += diff * diff;
                    }
                    // Weighting: LL is usually ~ -1000, SSE is ~ 10. Scale SSE.
                    err += opt_sse * 100.0; 
                }
                
                if (err < best_err) {
                    best_err = err;
                    best_p = p;
                    improved = 1;
                } else {
                    *val = old_val; // Revert
                }
            }
        }
        if (!improved) {
            for(int j=0; j<7; j++) steps[j] *= 0.6; // Decay steps
        }
    }
    return best_p;
}