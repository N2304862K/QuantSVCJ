/* svcj.c */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "svcj.h"

#define PI 3.14159265358979323846
#define MIN_VOL 1e-6
#define MAX_VOL 5.0

// --- 1. Characteristic Function (Heston + Merton) ---
double complex svcj_cf(double complex u, double T, double r, double S0, double V0, SVCJParams p) {
    double complex xi = p.kappa - p.sigma_v * p.rho * u * I;
    double complex d = csqrt(xi * xi + p.sigma_v * p.sigma_v * (u * u + u * I));
    double complex g = (xi - d) / (xi + d);
    
    double complex A = (p.kappa * p.theta / (p.sigma_v * p.sigma_v)) * 
                       ((xi - d) * T - 2.0 * clog((1.0 - g * cexp(-d * T)) / (1.0 - g)));
    double complex B = ((xi - d) / (p.sigma_v * p.sigma_v)) * 
                       ((1.0 - cexp(-d * T)) / (1.0 - g * cexp(-d * T)));
    
    double complex jump_drift = -p.lambda * (exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0) * I * u * T;
    double complex jump_part = p.lambda * T * (cexp(I * u * p.mu_j - 0.5 * p.sigma_j * p.sigma_j * u * u) - 1.0);
    
    return cexp(I * u * (log(S0) + r * T) + A + B * V0 + jump_part + jump_drift);
}

// --- 2. Option Pricing (Fourier) ---
// Returns Model Price for specified Type (1=Call, 0=Put)
double price_option_calc(double S0, double K, double T, double r, double V0, int is_call, SVCJParams p) {
    double alpha = 1.5;
    double k_log = log(K);
    double limit = 200.0;
    int N = 100; // Speed optimized
    double eta = limit / N;
    double complex sum = 0.0 + 0.0 * I;
    
    // Carr-Madan Call Pricing
    for (int j = 0; j < N; j++) {
        double v = j * eta;
        double weight = (j == 0) ? 0.5 : 1.0;
        double complex u = v - (alpha + 1.0) * I;
        double complex num = svcj_cf(u, T, r, S0, V0, p); 
        double complex denom = (alpha + I * v) * (alpha + 1.0 + I * v);
        sum += weight * cexp(-I * v * k_log) * num / denom;
    }
    
    double call_price = (exp(-alpha * k_log) / PI) * creal(sum * eta);
    call_price *= exp(-r*T);
    
    if (call_price < 0) call_price = 1e-5;

    // Logic Handling: If Put, convert via Parity
    if (is_call == 1) {
        return call_price;
    } else {
        // P = C - S + K*e^-rT
        return call_price - S0 + K * exp(-r * T);
    }
}

// --- 3. Robust Edge-Stabilized Filter ---
double run_filter(double* returns, int n, double dt, SVCJParams p, double* final_state) {
    double v = p.theta; 
    double log_lik = 0.0;
    double jump_drift = p.lambda * (exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0);
    double run_var = p.theta * dt; 
    double current_jump_prob = 0.0;

    for (int t = 0; t < n; t++) {
        double v_pred = v + p.kappa * (p.theta - v) * dt;
        if (v_pred < MIN_VOL) v_pred = MIN_VOL;
        if (v_pred > MAX_VOL) v_pred = MAX_VOL;

        double expected_ret = (0.0 - 0.5 * v_pred - jump_drift) * dt; 
        double innovation = returns[t] - expected_ret;
        
        double var_diff = v_pred * dt;
        double var_jump = p.lambda * dt * (p.mu_j * p.mu_j + p.sigma_j * p.sigma_j);
        double var_tot = var_diff + var_jump;
        
        run_var = 0.9 * run_var + 0.1 * (innovation * innovation); // Faster adaptation

        double pdf = (1.0 / sqrt(2.0 * PI * var_tot)) * exp(-0.5 * (innovation * innovation) / var_tot);
        log_lik += log((pdf > 1e-15) ? pdf : 1e-15);

        // Edge Stabilization: Local Z-Score
        double sigma_local = sqrt(var_diff);
        double z_score = fabs(innovation) / (sigma_local + 1e-9);
        
        if (z_score > 3.0) current_jump_prob = 1.0;
        else if (z_score < 2.0) current_jump_prob = 0.0;
        else current_jump_prob = (z_score - 2.0);

        if (current_jump_prob > 0.5) {
            v = v_pred; // Jump: Ignore shock in vol update
        } else {
            double z_innov = innovation / sqrt(var_tot);
            v = v_pred + p.sigma_v * sqrt(v_pred * dt) * p.rho * z_innov;
        }
        if (v < MIN_VOL) v = MIN_VOL;
    }

    if (final_state != NULL) {
        final_state[0] = v;
        final_state[1] = current_jump_prob;
    }

    return -log_lik;
}

// --- 4. Optimizer ---
SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int* types, int n_opts,
                         double S0, double r, int mode) {
    
    // Heuristic Initialization
    SVCJParams p = {2.5, 0.04, 0.4, -0.5, 0.1, -0.05, 0.1};
    if (n_ret > 10) {
         double sum_sq = 0;
         for(int i=0; i<n_ret; i++) sum_sq += returns[i]*returns[i];
         p.theta = sum_sq / (n_ret * dt);
         if(p.theta > 0.5) p.theta = 0.04;
    }

    SVCJResult res;
    double best_err = 1e15;
    SVCJParams best_p = p;
    
    double steps[] = {0.5, 0.005, 0.1, 0.1, 0.1, 0.02, 0.02};
    int max_iter = (mode == 1) ? 40 : 50; 

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
            
            for (int dir = -1; dir <= 1; dir += 2) {
                *val = old_val + dir * steps[i];
                
                // Constraints
                if (p.theta < 1e-4) p.theta = 1e-4;
                if (p.sigma_v < 1e-3) p.sigma_v = 1e-3;
                if (p.rho < -0.99) p.rho = -0.99; if (p.rho > 0.99) p.rho = 0.99;
                if (p.lambda < 0) p.lambda = 0;
                if (p.sigma_j < 1e-3) p.sigma_j = 1e-3;
                
                double err = 0;
                
                if (n_ret > 0) err += run_filter(returns, n_ret, dt, p, NULL);
                
                if (mode == 1 && n_opts > 0) {
                    double sse = 0;
                    for(int o=0; o<n_opts; o++) {
                        // Pass type flag (types[o]) directly to C pricing logic
                        double mdl = price_option_calc(S0, strikes[o], T_exp[o], r, p.theta, types[o], p);
                        double diff = mdl - prices[o];
                        sse += diff * diff;
                    }
                    err += sse * 1000.0; // Weighting
                }
                
                // Soft Feller Constraint
                if (2.0 * p.kappa * p.theta < p.sigma_v * p.sigma_v) {
                    err += 500.0; 
                }

                if (err < best_err) {
                    best_err = err;
                    best_p = p;
                    improved = 1;
                } else {
                    *val = old_val;
                }
            }
        }
        if (!improved) {
            for(int j=0; j<7; j++) steps[j] *= 0.6;
        }
    }
    
    double final_state[2] = {best_p.theta, 0};
    if (n_ret > 0) run_filter(returns, n_ret, dt, best_p, final_state);
    
    res.p = best_p;
    res.spot_vol = final_state[0];
    res.jump_prob = final_state[1];
    res.error = best_err;
    return res;
}