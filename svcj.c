#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "svcj.h"

#define PI 3.14159265358979323846
#define MIN_VOL 1e-6

// Characteristic Function
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

// Option Pricing
double price_option_core(double S0, double K, double T, double r, double V0, SVCJParams p, int is_call) {
    double alpha = 1.5;
    double k_log = log(K);
    double limit = 200.0;
    int N = 120;
    double eta = limit / N;
    double complex sum = 0.0 + 0.0 * I;
    
    for (int j = 0; j < N; j++) {
        double v = j * eta;
        double weight = (j == 0) ? 0.5 : 1.0;
        double complex u = v - (alpha + 1.0) * I;
        double complex num = svcj_cf(u, T, r, S0, V0, p); 
        double complex denom = (alpha + I * v) * (alpha + 1.0 + I * v);
        sum += weight * cexp(-I * v * k_log) * num / denom;
    }
    
    double call_price = (exp(-alpha * k_log) / PI) * creal(sum * eta) * exp(-r*T);
    
    if (is_call) return (call_price > 0.0) ? call_price : 1e-5;
    
    double put_price = call_price - S0 + K * exp(-r * T);
    return (put_price > 0.0) ? put_price : 1e-5;
}

// Filter
double run_filter(double* returns, int n, double dt, SVCJParams p, double* final_state) {
    double v = p.theta; 
    double log_lik = 0.0;
    double jump_drift = p.lambda * (exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0);
    
    for (int t = 0; t < n; t++) {
        double v_pred = v + p.kappa * (p.theta - v) * dt;
        if (v_pred < MIN_VOL) v_pred = MIN_VOL;

        double mu = (0.0 - 0.5 * v_pred - jump_drift) * dt;
        double innov = returns[t] - mu;
        double var_tot = v_pred * dt + p.lambda * dt * (p.mu_j*p.mu_j + p.sigma_j*p.sigma_j);
        
        double pdf = (1.0 / sqrt(2.0 * PI * var_tot)) * exp(-0.5 * innov * innov / var_tot);
        log_lik += log((pdf > 1e-12) ? pdf : 1e-12);
        
        double z = fabs(innov) / (sqrt(v_pred * dt) + 1e-9);
        double j_prob = (z > 3.0) ? 1.0 : 0.0;
        
        if (j_prob > 0.5) v = v_pred;
        else v = v_pred + p.sigma_v * sqrt(v_pred * dt) * p.rho * (innov / sqrt(var_tot));
        
        if (v < MIN_VOL) v = MIN_VOL;
        if (t == n-1 && final_state != NULL) { final_state[0] = v; final_state[1] = j_prob; }
    }
    return -log_lik;
}

// Optimizer
SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int* types, int n_opts,
                         double S0, double r, int mode) {
    SVCJParams p = {2.0, 0.04, 0.3, -0.5, 0.1, -0.05, 0.1};
    SVCJResult res;
    double best_err = 1e15;
    
    if (n_ret > 10) {
        double sum = 0; 
        for(int i=0; i<n_ret; i++) sum += returns[i]*returns[i];
        p.theta = sum/(n_ret*dt);
        if (p.theta > 2.0) p.theta = 0.04;
    }

    double steps[] = {0.5, 0.01, 0.1, 0.1, 0.05, 0.02, 0.02};
    int max_iter = (mode == 1) ? 35 : 50;

    for (int k = 0; k < max_iter; k++) {
        int improved = 0;
        for (int i = 0; i < 7; i++) {
            double* val;
            switch(i) {
                case 0: val=&p.kappa; break; case 1: val=&p.theta; break;
                case 2: val=&p.sigma_v; break; case 3: val=&p.rho; break;
                case 4: val=&p.lambda; break; case 5: val=&p.mu_j; break;
                case 6: val=&p.sigma_j; break;
            }
            double old = *val;
            for (int dir = -1; dir <= 1; dir += 2) {
                *val = old + dir * steps[i];
                if (p.theta < 1e-5) p.theta = 1e-5;
                if (p.sigma_v < 1e-3) p.sigma_v = 1e-3;
                if (p.rho < -0.99) p.rho = -0.99; if (p.rho > 0.99) p.rho = 0.99;
                if (p.lambda < 0) p.lambda = 0;
                if (p.sigma_j < 1e-3) p.sigma_j = 1e-3;

                double err = 0;
                if (n_ret > 0) err += run_filter(returns, n_ret, dt, p, NULL);
                if (mode == 1 && n_opts > 0) {
                    double sse = 0;
                    for(int o=0; o<n_opts; o++) {
                        double mdl = price_option_core(S0, strikes[o], T_exp[o], r, p.theta, p, types[o]);
                        sse += pow(mdl - prices[o], 2);
                    }
                    err += sse * 3000.0;
                }
                if (2*p.kappa*p.theta < p.sigma_v*p.sigma_v) err += 1000.0;
                if (err < best_err) { best_err = err; improved = 1; }
                else *val = old;
            }
        }
        if (!improved) for(int j=0; j<7; j++) steps[j] *= 0.6;
    }
    double fst[2] = {p.theta, 0};
    if (n_ret > 0) run_filter(returns, n_ret, dt, p, fst);
    res.p = p; res.spot_vol = fst[0]; res.jump_prob = fst[1]; res.error = best_err;
    return res;
}