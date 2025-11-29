#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "svcj.h"

#define PI 3.14159265358979323846
#define MIN_VOL 1e-6

// --- Math Kernels ---

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

double price_option(double S0, double K, double T, double r, double V0, SVCJParams p) {
    double alpha = 1.25;
    double k_log = log(K);
    double limit = 200.0;
    int N = 128;
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
    
    double price = (exp(-alpha * k_log) / PI) * creal(sum * eta);
    price *= exp(-r*T); 
    return (price > 0.0) ? price : 1e-8;
}

double run_ukf(double* returns, int n, double dt, SVCJParams p, double* out_vol, double* out_jump) {
    double v = p.theta;
    double log_lik = 0.0;
    double jump_correction = p.lambda * (exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0);
    double v_smooth = v; 
    double j_prob = 0.0;

    for (int t = 0; t < n; t++) {
        double v_pred = v + p.kappa * (p.theta - v) * dt;
        if (v_pred < MIN_VOL) v_pred = MIN_VOL;
        
        double mu = (0.0 - 0.5 * v_pred - jump_correction) * dt;
        double error = returns[t] - mu;
        
        double var_diff = v_pred * dt;
        double var_jump = p.lambda * dt * (p.mu_j * p.mu_j + p.sigma_j * p.sigma_j);
        double var_tot = var_diff + var_jump;
        
        double pdf = (1.0 / sqrt(2.0 * PI * var_tot)) * exp(-0.5 * (error * error) / var_tot);
        log_lik += log((pdf > 1e-15) ? pdf : 1e-15);
        
        double sigma_diff = sqrt(var_diff);
        double z = fabs(error) / (sigma_diff + 1e-9);
        
        j_prob = 0.0;
        if (z > 3.0) j_prob = 1.0; else if (z > 2.0) j_prob = z - 2.0;

        if (j_prob > 0.6) {
            v = v_pred;
        } else {
            double z_norm = error / sqrt(var_tot);
            v = v_pred + p.sigma_v * sqrt(v_pred * dt) * p.rho * z_norm;
        }
        if (v < MIN_VOL) v = MIN_VOL;
        v_smooth = 0.8 * v_smooth + 0.2 * v;
    }
    if (out_vol) *out_vol = v_smooth;
    if (out_jump) *out_jump = j_prob;
    return -log_lik;
}

// --- Optimization ---

SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int n_opts,
                         double S0, double r, int mode) {
    SVCJParams p = {2.0, 0.04, 0.3, -0.6, 0.1, -0.05, 0.1}; 
    if (n_ret > 20) {
        double acc = 0;
        for(int i=0; i<n_ret; i++) acc += returns[i]*returns[i];
        p.theta = acc / (n_ret * dt);
        if(p.theta > 2.0) p.theta = 0.04;
    }

    SVCJResult res;
    SVCJParams best_p = p;
    double best_err = 1e15;
    double steps[] = {0.5, 0.01, 0.1, 0.1, 0.05, 0.02, 0.02};
    int max_iter = (mode == 1) ? 50 : 30;

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
            double old = *val;
            for (int dir = -1; dir <= 1; dir += 2) {
                *val = old + dir * steps[i];
                if (p.theta < 1e-4) p.theta = 1e-4; if (p.sigma_v < 1e-3) p.sigma_v = 1e-3;
                if (p.rho < -0.99) p.rho = -0.99; if (p.rho > 0.99) p.rho = 0.99;
                if (p.lambda < 0) p.lambda = 0; if (p.sigma_j < 1e-3) p.sigma_j = 1e-3;

                double err = 0.0;
                if (n_ret > 0) err += run_ukf(returns, n_ret, dt, p, NULL, NULL);
                if (mode == 1 && n_opts > 0) {
                    double sse = 0;
                    for(int o=0; o<n_opts; o++) {
                        double mdl = price_option(S0, strikes[o], T_exp[o], r, p.theta, p);
                        sse += pow(mdl - prices[o], 2);
                    }
                    err += sse * 2000.0;
                }
                if (2*p.kappa*p.theta < p.sigma_v*p.sigma_v) err += 1000.0;

                if (err < best_err) { best_err = err; best_p = p; improved = 1; } 
                else { *val = old; }
            }
        }
        if (!improved) for(int j=0; j<7; j++) steps[j] *= 0.6;
    }
    res.p = best_p; res.error = best_err;
    if (n_ret > 0) run_ukf(returns, n_ret, dt, best_p, &res.spot_vol, &res.jump_prob);
    else { res.spot_vol = best_p.theta; res.jump_prob = 0.0; }
    return res;
}