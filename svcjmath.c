#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "svcjmath.h"

#define PI 3.14159265358979323846
#define MAX_ITER 50
#define MIN_VOL 1e-6

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda, mu_j, sigma_j;
} Params;

// --- 1. Characteristic Function (Heston + Merton) ---
double complex cf_svcj(double complex u, double T, double r, double S0, double V0, Params p) {
    double complex xi = p.kappa - p.sigma_v * p.rho * u * I;
    double complex d = csqrt(xi*xi + p.sigma_v*p.sigma_v*(u*u + u*I));
    double complex g = (xi - d)/(xi + d);
    
    double complex A = (p.kappa*p.theta/(p.sigma_v*p.sigma_v)) * 
                       ((xi - d)*T - 2.0*clog((1.0 - g*cexp(-d*T))/(1.0 - g)));
    double complex B = ((xi - d)/(p.sigma_v*p.sigma_v)) * 
                       ((1.0 - cexp(-d*T))/(1.0 - g*cexp(-d*T)));
    
    double complex jump_drift = -p.lambda * (exp(p.mu_j + 0.5*p.sigma_j*p.sigma_j) - 1.0) * I * u * T;
    double complex jump_part = p.lambda * T * (cexp(I*u*p.mu_j - 0.5*p.sigma_j*p.sigma_j*u*u) - 1.0);
    
    return cexp(I*u*(log(S0) + r*T) + A + B*V0 + jump_part + jump_drift);
}

// --- 2. Option Pricing (Integration) ---
double price_opt(double S0, double K, double T, double r, double V0, Params p) {
    double alpha = 1.25;
    double k_log = log(K);
    double limit = 200.0;
    int N = 100;
    double eta = limit/N;
    double complex sum = 0.0 + 0.0*I;
    
    for(int j=0; j<N; j++) {
        double v = j*eta;
        double weight = (j==0)? 0.5 : 1.0;
        double complex u = v - (alpha + 1.0)*I;
        double complex num = cf_svcj(u, T, r, S0, V0, p);
        double complex den = (alpha + I*v)*(alpha + 1.0 + I*v);
        sum += weight * cexp(-I*v*k_log) * num / den;
    }
    double price = (exp(-alpha*k_log)/PI) * creal(sum*eta) * exp(-r*T);
    return (price > 0)? price : 1e-5;
}

// --- 3. UKF / Robust Filter ---
// Returns NLL. Fills final_vol and final_jump_prob.
double run_filter(const double* ret, int n, Params p, double* final_v, double* final_jp) {
    double v = p.theta;
    double nll = 0.0;
    double dt = 1.0/252.0;
    double jump_drift = p.lambda * (exp(p.mu_j + 0.5*p.sigma_j*p.sigma_j) - 1.0);
    double jp = 0.0;
    
    for(int t=0; t<n; t++) {
        // Predict
        double v_pred = v + p.kappa*(p.theta - v)*dt;
        if(v_pred < MIN_VOL) v_pred = MIN_VOL;
        
        // Innovation
        double mu_expect = (p.mu - 0.5*v_pred - jump_drift)*dt;
        double err = ret[t] - mu_expect;
        
        // Variance
        double var_diff = v_pred*dt;
        double var_jump = p.lambda*dt*(p.mu_j*p.mu_j + p.sigma_j*p.sigma_j);
        double var_tot = var_diff + var_jump;
        
        // Likelihood
        double pdf = (1.0/sqrt(2.0*PI*var_tot)) * exp(-0.5*err*err/var_tot);
        nll -= log((pdf > 1e-12)? pdf : 1e-12);
        
        // Update / Jump Detection
        double z = fabs(err) / (sqrt(var_diff) + 1e-9);
        jp = (z > 3.0)? 1.0 : (z > 2.0 ? z - 2.0 : 0.0);
        
        if(jp > 0.5) {
            v = v_pred; // Jump: Ignore shock in vol
        } else {
            double z_norm = err / sqrt(var_tot);
            v = v_pred + p.sigma_v * sqrt(v_pred*dt) * p.rho * z_norm;
        }
        if(v < MIN_VOL) v = MIN_VOL;
    }
    
    if(final_v) *final_v = v;
    if(final_jp) *final_jp = jp;
    return nll;
}

// --- 4. Optimizer (Coordinate Descent) ---
void optimize(const double* ret, int n, 
              const double* K, const double* P, const double* T, int n_opts, 
              double S0, double r, double* out) {
              
    Params p = {0.05, 2.0, 0.04, 0.3, -0.5, 0.1, -0.05, 0.1}; // Init
    double best_err = 1e15;
    double steps[] = {0.01, 0.5, 0.01, 0.05, 0.1, 0.1, 0.02, 0.02};
    
    // Heuristic Theta
    if(n > 10) {
        double sum_sq = 0;
        for(int i=0; i<n; i++) sum_sq += ret[i]*ret[i];
        p.theta = sum_sq / (n * (1.0/252.0));
        if(p.theta > 1.0) p.theta = 0.04;
    }
    
    for(int iter=0; iter<MAX_ITER; iter++) {
        int improved = 0;
        for(int i=0; i<8; i++) {
            double* val;
            switch(i) {
                case 0: val=&p.mu; break; case 1: val=&p.kappa; break;
                case 2: val=&p.theta; break; case 3: val=&p.sigma_v; break;
                case 4: val=&p.rho; break; case 5: val=&p.lambda; break;
                case 6: val=&p.mu_j; break; case 7: val=&p.sigma_j; break;
            }
            double old = *val;
            
            for(int dir=-1; dir<=1; dir+=2) {
                *val = old + dir * steps[i];
                // Constraints
                if(p.theta<1e-4) p.theta=1e-4; 
                if(p.sigma_v<1e-3) p.sigma_v=1e-3;
                if(p.rho<-0.99) p.rho=-0.99; if(p.rho>0.99) p.rho=0.99;
                if(p.lambda<0) p.lambda=0; if(p.sigma_j<1e-3) p.sigma_j=1e-3;
                
                double err = 0;
                // History NLL
                if(n > 0) err += run_filter(ret, n, p, NULL, NULL);
                
                // Option SSE
                if(n_opts > 0) {
                    double sse = 0;
                    for(int o=0; o<n_opts; o++) {
                        double mdl = price_opt(S0, K[o], T[o], r, p.theta, p);
                        sse += (mdl - P[o])*(mdl - P[o]);
                    }
                    err += sse * 5000.0;
                }
                
                // Feller Penalty
                if(2*p.kappa*p.theta < p.sigma_v*p.sigma_v) err += 1000.0;
                
                if(err < best_err) { best_err = err; improved = 1; }
                else *val = old;
            }
        }
        if(!improved) for(int k=0; k<8; k++) steps[k] *= 0.6;
    }
    
    // Output
    out[0]=p.mu; out[1]=p.kappa; out[2]=p.theta; out[3]=p.sigma_v;
    out[4]=p.rho; out[5]=p.lambda; out[6]=p.mu_j; out[7]=p.sigma_j;
    
    if(n > 0) run_filter(ret, n, p, &out[8], &out[9]); // Spot Vol, Jump Prob
    else { out[8]=p.theta; out[9]=0.0; }
}

// --- Entry Points ---

int svcj_matrix_fit(const double* ret_matrix, int n_time, int n_assets, 
                    int window, int step, double* out_buf) {
    int max_rolls = (n_time - window)/step + 1;
    if(max_rolls < 1) return 0;
    
    int idx = 0;
    for(int r=0; r<max_rolls; r++) {
        int start = r * step;
        for(int a=0; a<n_assets; a++) {
            double* win_data = (double*)malloc(window * sizeof(double));
            for(int k=0; k<window; k++) {
                win_data[k] = ret_matrix[(start+k)*n_assets + a];
            }
            
            double res[10];
            optimize(win_data, window, NULL, NULL, NULL, 0, 0, 0, res);
            
            for(int k=0; k<10; k++) out_buf[idx++] = res[k];
            
            free(win_data);
        }
    }
    return max_rolls;
}

void svcj_snapshot_fit(const double* returns, int n_ret, 
                       const double* strikes, const double* prices, const double* T, int n_opts,
                       double S0, double r, double* out_params) {
    optimize(returns, n_ret, strikes, prices, T, n_opts, S0, r, out_params);
}