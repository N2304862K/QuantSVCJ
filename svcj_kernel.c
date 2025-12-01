#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define SQRT_2PI 2.50662827463
#define INF_LOSS 1e15

// Helper: Gaussian PDF
double density_norm(double x, double mean, double var) {
    if (var < 1e-9) var = 1e-9;
    return (1.0 / (SQRT_2PI * sqrt(var))) * exp(-0.5 * (x - mean)*(x - mean) / var);
}

// --- Constraint Checker (Daily Units) ---
int check_constraints(SVCJParams* p) {
    // 1. Boundaries suitable for DAILY variance (approx 0.0001)
    if (p->theta < 1e-8 || p->theta > 0.01) return 0; // Max 10% daily move allowed
    if (p->v0 < 1e-8 || p->v0 > 0.01) return 0;
    
    // 2. Kappa (Daily reversion is slower than annual)
    // 0.001 (very slow) to 0.5 (half-life of 2 days)
    if (p->kappa < 0.001 || p->kappa > 1.0) return 0; 
    
    // 3. Sigma_v (Vol of Vol)
    if (p->sigma_v < 1e-4 || p->sigma_v > 0.5) return 0;
    
    // 4. Correlation
    if (p->rho < -0.99 || p->rho > 0.99) return 0;

    // 5. Feller Condition (2 * k * theta > sigma_v^2)
    // Relaxed slightly for daily estimation noise
    double lhs = 2.0 * p->kappa * p->theta;
    double rhs = p->sigma_v * p->sigma_v;
    if (lhs < rhs * 0.7) return 0; // Hard stop if violation is severe

    return 1;
}

// --- Likelihood (Daily Terms) ---
double calculate_nll(double* returns, int n_steps, SVCJParams* p) {
    if (!check_constraints(p)) return INF_LOSS;

    double nll = 0.0;
    double vt = p->v0;
    
    // Pre-calc Jump Variance Contribution (Daily)
    double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);

    for (int i = 0; i < n_steps; i++) {
        // Clamp (Prevent numeric explosion)
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.005) vt = 0.005; // Cap at ~7% daily vol

        // 1. Expected Variance (Discrete Euler)
        // No 'dt' multiplication needed if kappa/theta are daily
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift; 
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        // 2. Total Variance = Continuous + Jumps
        double total_var = vt_pred + jump_var_contrib;
        
        // 3. Likelihood
        double lik = density_norm(returns[i], 0.0, total_var);
        if (lik < 1e-10) lik = 1e-10;
        nll -= log(lik);

        // 4. State Update (Proxy for Filter)
        double shock = (returns[i]*returns[i]) - total_var;
        vt = vt_pred + (p->sigma_v * 0.1) * shock; 
    }
    return nll;
}

// --- Optimizer: Smart Grid (Daily Config) ---
void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // 1. Initial Guess based on Realized Variance
    double sum_sq = 0.0;
    for(int i=0; i<n_steps; i++) sum_sq += returns[i]*returns[i];
    double realized_daily_var = sum_sq / n_steps;
    
    out->theta = realized_daily_var;
    out->v0 = realized_daily_var;
    out->lambda_j = 0.05; // 5% chance of jump per day
    out->mu_j = -0.01;
    out->sigma_j = sqrt(realized_daily_var) * 2.0;

    // 2. Grid Search (Ranges adapted for Daily terms)
    double kappas[] = {0.01, 0.05, 0.15};
    double rhos[] = {-0.7, -0.4, 0.0};
    double sigmas[] = {1e-4, 1e-3, 5e-3}; // Daily vol-of-vol is small
    
    double best_nll = INF_LOSS;
    SVCJParams best_p = *out;
    SVCJParams candidate = *out;

    for(int k=0; k<3; k++) {
        for(int r=0; r<3; r++) {
            for(int s=0; s<3; s++) {
                candidate.kappa = kappas[k];
                candidate.rho = rhos[r];
                candidate.sigma_v = sigmas[s];
                
                // Enforce Feller inside grid
                if (2*candidate.kappa*candidate.theta < candidate.sigma_v*candidate.sigma_v) {
                    // Reduce sigma_v to satisfy feller
                    candidate.sigma_v = sqrt(1.8 * candidate.kappa * candidate.theta);
                }

                double nll = calculate_nll(returns, n_steps, &candidate);
                if (nll < best_nll) {
                    best_nll = nll;
                    best_p = candidate;
                }
            }
        }
    }
    *out = best_p;
}

// --- UKF Filter (Daily Terms) ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    
    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;

        // 1. Predict
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        // 2. Measure
        double jump_var = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double total_var = vt_pred + jump_var;
        double innovation = returns[i];

        // 3. Update (Kalman Gain)
        // Gain logic tailored for daily steps
        double kalman_gain = (p->sigma_v * p->rho) / (total_var + 1e-9);
        double innov_sq = innovation * innovation;
        
        vt = vt_pred + kalman_gain * (innov_sq - total_var);
        
        // Hard clamp for stability
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.01) vt = 0.01; // Max 10% daily vol

        // 4. Jump Prob
        double z_score = innovation / sqrt(vt_pred);
        double jump_p = 0.0;
        if (fabs(z_score) > 3.0) {
            jump_p = 1.0 - exp(-(fabs(z_score)-3.0));
        }

        out_states[i].vt = vt;
        out_states[i].spot_vol = sqrt(vt); // This is DAILY Vol
        out_states[i].jump_prob = jump_p;
        out_states[i].drift_residue = innovation; // Simple residue
    }
}

// --- Option Calibration (Annual -> Daily Conversion) ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    if (n_opts == 0) return;

    double sum_implied_daily_var = 0.0;
    int count = 0;
    double skew_accum = 0.0;

    for(int i=0; i<n_opts; i++) {
        double moneyness = options[i].strike / spot;
        double price_norm = options[i].price / spot;
        double t_sqrt = sqrt(options[i].T); // T is in Years
        
        // ATM Approximation
        if (fabs(moneyness - 1.0) < 0.05) {
            // Brenner-Subrahmanyam: Vol_Ann ~ (Price/S) * 2.5 / sqrt(T)
            double vol_ann = (price_norm * 2.50) / t_sqrt;
            
            // CONVERT TO DAILY VARIANCE
            // Var_Daily = (Vol_Ann^2) / 252
            double var_daily = (vol_ann * vol_ann) / 252.0;
            
            sum_implied_daily_var += var_daily;
            count++;
        }
        
        // Skew Check
        if (moneyness < 0.95 && options[i].is_call == 0) skew_accum -= 1.0;
    }

    // Update Internal State to Match Option Market
    if (count > 0) {
        double avg_daily_var = sum_implied_daily_var / count;
        out->theta = avg_daily_var; // Long run daily variance
        out->v0 = avg_daily_var;    // Current daily variance
    }

    // Skew adjustment
    if (skew_accum < -3.0) out->rho = -0.8;
    else out->rho = -0.5;
}