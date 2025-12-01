#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define SQRT_2PI 2.50662827463
#define INF_LOSS 1e15

// --- Internal Helper: Log Return Calculation ---
// Converts Raw Prices -> Log Returns in place (buffer size n-1)
void compute_log_returns(double* prices, int n, double* returns) {
    for(int i=0; i < n-1; i++) {
        if(prices[i] > 0 && prices[i+1] > 0)
            returns[i] = log(prices[i+1] / prices[i]);
        else
            returns[i] = 0.0;
    }
}

// --- Internal Helper: Option Surface Logic ---
// 1. Filters far OTM/illiquid options
// 2. Converts Days to Years
// 3. Estimates implied daily variance
double extract_implied_theta(RawOption* opts, int n_opts, double spot) {
    if(n_opts == 0) return 0.0;

    double sum_var = 0.0;
    int count = 0;
    double skew_sum = 0.0;

    for(int i=0; i<n_opts; i++) {
        // FILTER 1: Moneyness (Keep within 15% of spot)
        double moneyness = opts[i].strike / spot;
        if(moneyness < 0.85 || moneyness > 1.15) continue;

        // FILTER 2: Time (Ignore options < 7 days or > 1 year)
        if(opts[i].T_days < 7.0 || opts[i].T_days > 365.0) continue;

        // FILTER 3: Intrinsic Value check (remove deep ITM arbitrage)
        double intrinsic = opts[i].is_call ? (spot - opts[i].strike) : (opts[i].strike - spot);
        if (intrinsic > 0 && opts[i].price < intrinsic) continue;

        // MATH: Brenner-Subrahmanyam Approx for ATM Vol
        // Vol_Ann = (Price / Spot) * sqrt(2*PI / T_years)
        // We accept a wider range for "ATM" here to get more data points
        if(fabs(moneyness - 1.0) < 0.1) {
            double t_years = opts[i].T_days / 365.0;
            double vol_ann = (opts[i].price / spot) * sqrt(2.0 * PI / t_years);
            
            // Convert to Daily Variance
            double daily_var = (vol_ann * vol_ann) / 252.0;
            if(daily_var > 0 && daily_var < 0.01) { // Sanity check
                sum_var += daily_var;
                count++;
            }
        }
    }
    
    if(count == 0) return -1.0; // Signal failure
    return sum_var / count;
}

// --- Internal: Likelihood & Optimization (Same as before but condensed) ---
int check_constraints(SVCJParams* p) {
    if (p->theta < 1e-8 || p->theta > 0.01) return 0;
    if (p->kappa < 0.001 || p->kappa > 1.0) return 0;
    if (p->sigma_v < 1e-4 || p->sigma_v > 0.5) return 0;
    if (p->rho < -0.99 || p->rho > 0.99) return 0;
    return 1;
}

double calc_nll(double* rets, int n, SVCJParams* p) {
    if (!check_constraints(p)) return INF_LOSS;
    double nll = 0.0;
    double vt = p->v0;
    double jump_var = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);

    for (int i=0; i<n; i++) {
        if(vt < 1e-8) vt = 1e-8;
        if(vt > 0.005) vt = 0.005;
        
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        double tot_var = vt_pred + jump_var;
        
        double pdf = (1.0/(SQRT_2PI*sqrt(tot_var))) * exp(-0.5*rets[i]*rets[i]/tot_var);
        if(pdf < 1e-10) pdf = 1e-10;
        nll -= log(pdf);
        
        double shock = (rets[i]*rets[i]) - tot_var;
        vt = vt_pred + (p->sigma_v * 0.1) * shock;
    }
    return nll;
}

void optimize_internal(double* rets, int n, SVCJParams* out) {
    // 1. Init from Realized
    double sum_sq = 0.0;
    for(int i=0; i<n; i++) sum_sq += rets[i]*rets[i];
    double realized = sum_sq / n;
    
    out->theta = realized;
    out->v0 = realized;
    out->lambda_j = 0.05; out->mu_j = -0.01; out->sigma_j = sqrt(realized)*2.0;

    // 2. Grid Search
    double kappas[] = {0.02, 0.05, 0.10};
    double rhos[] = {-0.6, -0.3, 0.0};
    double sigmas[] = {0.0005, 0.002, 0.005};
    
    double best_nll = INF_LOSS;
    SVCJParams cand = *out;
    SVCJParams best = *out;
    
    for(int k=0; k<3; k++) {
        for(int r=0; r<3; r++) {
            for(int s=0; s<3; s++) {
                cand.kappa = kappas[k]; cand.rho = rhos[r]; cand.sigma_v = sigmas[s];
                double val = calc_nll(rets, n, &cand);
                if(val < best_nll) { best_nll = val; best = cand; }
            }
        }
    }
    *out = best;
}

void run_filter_internal(double* rets, int n, SVCJParams* p, FilterResult* out) {
    double vt = p->v0;
    for(int i=0; i<n; i++) {
        if(vt < 1e-8) vt = 1e-8;
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        double tot_var = vt_pred + p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        
        // Update
        double innov = rets[i];
        double kgain = (p->sigma_v * p->rho) / (tot_var + 1e-9);
        vt = vt_pred + kgain * (innov*innov - tot_var);
        
        if(vt < 1e-8) vt = 1e-8;
        if(vt > 0.01) vt = 0.01;
        
        double z = innov / sqrt(vt_pred);
        double jp = (fabs(z) > 3.0) ? (1.0 - exp(-(fabs(z)-3.0))) : 0.0;
        
        out[i].spot_vol = sqrt(vt);
        out[i].jump_prob = jp;
        out[i].drift_residue = innov;
    }
}

// --- PUBLIC EXPOSED FUNCTIONS ---

void full_svcj_pipeline(double* raw_prices, int n_price_steps,
                        RawOption* raw_options, int n_opts,
                        SVCJParams* out_params, FilterResult* out_states) {
    
    // 1. Pre-process: Calculate Returns internally
    int n_ret = n_price_steps - 1;
    double* returns = (double*)malloc(n_ret * sizeof(double));
    compute_log_returns(raw_prices, n_price_steps, returns);
    
    // 2. Optimization: Fit History
    optimize_internal(returns, n_ret, out_params);
    
    // 3. Option Integration: Filter & Smooth Option Surface
    // We pass the END spot price (last element)
    double spot_end = raw_prices[n_price_steps - 1];
    double implied_theta = extract_implied_theta(raw_options, n_opts, spot_end);
    
    if(implied_theta > 0) {
        // If options are valid, they override the historical long-run variance
        out_params->theta = implied_theta;
        out_params->v0 = implied_theta; // Anchor current state to option market
    }
    
    // 4. Run Filter
    run_filter_internal(returns, n_ret, out_params, out_states);
    
    free(returns);
}

void batch_process_matrix(double* price_matrix, int n_assets, int n_time, 
                          double* out_spot_vols, double* out_jump_probs) {
    
    // Used for screening. No options involved, just pure history fit.
    int n_ret = n_time - 1;
    
    for(int i=0; i<n_assets; i++) {
        double* asset_prices = &price_matrix[i * n_time];
        double* returns = (double*)malloc(n_ret * sizeof(double));
        compute_log_returns(asset_prices, n_time, returns);
        
        SVCJParams p;
        memset(&p, 0, sizeof(SVCJParams));
        optimize_internal(returns, n_ret, &p);
        
        FilterResult* res = (FilterResult*)malloc(n_ret * sizeof(FilterResult));
        run_filter_internal(returns, n_ret, &p, res);
        
        // Output last state
        out_spot_vols[i] = res[n_ret-1].spot_vol;
        out_jump_probs[i] = res[n_ret-1].jump_prob;
        
        free(returns);
        free(res);
    }
}