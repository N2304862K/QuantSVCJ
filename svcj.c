#include "svcj.h"

// --- Helper: Denoise & Init ---
void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        if(fabs(returns[i]) < JITTER) {
            returns[i] = (i%2==0) ? JITTER : -JITTER;
        }
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->theta < 1e-5) p->theta = 1e-5;
    if(p->sigma_v < 0.05) p->sigma_v = 0.05;
    if(p->lambda_j < 1e-4) p->lambda_j = 1e-4;
    
    if(p->rho > 0.99) p->rho = 0.99;
    if(p->rho < -0.99) p->rho = -0.99;
    
    // Feller Constraint: 2*kappa*theta > sigma_v^2
    double feller = 2.0 * p->kappa * p->theta;
    if(p->sigma_v * p->sigma_v > feller) {
        p->sigma_v = sqrt(feller) * 0.99;
    }
}

// --- Core: UKF-QMLE Logic ---
double run_filter(double* returns, int n, SVCJParams* p, double* out_spot, double* out_jump) {
    double nll = 0.0;
    
    // Auto-initialize variance from first 20 days
    double v = 0.0; 
    int init_win = (n < 20) ? n : 20;
    for(int i=0; i<init_win; i++) v += returns[i]*returns[i];
    v = (v/init_win) / DT;
    if(v < 1e-5) v = p->theta;

    for(int t=0; t<n; t++) {
        // 1. Prediction
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6;

        // 2. Innovation
        double expected_ret = (p->mu - 0.5 * v_pred) * DT;
        double y = returns[t] - expected_ret;

        // 3. Variance (Diffusive + Jump)
        double jump_var = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double h = v_pred * DT + jump_var * DT;
        
        // 4. Update
        double k = (p->rho * p->sigma_v * DT) / h;
        v = v_pred + k * y;
        
        // Clamp
        if(v < 1e-6) v = 1e-6;
        if(v > 10.0) v = 10.0;

        // 5. Likelihood & Outputs
        nll += 0.5 * log(2*M_PI*h) + 0.5 * (y*y)/h;
        
        if(out_spot) out_spot[t] = sqrt(v);
        if(out_jump) {
            double z = (y*y)/h;
            out_jump[t] = (z > 9.0) ? 1.0 : 0.0; // 3-sigma event
        }
    }
    return nll;
}

// --- Coordinate Descent Optimizer ---
void fit_history(double* returns, int n, SVCJParams* p) {
    double best_nll = run_filter(returns, n, p, NULL, NULL);
    double step_sizes[] = {0.5, 0.01, 0.1, 0.05};
    double* ptrs[] = {&p->kappa, &p->theta, &p->sigma_v, &p->rho};

    for(int i=0; i<MAX_ITER; i++) {
        int improved = 0;
        for(int j=0; j<4; j++) {
            double orig = *ptrs[j];
            double step = step_sizes[j];
            
            // Try +
            *ptrs[j] += step; check_constraints(p);
            double nll = run_filter(returns, n, p, NULL, NULL);
            if(nll < best_nll) { best_nll = nll; improved=1; continue; }
            
            // Try -
            *ptrs[j] = orig - step; check_constraints(p);
            nll = run_filter(returns, n, p, NULL, NULL);
            if(nll < best_nll) { best_nll = nll; improved=1; continue; }
            
            *ptrs[j] = orig;
        }
        if(!improved) break;
    }
}

// --- Option Calibration ---
void calibrate_options(double s0, double* strikes, double* expiries, int* types, double* prices, int n_opts, SVCJParams* p) {
    // Calibrate Theta (Risk Neutral) to minimize pricing error
    double best_theta = p->theta;
    double min_err = 1e15;
    double base_theta = p->theta;

    for(double m=0.5; m<2.5; m+=0.1) {
        p->theta = base_theta * m;
        double err = 0.0;
        
        double tot_vol = sqrt(p->theta + p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j));
        
        for(int i=0; i<n_opts; i++) {
            double T = expiries[i];
            double K = strikes[i];
            
            double d1 = (log(s0/K) + 0.5*tot_vol*tot_vol*T) / (tot_vol*sqrt(T));
            double d2 = d1 - tot_vol*sqrt(T);
            double nd1 = 0.5 * erfc(-d1 * M_SQRT1_2);
            double nd2 = 0.5 * erfc(-d2 * M_SQRT1_2);
            
            double model = (types[i]==1) ? (s0*nd1 - K*nd2) : (K*(1-nd2) - s0*(1-nd1));
            err += (model - prices[i])*(model - prices[i]);
        }
        
        if(err < min_err) { min_err = err; best_theta = p->theta; }
    }
    p->theta = best_theta;
}