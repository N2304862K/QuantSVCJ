#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double kappa;       
    double theta;       
    double sigma_v;     
    double rho;         
    double lambda_j;    
    double mu_j;        
    double sigma_j;     
    double v0;          
} SVCJParams;

typedef struct {
    double spot_vol;       
    double jump_prob;      
    double drift_residue;  
} FilterState;

typedef struct {
    double strike;
    double price;
    double T_years;
    int is_call;
    int valid; 
} OptionContract;

void calculate_returns_from_prices(double* prices, int n_prices, double* out_returns);
void optimize_params_history(double* returns, int n_steps, SVCJParams* out);
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);

#endif