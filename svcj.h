#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>

// Configuration Constants (tuned for Daily Scale)
#define MAX_ITER 200
#define TOLERANCE 1e-6
#define JITTER_THRESHOLD 1e-9  
#define DT 1.0                 

typedef struct {
    double mu;          
    double kappa;       
    double theta;       
    double sigma_v;     
    double rho;         
    double lambda_j;    
    double mu_j;        
    double sigma_j;     
} SVCJParams;

void clean_returns(double* returns, int n, int stride);
void check_feller_and_fix(SVCJParams* params);
// We use the same 'stride' for reading 'returns' and writing 'out_*' 
double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices);

#endif