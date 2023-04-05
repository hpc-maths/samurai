#ifndef INTEGRATION_STIFF_H
#define INTEGRATION_STIFF_H

void test();
typedef void (*func_radau)(const int*, const double*, const double*, double*, const double*, const int*);
typedef void (*func_rock)(const int*, const double*, const double*, double*);

typedef void (*func_solout_radau)(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*);

void radau5_integration(double tini,
                        double tend,
                        int neq,
                        /*double *uini, */ double* u,
                        func_radau fcn,
                        func_solout_radau solout,
                        double tol,
                        int mljac,
                        int* info);

void rock4_integration(double tini, double tend, int neq, /*double *uini, */ double* u, func_rock fcn, double tol, int* info);

void rock4_integration_history(double tini, double tend, int neq, double* uini, double* u, func_rock fcn, double tol, double* dt_rock, int* info);

#ifdef __cplusplus
extern "C"
{
#endif
    void radau5_(int* n,
                 func_radau fcn,
                 double* x,
                 double* y,
                 double* xend,
                 double* h,
                 double* rtol,
                 double* atol,
                 int* itol,
                 void jac_radau(int*, double*, double*, double*, int*, double*, double*),
                 int* ijac,
                 int* mljac,
                 int* mujac,
                 void mas_radau(int*, double*, int*, int*, int*),
                 int* imas,
                 int* mlmas,
                 int* mumas,
                 func_solout_radau solout,
                 int* iout,
                 double* work,
                 int* lwork,
                 int* iwork,
                 int* liwork,
                 double* rpar,
                 int* ipar,
                 int* idid);

    void
    rock4_(int* n, double* t, double* tend, double* dt, double* u, func_rock fcn, double* atol, double* rtol, double* work, int* iwork, int* idid);

    double rho_(int* n, double* t, double u);

#ifdef __cplusplus
}
#endif

void jac_radau(int* n, double* x, double* y, double* dfy, int* ldfy, double* rpar, double* ipar);

void mas_radau(int* n, double* am, int* lmas, int* rpar, int* ipar);

#endif
