#include "integration_stiff.h"

void radau5_integration(double tini,
                        double tend,
                        int neq,
                        /*double *uini, */ double* u,
                        func_radau fcn,
                        func_solout_radau solout,
                        double tol,
                        int mljac,
                        int* info)
{
    double t, dt;

    //  required tolerance
    double rtol = tol;
    double atol = rtol;

    int itol = 0;

    // jacobian is computed internally by finite differences
    int ijac = 0;
    // jacobian is a full matrix
    int mujac = mljac;

    // mass matrix (assumed to be the identity matrix)
    int imas = 0;
    int mlmas;
    int mumas;

    // output routine is used during integration
    int iout = 1;

    int ljac  = mljac + mujac + 1;
    int le    = 2 * mljac + mujac + 1;
    int lmas  = 0;
    int lwork = neq * (ljac + lmas + 3 * le + 12) + 20;
    double work[lwork];
    int liwork = 3 * neq + 20;
    int iwork[liwork];

    double rpar;
    int ipar;
    int idid;

    int i;

    // init values
    t  = tini;
    dt = 0.0;
    // for (ieq=0; ieq<neq; ++ieq) u[ieq] = uini[ieq];

    for (i = 0; i < lwork; i++)
    {
        work[i] = 0.0;
    }
    for (i = 0; i < liwork; i++)
    {
        iwork[i] = 0.0;
    }

    // directly calling fortran
    radau5_(&neq,
            fcn,
            &t,
            u,
            &tend,
            &dt,
            &rtol,
            &atol,
            &itol,
            jac_radau,
            &ijac,
            &mljac,
            &mujac,
            mas_radau,
            &imas,
            &mlmas,
            &mumas,
            solout,
            &iout,
            work,
            &lwork,
            iwork,
            &liwork,
            &rpar,
            &ipar,
            &idid);

    // save & print statistics
    info[0] = iwork[13];
    info[1] = iwork[14];
    info[2] = iwork[15];
    info[3] = iwork[16];
    info[4] = iwork[17];
    info[5] = iwork[18];
    info[6] = iwork[19];
    info[7] = iwork[20];
}

void rock4_integration(double tini,
                       double tend,
                       int neq,
                       /*double *uini, */ double* u,
                       func_rock fcn,
                       double tol,
                       int* info)
{
    double t, dt;
    double atol, rtol;
    int iwork[12];
    double work[8 * neq];
    int idid;
    // int ieq;

    // initialisation
    t = tini;
    // initial step size
    dt = 1.e-6;

    // for (ieq=0; ieq<neq; ++ieq) u[ieq] = uini[ieq];

    // initialize iwork:
    // iwork[1]=0  ROCK4 attempts to compute the spectral radius internaly
    // iwork[2]=1  The Jacobian is constant
    // iwork[3]=0  Return and solution at tend.
    // iwork[4]=0  Atol and rtol are scalars.
    iwork[0] = 0;
    iwork[1] = 1;
    iwork[2] = 0;
    iwork[3] = 0;

    // required tolerance
    rtol = tol;
    atol = rtol;

    idid = 1;

    // directly calling fortran
    rock4_(&neq, &t, &tend, &dt, u, fcn, &atol, &rtol, work, iwork, &idid);

    // save & print statistics
    info[0] = iwork[4];
    info[1] = iwork[5];
    info[2] = iwork[6];
    info[3] = iwork[7];
    info[4] = iwork[8];
    info[5] = iwork[9];
    info[6] = iwork[10];
    info[7] = iwork[11];
}

void rock4_integration_history(double tini, double tend, int neq, double* uini, double* u, func_rock fcn, double tol, double* dt_rock, int* info)
{
    double t, dt;
    double atol, rtol;
    int iwork[12];
    double work[8 * neq];
    int ieq;
    int idid;
    int icmpt;

    // initialisation
    t = tini;
    // initial step size
    dt = 1.e-6;

    for (ieq = 0; ieq < neq; ++ieq)
    {
        u[ieq] = uini[ieq];
    }

    // initialize iwork:
    // iwork[1]=0  ROCK4 attempts to compute the spectral radius internaly
    // iwork[2]=1  The Jacobian is constant
    // iwork[3]=0  Return and solution at tend.
    // iwork[4]=0  Atol and rtol are scalars.
    iwork[0] = 0;
    iwork[1] = 1;
    iwork[2] = 1;
    iwork[3] = 0;

    // required tolerance
    rtol = tol;
    atol = rtol;

    idid  = 2;
    icmpt = 0;
    // directly calling fortran
    while (idid == 2)
    {
        dt_rock[icmpt] = dt;
        rock4_(&neq, &t, &tend, &dt, u, fcn, &atol, &rtol, work, iwork, &idid);
        icmpt = icmpt + 1;
    }

    // save & print statistics
    info[0] = icmpt;
}

double rho_(int* n, double* t, double u)
{
    return 0.0;
}

void jac_radau(int* n, double* x, double* y, double* dfy, int* ldfy, double* rpar, double* ipar)
{
}

void mas_radau(int* n, double* am, int* lmas, int* rpar, int* ipar)
{
}
