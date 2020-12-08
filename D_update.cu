#define _USE_MATH_DEFINES
#include <cmath>
#include "fdtd3d.h"

__global__ void D_update( int Nr, int Nth, int Nph, float *nDr, float *nDth, float *nDph,
                         float *oDr, float *oDth, float *oDph, float *Hr, float *Hth, float *Hph,
                         float del_r, float del_th, float del_ph, float dt, float th0, float r0 )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( ((i >= 0) && (i < Nr)) && ((j >= 1) && (j < Nth)) && ((k >= 1) && (k < Nph)) ){
        int idx_Dr = i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k;
        int idx_Hph1 = i*(Nth*(Nph+1)) + j*(Nph+1) + k;
        int idx_Hph2 = i*(Nth*(Nph+1)) + (j-1)*(Nph+1) + k;
        int idx_Hth1 = i*((Nth+1)*Nph) + j*Nph + k;
        int idx_Hth2 = i*((Nth+1)*Nph) + j*Nph + k - 1;

        float r_i2 = (float(i)+0.5)*del_r + r0;
        float si_th1 = std::sin(th0 + (float(j)-0.5)*del_th);
        float si_th2 = std::sin(th0 + float(j)*del_th);
        float si_th3 = std::sin(th0 + (float(j)+0.5)*del_th);
        
        float CDr1 = dt/r_i2/si_th2/del_th;
        float CDr2 = dt/r_i2/si_th2/del_ph;

        nDr[idx_Dr] = oDr[idx_Dr]
            + CDr1*( si_th3*Hph[idx_Hph1] - si_th1*Hph[idx_Hph2] )
            - CDr2*( Hth[idx_Hth1] - Hth[idx_Hth2] );
    }

    if( ((i >= 1) && (i < Nr)) && ((j >= 0) && (j < Nth)) && ((k >= 1) && (k < Nph)) ){
        int idx_Dth = i*(Nth*(Nph+1)) + j*(Nph+1) + k;
        int idx_Hr1 = i*(Nth*Nph) + j*Nph + k;
        int idx_Hr2 = i*(Nth*Nph) + j*Nph + k - 1;
        int idx_Hph1 = i*(Nth*(Nph+1)) + j*(Nph+1) + k;
        int idx_Hph2 = (i-1)*(Nth*(Nph+1)) + j*(Nph+1) + k;

        float r_i1 = (float(i)-0.5)*del_r + r0;
        float r_i2 = float(i)*del_r + r0;
        float r_i3 = (float(i)+0.5)*del_r + r0;
        float si_th3 = std::sin(th0 + (float(j)+0.5)*del_th);

        float CDth1 = dt/r_i2/si_th3/del_ph;
        float CDth2 = dt/r_i2/del_r;

        nDth[idx_Dth] = oDth[idx_Dth]
            + CDth1*( Hr[idx_Hr1] - Hr[idx_Hr2])
            - CDth2*( r_i3*Hph[idx_Hph1] - r_i1*Hph[idx_Hph2]);
    }

    if( ((i >= 1) && (i < Nr)) && ((j >= 1) && (j < Nth)) && ((k >= 0) && (k < Nph)) ){
        int idx_Dph = i*((Nth+1)*Nph) + j*Nph + k;
        int idx_Hth1 = i*((Nth+1)*Nph) + j*Nph + k;
        int idx_Hth2 = (i-1)*((Nth+1)*Nph) + j*Nph + k;
        int idx_Hr1 = i*(Nth*Nph) + j*Nph + k;
        int idx_Hr2 = i*(Nth*Nph) + (j-1)*Nph + k;
        
        float r_i1 = (float(i)-0.5)*del_r + r0;
        float r_i2 = float(i)*del_r + r0;
        float r_i3 = (float(i)+0.5)*del_r + r0;

        float CDph1 = dt/r_i2/del_r;
        float CDph2 = dt/r_i2/del_th;

        nDph[idx_Dph] = oDph[idx_Dph]
            + CDph1*( r_i3*Hth[idx_Hth1] - r_i1*Hth[idx_Hth2])
            - CDph2*( Hr[idx_Hr1] - Hr[idx_Hr2] );
    }

}