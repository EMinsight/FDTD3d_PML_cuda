#define _USE_MATH_DEFINES
#include <cmath>
#include "fdtd3d.h"

__global__ void H_update( int Nr, int Nth, int Nph, float *Er, float *Eth, float *Eph,
                    float *Hr, float *Hth, float *Hph, float del_r, float del_th, float del_ph, float dt,
                    float th0, float r0, float mu )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( ((i >= 1) && (i < Nr+1)) && ((j >= 0) && (j < Nth)) && ((k >= 0) && (k < Nph)) ){
        int idx_Hr = i*(Nth*Nph) + j*Nph + k;
        int idx_Eph1 = i*((Nth+1)*Nph) + (j+1)*Nph + k;
        int idx_Eph2 = i*((Nth+1)*Nph) + j*Nph + k;
        int idx_Eth1 = i*(Nth*(Nph+1)) + j*(Nph+1) + k + 1;
        int idx_Eth2 = i*(Nth*(Nph+1)) + j*(Nph+1) + k;

        float r_i1 = float(i)*del_r + r0;
        float si_th1 = std::sin(th0 + float(j)*del_th);
        float si_th2 = std::sin(th0 + float(j+0.5)*del_th);
        float si_th3 = std::sin(th0 + float(j+1.0)*del_th);

        float CHr1 = dt/mu/r_i1/si_th2/del_th;
        float CHr2 = dt/mu/r_i1/si_th2/del_ph;

        Hr[idx_Hr] = Hr[idx_Hr]
            - CHr1*(si_th3*Eph[idx_Eph1] - si_th1*Eph[idx_Eph2])
            + CHr2*(Eth[idx_Eth1] - Eth[idx_Eth2] );
    }

    if( ((i >= 0) && (i < Nr)) && ((j >= 1) && (j < Nth+1)) && ((k >= 0) && (k < Nph)) ){
        int idx_Hth = i*((Nth+1)*Nph) + j*Nph + k;
        int idx_Er1 = i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k + 1;
        int idx_Er2 = i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k;
        int idx_Eph1 = (i+1)*((Nth+1)*Nph) + j*Nph + k;
        int idx_Eph2 = i*((Nth+1)*Nph) + j*Nph + k;

        float r_i1 = float(i)*del_r + r0;
        float r_i2 = float(i+0.5)*del_r + r0;
        float r_i3 = float(i+1.0)*del_r + r0;
        float si_th1 = std::sin(th0 + (float)j*del_th);

        float CHth1 = dt/mu/r_i2/si_th1/del_ph;
        float CHth2 = dt/mu/r_i2/del_r;

        Hth[idx_Hth] = Hth[idx_Hth]
            - CHth1*(Er[idx_Er1] - Er[idx_Er2])
            + CHth2*(r_i3*Eph[idx_Eph1] - r_i1*Eph[idx_Eph2]);

    }

    if( (i >= 0) && (i < Nr) && (j >= 0) && (j < Nth) && (k >= 1) && (k < Nph+1) ){
        int idx_Hph = i*(Nth*(Nph+1)) + j*(Nph+1) + k;
        int idx_Eth1 = (i+1)*(Nth*(Nph+1)) + j*(Nph+1)  + k;
        int idx_Eth2 = i*(Nth*(Nph+1)) + j*(Nph+1)  + k;
        int idx_Er1 = i*((Nth+1)*(Nph+1)) + (j+1)*(Nph+1) + k;
        int idx_Er2 = i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k;

        float r_i1 = float(i)*del_r + r0;
        float r_i2 = (float(i)+0.5)*del_r + r0;
        float r_i3 = (float(i)+1.0)*del_r + r0;
    
        float CHph1 = dt/mu/r_i2/del_r;
        float CHph2 = dt/mu/r_i2/del_th;

        Hph[idx_Hph] = Hph[idx_Hph]
            - CHph1*(r_i3*Eth[idx_Eth1] - r_i1*Eth[idx_Eth2])
            + CHph2*(Er[idx_Er1] - Er[idx_Er2]);
    }
}