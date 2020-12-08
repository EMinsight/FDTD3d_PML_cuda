#define _USE_MATH_DEFINES
#include <cmath>
#include "fdtd3d.h"

__global__ void E_update( int Nr, int Nth, int Nph,
                    float *Er, float *Eth, float *Eph,
                    float *nDr, float *nDth, float *nDph,
                    float *oDr, float *oDth, float *oDph, float eps)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( ((i >= 0 ) && (i < Nr)) && ((j >= 1) && (j < Nth)) && ((k >= 1) && (k < Nph)) ){
        int idx = i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k;
        Er[idx] = Er[idx]
                + (nDr[idx] - oDr[idx])/eps;
        
       oDr[idx] =  nDr[idx];
    }

    if( ((i >= 1) && (i < Nr)) && ((j >= 0) && (j < Nth)) && ((k >= 1) && (k < Nph)) ){
        int idx = i*(Nth*(Nph+1)) + j*(Nph+1) + k;
        
        Eth[idx] = Eth[idx]
                + (nDth[idx] - oDth[idx])/eps;
    
        oDth[idx] = nDth[idx];
    }

    if( ((i >= 1) && (i < Nr)) && ((j >= 1) && (j < Nth)) && ((k >= 0) && (k < Nph)) ){
        int idx = i*((Nth+1)*Nph) + j*Nph + k;

        Eph[idx] = Eph[idx]
                + (nDph[idx] - oDph[idx])/eps;
        
        oDph[idx] = nDph[idx];
    }
}