#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include "fdtd3d.h"

__global__ void add_J( 
float del_r, float del_th, float del_ph, 
float *Eth, int i_s, int j_s, int k_s, float t, 
float t0, float sig, float r0, int Nr, int Nth, int Nph )
{
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
        int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

        if( (i == i_s) && (j == j_s) && (k == k_s) ){
                int idx = i*(Nth*(Nph+1)) + j*(Nph+1) + k;

                float J = -( ( t-t0 )/sig/sig/del_r/(((float(i)+0.5)*del_r+ r0)*del_th)/(((float(i)+0.5)*del_r + r0)*del_ph) )
                * std::exp( -(t-t0)*(t-t0)/2.0/sig/sig);

                Eth[idx] = Eth[idx] + J;
        }

}