#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

const int Nr{100};
const int Nth{100};
const int Nph{1000};

constexpr float R_r{100.0e3};

const float delta_r{ R_r/float(Nr) };
const float delta_th{ 1.0e3/float(R0) };
const float delta_ph{ 1.0e3/float(R0) };
const float Dt { float( 0.99/C0/std::sqrt(1.0/delta_r/delta_r
 + 1.0/R0/R0/delta_th/delta_th
 + 1.0/R0/R0/std::sin(THETA0)/std::sin(THETA0)/delta_ph/delta_ph) ) };
const float inv_Dt = float(1.0/Dt);
const float sigma_t = float(7.0*Dt);
const float t0 = float(6.0*sigma_t);

// center point //
/*const int i_0{ Nr/2 };
const int j_0{ Nth/2 };
const int k_0{ Nph/2 };*/

// Source point //
const int i_s{1};
const int j_s{50};
const int k_s{100};

// Receive Point //
const int i_r{1};
const int j_r{50};
const int k_r{ Nph - 50 };

void calc_fdtd(void)
{

    float *Hr = new float[(Nr+1)*Nth*Nph];
    float *Hth = new float[Nr*(Nth+1)*Nph];
    float *Hph = new float[Nr*Nth*(Nph+1)];
    array_ini(Hr, (Nr+1)*Nth*Nph );
    array_ini(Hth, Nr*(Nth+1)*Nph);
    array_ini(Hph, Nr*Nth*(Nph+1));

    float *Er = new float[Nr*(Nth+1)*(Nph+1)];
    float *Eth = new float[(Nr+1)*Nth*(Nph+1)];
    float *Eph = new float[(Nr+1)*(Nth+1)*Nph];
    array_ini( Er, Nr*(Nth+1)*(Nph+1) );
    array_ini( Eth, (Nr+1)*Nth*(Nph+1) );
    array_ini( Eph, (Nr+1)*(Nth+1)*Nph );

    float *nDr = new float[Nr*(Nth+1)*(Nph+1)];
    float *nDth = new float[(Nr+1)*Nth*(Nph+1)];
    float *nDph = new float[(Nr+1)*(Nth+1)*Nph];
    array_ini( nDr, Nr*(Nth+1)*(Nph+1) );
    array_ini( nDth, (Nr+1)*Nth*(Nph+1) );
    array_ini( nDph, (Nr+1)*(Nth+1)*Nph );

    float *oDr = new float[Nr*(Nth+1)*(Nph+1)];
    float *oDth = new float[(Nr+1)*Nth*(Nph+1)];
    float *oDph = new float[(Nr+1)*(Nth+1)*Nph];
    array_ini( oDr, Nr*(Nth+1)*(Nph+1) );
    array_ini( oDth, (Nr+1)*Nth*(Nph+1) );
    array_ini( oDph, (Nr+1)*(Nth+1)*Nph );

    // device allocate //
    float *Hr_d, *Hth_d, *Hph_d;
    cudaMalloc( (void**)&Hr_d, sizeof(float)*(Nr+1)*Nth*Nph );
    cudaMalloc( (void**)&Hth_d, sizeof(float)*Nr*(Nth+1)*Nph );
    cudaMalloc( (void**)&Hph_d, sizeof(float)*Nr*Nth*(Nph+1) );

    float *Er_d, *Eth_d, *Eph_d;
    cudaMalloc( (void**)&Er_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) );
    cudaMalloc( (void**)&Eth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1) );
    cudaMalloc( (void**)&Eph_d, sizeof(float)*(Nr+1)*(Nth+1)*Nph );

    float *nDr_d, *nDth_d, *nDph_d;
    cudaMalloc( (void**)&nDr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) );
    cudaMalloc( (void**)&nDth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1) );
    cudaMalloc( (void**)&nDph_d, sizeof(float)*(Nr+1)*(Nth+1)*Nph );

    float *oDr_d, *oDth_d, *oDph_d;
    cudaMalloc( (void**)&oDr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) );
    cudaMalloc( (void**)&oDth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1) );
    cudaMalloc( (void**)&oDph_d, sizeof(float)*(Nr+1)*(Nth+1)*Nph );

    // E, D, H memory copy(H to D) //
    cudaMemcpy( Hr_d, Hr, sizeof(float)*(Nr+1)*Nth*Nph, cudaMemcpyHostToDevice );
    cudaMemcpy( Hth_d, Hth, sizeof(float)*Nr*(Nth+1)*Nph, cudaMemcpyHostToDevice );
    cudaMemcpy( Hph_d, Hph, sizeof(float)*Nr*Nth*(Nph+1), cudaMemcpyHostToDevice );

    cudaMemcpy( Er_d, Er, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( Eth_d, Eth, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( Eph_d, Eph, sizeof(float)*(Nr+1)*(Nth+1)*Nph, cudaMemcpyHostToDevice );

    cudaMemcpy( nDr_d, nDr, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( nDth_d, nDth, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( nDph_d, nDph, sizeof(float)*(Nr+1)*(Nth+1)*Nph, cudaMemcpyHostToDevice );

    cudaMemcpy( oDr_d, oDr, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( oDth_d, oDth, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( oDph_d, oDph, sizeof(float)*(Nr+1)*(Nth+1)*Nph, cudaMemcpyHostToDevice );
    
    // define grid, block size //
    int block_x = 16;
    int block_y = 16;
    dim3 Db( block_x, block_y, 1 );
    int grid_r = 128;
    int grid_th = 128;
    int grid_ph = 1024;
    dim3 Dg( grid_r/block_x, grid_th*grid_ph/block_y, 1 );

    int time_step = 1700;

    std::chrono::system_clock::time_point start
     = std::chrono::system_clock::now();
    
    for( int n = 0; n < time_step; n++ ){

        float t = float(n - 0.5)*Dt;

        // Add J //
        add_J <<<Dg, Db>>> ( delta_r, delta_th, delta_ph, Eth_d, 
                    i_s, j_s, k_s, t, t0, sigma_t, R0, Nr, Nth, Nph );

        // D update //
        D_update <<<Dg, Db>>> ( Nr, Nth, Nph, nDr_d, nDth_d, nDph_d,
                    oDr_d, oDth_d, oDph_d, Hr_d, Hth_d, Hph_d, 
                    delta_r, delta_th, delta_ph, Dt, THETA0, R0 );

        // E update //
        E_update <<<Dg, Db>>> ( Nr, Nth, Nph, Er_d, Eth_d, Eph_d, 
                    nDr_d, nDth_d, nDph_d, oDr_d, oDth_d, oDph_d, EPS0 );

        cudaDeviceSynchronize();

        H_update <<<Dg, Db>>> ( Nr, Nth, Nph, Er_d, Eth_d, Eph_d, 
                    Hr_d, Hth_d, Hph_d, delta_r, delta_th, delta_ph, 
                    Dt, THETA0, R0, MU0 );
        
        cudaMemcpy(Eth, Eth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyDeviceToHost );

        cudaDeviceSynchronize();

    }
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    
    double total_time;
    total_time = std::chrono::duration_cast <std::chrono::milliseconds>
    (end - start).count();

    std::cout << "elapsed_time : " << total_time*1.e-3 << "\n";

    cudaMemcpy( Er, Er_d, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyDeviceToHost);
    std::cout << "Source point : " << Er[idx_Er(i_s, j_s, k_s)] << "\n";
    std::cout << "Receive point : " << Er[idx_Er(i_r, j_r, k_r)] << "\n";

    cudaFree( Er_d );
    cudaFree( Eth_d );
    cudaFree( Eph_d );
    cudaFree( Hr_d );
    cudaFree( Hth_d );
    cudaFree( Hph_d );
    cudaFree( nDr_d );
    cudaFree( nDth_d );
    cudaFree( nDph_d );
    cudaFree( oDr_d );
    cudaFree( oDth_d );
    cudaFree( oDph_d );
}