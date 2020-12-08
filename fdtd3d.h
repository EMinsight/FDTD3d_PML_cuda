#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <stdio.h>

#define PI (3.14159265258979)
#define C0 (3.0e8)
#define MU0 (4.0*PI*1.0e-7)
#define EPS0 (1.0/MU0/C0/C0)
#define R0 (6370e3)
#define THETA0 (PI*0.5 - std::atan(50e3/R0))
#define E_Q (1.6e-19)
#define E_M (9.11e-31)

//#define GRID_SIZE ( 256 )

extern const int Nr;
extern const int Nth;
extern const int Nph;

extern const int nElem;

extern const float delta_r;
extern const float delta_th;
extern const float delta_ph;
extern const float Dt;
extern const float inv_Dt;

void array_ini(
    float *array, int size
);

__global__ void add_J( float del_r, float del_th, float del_ph, float *Eth,
                    int i_s, int j_s, int k_s, float t, float t0, float sig, float r0, int nr, int Nth, int Nph );

__global__ void D_update( int nr, int nth, int nph, float *nDr, float *nDth, float *nDph,
                    float *oDr, float *oDth, float *oDph, float *Hr, float *Hth, float *Hph,
                    float del_r, float del_th, float del_ph, float dt, float theta0, float r0 );

__global__ void E_update( int nr, int nth, int nph, float *Er, float *Eth, float *Eph,
                        float *nDr, float *nDth, float *nDph, float *oDr, float *oDth, float *oDph, float eps);

__global__ void H_update( int nr, int nth, int nph, float *Er, float *eth, float *Eph,
                        float *Hr, float *Hth, float *Hph, float del_r, float del_th, float del_ph,
                        float dt, float th0, float r0, float mu );

inline float dist(float i){return R0 + i*delta_r;};
inline float th(float j){return THETA0 + j*delta_th;};
inline float ph(float k){return k*delta_ph;};

inline int idx_Er( int i, int j, int k ){ return i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k; }
inline int idx_Eth( int i, int j, int k ){ return i*(Nth*(Nph+1)) + j*(Nph+1) + k; }
inline int idx_Eph( int i, int j, int k ){ return i*((Nth+1)*Nph) + j*Nph + k; }
inline int idx_Hr( int i, int j, int k ){ return i*(Nth*Nph) + j*Nph + k; }
inline int idx_Hth( int i, int j, int k ){ return i*((Nth+1)*Nph) + j*Nph + k; }
inline int idx_Hph( int i, int j, int k ){ return i*(Nth*(Nph+1)) + j*(Nph+1) + k; }
