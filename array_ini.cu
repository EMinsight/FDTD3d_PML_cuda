#include "fdtd3d.h"

void array_ini( float *array, int size )
{
    for( int i = 0; i < size; i++ )array[i] = float(0);
}