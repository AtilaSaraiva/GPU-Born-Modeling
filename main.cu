/*
Hello world of wave propagation in CUDA. FDTD acoustic wave propagation in homogeneous medium. Second order accurate in time and eigth in space.

Oleg Ovcharenko
Vladimir Kazei, 2019

oleg.ovcharenko@kaust.edu.sa
vladimir.kazei@kaust.edu.sa
*/

#include <rsf.hh>
#include <iostream>
#include <string>
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
/*
Add this to c_cpp_properties.json if linting isn't working for CUDA libraries
"includePath": [
                "/usr/local/cuda-10.0/targets/x86_64-linux/include",
                "${workspaceFolder}/**"
            ],
*/

#include "cuwaveprop2d.cu"

using namespace std;

void modeling(int nx, int ny, int nb, int nr, int nt, int gxbeg, int gxend, int isrc, int jsrc, float dx, float dy, float dt, float *h_vpe, float *h_dvpe, float *h_tapermask, float *h_data, float *h_directwave, float * h_wavelet, bool snaps, int nshots, int incShots, sf_file Fonly_directWave, sf_file Fdata_directWave, sf_file Fdata);

void dummyVelField(int nxb, int nyb, int nb, float *h_vpe, float *h_dvpe)
{
    for (int i = 0; i < nyb; i++){
        for (int j = 0; j < nxb; j++){
            h_dvpe[j * nyb + i]  = h_vpe[j * nyb + nb];
        }
    }
}

void expand(int nb, int nyb, int nxb, int nz, int nx, float *a, float *b)
/*< expand domain of 'a' to 'b':  a, size=nz*nx; b, size=nyb*nxb;  >*/
{
    int iz,ix;
    for     (ix=0;ix<nx;ix++) {
        for (iz=0;iz<nz;iz++) {
            b[(nb+ix)*nyb+(nb+iz)] = a[ix*nz+iz];
        }
    }
    for     (ix=0; ix<nxb; ix++) {
        for (iz=0; iz<nb; iz++)         b[ix*nyb+iz] = b[ix*nyb+nb];//top
        for (iz=nz+nb; iz<nyb; iz++) b[ix*nyb+iz] = b[ix*nyb+nb+nz-1];//bottom
    }
    for (iz=0; iz<nyb; iz++){
        for(ix=0; ix<nb; ix++)  b[ix*nyb+iz] = b[nb*nyb+iz];//left
        for(ix=nb+nx; ix<nxb; ix++)     b[ix*nyb+iz] = b[(nb+nx-1)*nyb+iz];//right
    }
}

void abc_coef (int nb, float *abc)
{
    for(int i=0; i<nb; i++){
        abc[i] = exp (-pow(0.001 * (nb - i + 1),2.0));
    }
}

void taper (int nx, int ny, int nb, float *abc, float *campo)
{
    int nxb = nx + 2 * nb;
    int nyb = ny + 2 * nb;
    for(int j=0; j<nxb; j++){
        for(int i=0; i<nb; i++){
            campo[j * nyb + i] *= abc[i];
            campo[j * nyb + (nb + ny + i)] *= abc[nb - i - 1];
        }
    }
    for(int i=0; i<nyb; i++){
        for(int j=0; j<nb; j++){
            campo[j * nyb + i] *= abc[j];
            campo[(nb + nx + j) * nyb + i] *= abc[nb - j - 1];
        }
    }
}

sf_file createFile3D (const char *name, int dimensions[3], float spacings[3], int origins[3])
{
    sf_file Fdata = NULL;
    Fdata = sf_output(name);
    char key_n[6],key_d[6],key_o[6];
    for (int i = 0; i < 3; i++){
        sprintf(key_n,"n%i",i+1);
        sprintf(key_d,"d%i",i+1);
        sprintf(key_o,"o%i",i+1);
        sf_putint(Fdata,key_n,dimensions[i]);
        sf_putint(Fdata,key_d,spacings[i]);
        sf_putint(Fdata,key_o,origins[i]);
    }

    return Fdata;
}

typedef struct{
    int nShots;
    int srcPosX;
    int srcPosY;
    int firstReceptorPos;
    int nReceptors;
    int lastReceptorPos;
    int incShots;
    int modelNx;
    int modelNy;
    int modelNxBorder;
    int modelNyBorder;
    int modelDx;
    int modelDy;
    int taperBorder;
    // Auxiliaries
    size_t nxy;
    size_t nbxy;
    size_t nbytes;
} geometry;

geometry getParameters(sf_file FvelModel)
{
    geometry param;
    sf_getint("nr",&param.nReceptors);
    sf_getint("isrc",&param.srcPosY);
    sf_getint("jsrc",&param.srcPosX);
    sf_getint("gxbeg",&param.firstReceptorPos);
    sf_getint("nshots",&param.nShots);
    sf_getint("incShots",&param.incShots);
    sf_histint(FvelModel, "n1",&param.modelNy);
    sf_histint(FvelModel, "n2", &param.modelNx);
    sf_histint(FvelModel, "d1",&param.modelDy);
    sf_histint(FvelModel, "d2", &param.modelDx);
    param.lastReceptorPos = param.firstReceptorPos + param.nReceptors;
    param.taperBorder = 0.2 * param.modelNx;
    param.nxy = param.modelNx * param.modelNy;
    param.modelNxBorder = param.modelNx + 2 * param.taperBorder;
    param.modelNyBorder = param.modelNy + 2 * param.taperBorder;
    param.nbxy = param.modelNxBorder * param.modelNyBorder;
    param.nbytes = param.nbxy * sizeof(float); // bytes to store modelNxBorder * modelNyBorder
    return param;
}

void test_getParameters (geometry param)
{
    cerr<<"param.incShots: "<<param.incShots<<endl;
    cerr<<"param.modelDims[0] "<<param.modelNx<<param.modelNy<<endl;
    cerr<<"param.nShots "<<param.nShots<<endl;
    cerr<<"param.nReceptors "<<param.nReceptors<<endl;
    cerr<<"param.firstReceptorPos "<<param.firstReceptorPos<<endl;
    cerr<<"param.lastReceptorPos "<<param.lastReceptorPos<<endl;
}

typedef struct{
    float *velField;
    float *extVelField;
    float *firstLayerVelField;
    float maxVel;
} velocity;

velocity getVelFields(sf_file FvelModel, geometry param)
{
    velocity h_model;

    h_model.velField = new float[param.nxy];
    sf_floatread(h_model.velField, param.nxy, FvelModel);

    h_model.extVelField = new float[param.nbxy];
    memset(h_model.extVelField,0,param.nbytes);
    expand(param.taperBorder, param.modelNyBorder, param.modelNxBorder, param.modelNy, param.modelNx, h_model.velField, h_model.extVelField);

    h_model.maxVel = h_model.velField[0];
    for(int i=1; i < param.nxy; i++){
        if(h_model.velField[i] > h_model.maxVel){
            h_model.maxVel = h_model.velField[i];
        }
    }

    h_model.firstLayerVelField = new float[param.nbxy];
    dummyVelField(param.modelNxBorder, param.modelNyBorder, param.taperBorder, h_model.extVelField, h_model.firstLayerVelField);

    return h_model;
}

/*
===================================================================================
MAIN
===================================================================================
*/
int main(int argc, char *argv[])
{
    /* Main program that reads and writes data and read input variables */
    bool verb;
    sf_init(argc,argv); // init RSF
    if(! sf_getbool("verb",&verb)) verb=0;

    // Setting up I/O files
    sf_file Fvel=NULL;
    Fvel = sf_input("vel");

    // Getting command line parameters
    geometry param = getParameters(Fvel);

    // Allocate memory for velocity model
    velocity h_model = getVelFields (Fvel, param);

    printf("MODEL:\n");
    printf("\t%i x %i\t:param.modelNy x param.modelNx\n", param.modelNy, param.modelNx);
    printf("\t%f\t:param.modelDx\n", param.modelDx);
    printf("\t%f\t:h_model.velField[0]\n", h_model.velField[0]);

    cerr<<"vp = "<<h_model.maxVel<<endl;
    cerr<<"param.taperBorder = "<<param.taperBorder<<endl;

    // Taper mask
    float *h_abc = new float[param.taperBorder];
    float *h_tapermask = new float[nbxy];
    for(int i=0; i < nbxy; i++){
        h_tapermask[i] = 1;
    }
    abc_coef(param.taperBorder, h_abc);
    taper(param.modelNx, param.modelNy, param.taperBorder, h_abc, h_tapermask);


    // Time stepping
    float t_total = 2.5;               /* total time of wave propagation, sec */
    float dt = 0.5 * param.modelDx / h_model.maxVel;         /* time step assuming constant vp, sec */
    int nt = round(t_total / dt);      /* number of time steps */
    int snap_step = round(0.1 * nt);   /* save snapshot every ... steps */

    printf("TIME STEPPING:\n");
    printf("\t%e\t:t_total\n", t_total);
    printf("\t%e\t:dt\n", dt);
    printf("\t%i\t:nt\n", nt);

    // Data
    float *h_data = new float[param.nReceptors * nt];
    float *h_directwave = new float[param.nReceptors * nt];

    // Source
    float f0 = 10.0;                    /* source dominant frequency, Hz */
    float t0 = 1.2 / f0;                /* source padding to move wavelet from left of zero */

    float *h_wavelet, *h_time;
    float tbytes = nt * sizeof(float);
    h_time = (float *)malloc(tbytes);
    h_wavelet = (float *)malloc(tbytes);

    // Fill source waveform vector
    float a = PI * PI * f0 * f0;            /* const for wavelet */
    float dt2dx2 = (dt * dt) / (param.modelDx * param.modelDx);   /* const for fd stencil */
    for (int it = 0; it < nt; it++)
    {
        h_time[it] = it * dt;
        // Ricker wavelet (Mexican hat), second derivative of Gaussian
        h_wavelet[it] = 1e10 * (1.0 - 2.0 * a * pow(h_time[it] - t0, 2)) * exp(-a * pow(h_time[it] - t0, 2));
        h_wavelet[it] *= dt2dx2;
    }

    printf("SOURCE:\n");
    printf("\t%f\t:f0\n", f0);
    printf("\t%f\t:t0\n", t0);
    printf("\t%i\t:param.srcPosY - ox\n", param.srcPosY);
    printf("\t%i\t:param.srcPosX - oy\n", param.srcPosX);
    printf("\t%e\t:dt2dx2\n", dt2dx2);
    printf("\t%f\t:min wavelength [m]\n",(float)h_model.maxVel / (2*f0));
    printf("\t%f\t:ppw\n",(float)h_model.maxVel / (2*f0) / param.modelDx);

    // Set Output files

    int dimensions[3] = {nt,param.nReceptors,param.nShots};
    float spacings[3] = {1,1,1};
    int origins[3] = {0,0,0};
    sf_file Fdata_directWave = createFile3D("comOD",dimensions,spacings,origins);
    sf_file Fonly_directWave = createFile3D("OD",dimensions,spacings,origins);
    sf_file Fdata = createFile3D("data",dimensions,spacings,origins);

    // ===================MODELING======================
    modeling(param.modelNx, param.modelNy, param.taperBorder, param.nReceptors, nt, param.firstReceptorPos, param.lastReceptorPos, param.srcPosY, param.srcPosX, param.modelDx, param.modelDy, dt, h_model.extVelField, h_model.firstLayerVelField, h_tapermask, h_data, h_directwave,  h_wavelet, false, param.nShots, param.incShots, Fonly_directWave, Fdata_directWave, Fdata);
    // =================================================


    printf("Clean memory...");
    delete[] h_model.velField;
    delete[] h_model.extVelField;
    delete[] h_model.firstLayerVelField;
    delete[] h_data;
    delete[] h_directwave;
    delete[] h_abc;
    delete[] h_tapermask;
    delete[] h_time;
    delete[] h_wavelet;


    return 0;
}
