/*
Hello world of wave propagation in CUDA. FDTD acoustic wave propagation in homogeneous medium. Second order accurate in time and eigth in space.

Oleg Ovcharenko
Vladimir Kazei, 2019

oleg.ovcharenko@kaust.edu.sa
vladimir.kazei@kaust.edu.sa
*/

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

#include "btree.cuh"

using namespace std;


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
    sf_file Freflectivity=NULL;
    Freflectivity = sf_input("ref");

    // Getting command line parameters
    geometry param = getParameters(Fvel);

    // Allocate memory for velocity model
    velocity h_model = getVelFields (Fvel, Freflectivity, param);

    cerr<<"vp = "<<h_model.maxVel<<endl;
    cerr<<"param.taperBorder = "<<param.taperBorder<<endl;

    // Taper mask
    float *h_tapermask = tapermask(param);

    // Time stepping
    source h_wavelet = fillSrc(param, h_model);

    // Data
    seismicData h_seisData = allocHostSeisData(param, h_wavelet.timeSamplesNt);

    // Set Output files
    int dimensions[3] = {h_wavelet.timeSamplesNt,param.nReceptors,param.nShots};
    float spacings[3] = {h_wavelet.timeStep,param.modelDx,10};
    int origins[3] = {0,0,0};
    sf_file Fdata_directWave = createFile3D("comOD",dimensions,spacings,origins);
    sf_file Fonly_directWave = createFile3D("OD",dimensions,spacings,origins);
    sf_file Fdata = createFile3D("data",dimensions,spacings,origins);


    sf_putint(Fdata,"incShots",param.incShots);
    sf_putint(Fdata,"incRec",param.incRec);
    sf_putint(Fdata,"gxbeg",param.firstReceptorPos);
    sf_putint(Fdata,"sxbeg",param.srcPosX);
    sf_putint(Fdata,"sybeg",param.srcPosY);

    test_getParameters(param, h_wavelet);


    // ===================MODELING======================
    born(param, h_model, h_wavelet, h_tapermask, h_seisData, Fonly_directWave, Fdata_directWave, Fdata, false);
    // =================================================

    printf("Clean memory...");
    delete[] h_model.velField;
    delete[] h_model.extVelField;
    delete[] h_model.firstLayerVelField;
    delete[] h_seisData.seismogram;
    delete[] h_seisData.directWaveOnly;
    delete[] h_tapermask;
    //delete[] h_time;
    delete[] h_wavelet.timeSeries;


    return 0;
}
