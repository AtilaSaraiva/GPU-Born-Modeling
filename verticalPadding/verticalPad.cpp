#include <rsf.hh>
#include<iostream>

using namespace std;

float* padVelField(int ny, int nx, int padding, float *velField)
{
    int nyPlusPadding = ny + padding;
    size_t totalElem = nyPlusPadding * nx;
    float *velPadded = new float[totalElem];
    memset(velPadded,0,totalElem * sizeof(float));
    int i;
    for (int j = 0; j < nx; j++){
        for (i = 0; i < padding; i++){
            velPadded[j * nyPlusPadding + i] = velField[j * ny + 0];
        }
        for (i = padding; i < ny + padding; i++){
            velPadded[j * nyPlusPadding + i]  = velField[j * ny + (i - padding)];
        }
    }
    return velPadded;
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


int main(int argc, char* argv[])
{
    sf_init(argc,argv);
    sf_file Fvel=NULL;
    Fvel = sf_input("in");
    int padding; sf_getint("padding", &padding);
    int nx,ny;
    sf_histint(Fvel, "n1", &ny);
    sf_histint(Fvel, "n2", &nx);
    float *velField = new float[ny * nx];
    sf_floatread(velField, ny * nx, Fvel);

    int dimensions[3] = {ny + padding,nx,1};
    float spacings[3] = {1,1,1};
    int origins[3] = {0,0,0};

    float *velPadded = padVelField(ny, nx, padding, velField);
    sf_file FpaddedField = createFile3D("out",dimensions,spacings,origins);
    sf_floatwrite(velPadded, (ny + padding) * nx, FpaddedField);

    delete[] velField;
}
