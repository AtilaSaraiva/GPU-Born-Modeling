# Synthetic Seismic Data using acoustic wave propagation written in CUDA C++

Simple implementation of the acoustic wave propagation in a 2D model using finite-differences in the
time-domain and optimized in CUDA.

To compile the code just run:

```
make
```

and to perform a test run on a simple two-layer velocity model run:

```
make run
```

the result will be called seismicData.rsf. You can use:

```
sfimage <seismicData.rsf
```
to visualize it.


This code uses the [Madagascar API][http://www.ahay.org/wiki/Main_Page] to read and write files so
it is required to have it installed and on your $PATH for this code to compile and run. If you want
to run a SCons script using this code on the Marmousi velocity field, run:

```
cd madagascarBuild
scons
```
