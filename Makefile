LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -ltirpc -lfftw3f -lfftw3 -O3

CULIBS= -L /opt/cuda/lib -I /opt/cuda/include -lcudart -lcuda -lstdc++ -lcufft

mod: cuwaveprop2d-modified.cu
	nvcc cuwaveprop2d-modified.cu $(LDFLAGS) -o mod

run: mod
	./mod nr=400 isrc=0 jsrc=350 gxbeg=150 vel=vel.rsf data=seismicData.rsf OD=directWave.rsf comOD=seismicDataWithDirectWave.rsf

