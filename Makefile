LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -ltirpc -lfftw3f -lfftw3 -O3

CULIBS= -L /opt/cuda/lib -I /opt/cuda/include -lcudart -lcuda -lstdc++ -lcufft

dFold=testData
data=seismicData.rsf
OD=directwave.rsf
comOD=seismicDataWithDirectWave.rsf
vel=vel.rsf


mod: main.cu
	nvcc main.cu $(LDFLAGS) -o mod

run: mod
	./mod nr=400 isrc=0 jsrc=350 gxbeg=150 vel=$(dFold)/$(vel) data=$(dFold)/$(data) OD=$(dFold)/$(OD) comOD=$(dFold)/$(comOD)

