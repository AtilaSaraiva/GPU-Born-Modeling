LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -ltirpc -lfftw3f -lfftw3 -O3

CULIBS= -L /opt/cuda/lib -I /opt/cuda/include -lcudart -lcuda -lstdc++ -lcufft

dFold=testData
data=seismicData.rsf
OD=directwave.rsf
comOD=seismicDataWithDirectWave.rsf
vel=vel.rsf


mod: main.cu cuwaveprop2d.cu cudaKernels.cu
	nvcc main.cu $(LDFLAGS) -o mod

run: mod
	./mod nr=400 nshots=2 incShots=100 isrc=0 jsrc=200 gxbeg=0 vel=$(dFold)/$(vel) data=$(dFold)/$(data) OD=$(dFold)/$(OD) comOD=$(dFold)/$(comOD)
	#sfimage <$(dFold)/$(data)
	sfgrey <$(dFold)/$(data) | sfpen &
	#ximage n1=780 <snap/snap_u3_s0_0_780_980 &
	#ximage n1=780 <snap/snap_u3_s1_0_780_980 &

profile: mod
	nvprof ./mod nr=400 nshots=2 incShots=100 isrc=0 jsrc=200 gxbeg=0 vel=$(dFold)/$(vel) data=$(dFold)/$(data) OD=$(dFold)/$(OD) comOD=$(dFold)/$(comOD)
