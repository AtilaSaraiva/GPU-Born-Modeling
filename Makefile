host=$(shell hostname)
ifeq ($(host),JurosComposto)
    LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -ltirpc -lfftw3f -lfftw3 -O3
endif
ifeq ($(host),marreca)
    LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -lfftw3f -lfftw3 -O3
endif

CULIBS= -L /opt/cuda/lib -I /opt/cuda/include -lcudart -lcuda -lstdc++ -lcufft

ODIR = ../../library
IDIR = ../../include

#SOURCE = $(wildcard $(ODIR)/*.cu)
#OBJ = $(SOURCE:.cu=.o)

_OBJ = born.o snap.o io.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

DEPS = $(wildcard $(IDIR)/*.cuh)

CFLAGS = -I$(IDIR) -arch=sm_30

PROG = mod

dFold=testData
data=seismicData.rsf
OD=directwave.rsf
comOD=seismicDataWithDirectWave.rsf
vel=vel.rsf
ref=ref.rsf

$(PROG): main.o $(OBJ)
	nvcc main.o $(OBJ) $(CFLAGS) $(LDFLAGS) -o $@

main.o: main.cu $(DEPS)
	nvcc -x cu $(CFLAGS) $(LDFLAGS) -o $@ -dc $<

$(ODIR)/%.o: $(ODIR)/%.cu $(DEPS)
	nvcc -x cu $(CFLAGS) $(LDFLAGS) -o $@ -dc $<



run: mod
	#./mod nr=400 nshots=2 incShots=100 isrc=0 jsrc=200 gxbeg=0 ref=$(dFold)/$(ref) vel=$(dFold)/$(vel) data=$(dFold)/$(data) OD=$(dFold)/$(OD) comOD=$(dFold)/$(comOD)
	./mod nr=368 nshots=3 incShots=100 incRec=0 isrc=0 jsrc=100 gxbeg=0 ref=$(dFold)/$(ref) vel=$(dFold)/$(vel) data=$(dFold)/$(data) OD=$(dFold)/$(OD) comOD=$(dFold)/$(comOD)
	#sfimage <$(dFold)/$(data)
	sfgrey <$(dFold)/$(data) >seismicData.vpl
	#ximage n1=920 <snap/snap_u3_s0_1290_920_1120
	#ximage n1=780 <test_kernel_add_sourceArray &
	#ximage n1=780 <snap/snap_u3_s0_0_780_980 &
	#ximage n1=780 <snap/snap_u3_s1_0_780_980 &

profile: mod
	nvprof ./mod nr=400 nshots=2 incShots=100 isrc=0 jsrc=200 gxbeg=0 vel=$(dFold)/$(vel) data=$(dFold)/$(data) OD=$(dFold)/$(OD) comOD=$(dFold)/$(comOD)

PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(PROG) *.o
