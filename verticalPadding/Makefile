LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -ltirpc -lfftw3f -lfftw3 -O3


pad: verticalPad.cpp
	g++ verticalPad.cpp $(LDFLAGS) -o pad

run: pad
	./pad padding=50 <../testData/vel.rsf >velPadded.rsf
	sfimage padding=50 <velPadded.rsf
