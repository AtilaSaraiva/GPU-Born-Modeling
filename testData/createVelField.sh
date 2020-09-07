sfspike n1=500 n2=700 d1=10 d2=10 mag=3300 > vel1.rsf
sfspike n1=500 n2=700 d1=10 d2=10 mag=1000 k1=250 l1=500 > vel2.rsf
sfadd mode=a <vel1.rsf vel2.rsf >vel3.rsf
