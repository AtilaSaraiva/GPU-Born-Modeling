sfspike n1=425 n2=368 d1=8 d2=25 mag=3300 > vel1.rsf
sfspike n1=425 n2=368 d1=8 d2=25 mag=1000 k1=250 l1=425 > vel2.rsf
sfadd mode=a <vel1.rsf vel2.rsf >vel.rsf
