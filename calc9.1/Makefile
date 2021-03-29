
CC	=	gcc
FCC =   gfortran
CFLAGS	=	-W -Wall -g -fPIC
FFLAGS  =   -W -Wall -g -O2 -fPIC -fno-second-underscore

srcs = $(wildcard *.c)
hdrs = $(wildcard *.h) $(wildcard *.H)

fsrcs = $(wildcard *.f)
fhdrs = $(wildcard *.i) $(wildcard *.INC)

objs = $(srcs:.c=.o) 
fobjs = $(fsrcs:.f=.o) 

target	=	libcalc_cwrapper.so
		  
$(target): $(objs) $(fobjs)
	$(CC) -o $(target) $(objs) $(fobjs) -lm -lgfortran -shared

$(objs): %.o: %.c $(hdrs)
	$(CC) -c $(CFLAGS) $< -o $@

$(fobjs): %.o: %.f $(fhdrs)
	$(FCC) -c $(FFLAGS) $< -o $@

.PHONY:	clean
clean: 
	rm -f $(objs) $(fobjs) $(target)
