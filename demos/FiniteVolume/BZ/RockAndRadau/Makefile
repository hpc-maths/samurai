CC = gcc -fPIC
FC = gfortran -fPIC
# -fno-underscoring
LD = gcc

CCFLAGS =
LDFLAGS =

OBJS = dc_decsol.o  decsol.o radau5.o rock4.o integration_stiff.o
LIB = libintegration_stiff.so

all: $(LIB)

$(LIB): $(OBJS)
	$(LD) -shared -o $@ $^ -lgfortran

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

%.o : %.f90
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	rm -f $(LIB) $(OBJS)
