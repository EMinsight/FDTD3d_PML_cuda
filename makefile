OBJS = add_J.o array_ini.o calc_fdtd.o main.o D_update.o E_update.o H_update.o
HEADERS = fdtd3d.h main.h
OPTS = -O3

main: $(OBJS)
	nvcc -o $@ $(OBJS)
%.o: %.cu $(HEADERS)
	nvcc -c $< $(OPTS)