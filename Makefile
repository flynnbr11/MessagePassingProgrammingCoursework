CC = mpicc
OBJ = image_reconstruction.c pgmio.c arralloc.c
TEST = test_code.c pgmio.c arralloc.c
LIB = -lm

image : $(OBJ)	
	$(CC) $(OBJ) $(LIB) -o $@
	

test_code : $(TEST)	
	$(CC) $(TEST) $(LIB) -o $@	
