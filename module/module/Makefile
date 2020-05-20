include ../common.mk

MAIN_OBJ=create_constants.o
MAIN_SRC=create_constants.c
MAIN_EXE=create_constants

all: create_constants

create_constants:
	$(CC) $(CFLAGS) $(MAIN_SRC) -c $(LDFLAGS)
	$(CC) $(CFLAGS) $(MAIN_OBJ) $(ODIR)/*.o -o $(MAIN_EXE) $(LDFLAGS)
	./create_constants -p $(COMPILE_CONF_FILE)

clean:
	rm -rf *.o *~ *#*
	rm -rf $(MAIN_EXE)
