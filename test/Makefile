include ../common.mk

MAIN_OBJ=tests.o
MAIN_SRC=tests.c
MAIN_EXE=tests

all: bins

binaries:
	mkdir -p $(BDIR)

bins: binaries
	$(CC) $(CFLAGS) $(MAIN_SRC) -c $(LDFLAGS)
	$(CC) $(CFLAGS) $(MAIN_OBJ) $(ODIR)/*.o -o $(BDIR)/$(MAIN_EXE) $(LDFLAGS)

clean:
	rm -rf *.o *~ *#*
	rm -rf $(BDIR)
