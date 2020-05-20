include ../common.mk

SRCS=src/*/*.c*

all: objs

objects:
	mkdir -p $(ODIR)

objs:	objects
	$(CC) $(CFLAGS) $(SRCS) -c $(LDFLAGS)
	mv *.o $(ODIR)


clean:
	rm -f *.o *~ *#*
	rm -rf $(ODIR)
	rm -f */.o */*~ */*#*
	rm -f src/*/*~ src/*/*#* src/*/*.o
	rm -f include/*/*~ include/*/*#* include/*/*.o
