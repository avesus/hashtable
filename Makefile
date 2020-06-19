include common.mk


$(shell mkdir -p $(OBJ_DIR))

$(shell mkdir -p bin)


all:
	(make -C lib/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	(make -C main/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR) BDIR=$(BIN_DIR))

clean:
	(make clean -C lib/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	(make clean -C main/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	rm -rf bin $(OBJ_DIR)
	rm -f *~ *.o *#*
