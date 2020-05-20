include common.mk


$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p bin)


all:

ifneq ($(WITH_CONSTANTS), 0)
	rm -f $(COMPILE_CONF_FILE)
	(make -C lib/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	(make -C module/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR) OFILE=$(COMPILE_CONF_FILE))
endif
	(make -C lib/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
ifneq ($(WITH_TESTS), 0)
	(make -C test/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR) BDIR=$(BIN_DIR))
endif
	(make -C main/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR) BDIR=$(BIN_DIR))

clean:
	(make clean -C lib/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	(make clean -C main/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	(make clean -C test/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	(make clean -C module/ BASE_DIR=${CURDIR} ODIR=$(OBJ_DIR))
	rm -rf bin $(OBJ_DIR)
	rm -f *~ *.o *#*
