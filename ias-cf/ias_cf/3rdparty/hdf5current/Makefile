all: installed

KERNEL_NAME = $(shell uname -m)
ifeq (x86_64,$(KERNEL_NAME))
	KERNEL_STR = linux-x86_64
else
	KERNEL_STR = linux
endif

BUILD_DIR=$(shell pwd)/build
INCLUDE_DIR=$(shell pwd)/include
LIB_DIR=$(shell pwd)/lib

NAME=hdf5
VERSION=1.8.10

checkedout:
	mkdir -p $(BUILD_DIR) $(INCLUDE_DIR) $(LIB_DIR)
	cd $(BUILD_DIR) &&\
	wget -O $(NAME)-$(VERSION)-$(KERNEL_STR)-shared.tar.gz http://www.hdfgroup.org/ftp/HDF5/releases/${NAME}-${VERSION}/bin/$(KERNEL_STR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared.tar.gz && tar xf $(NAME)-$(VERSION)-$(KERNEL_STR)-shared.tar.gz
	touch checkedout

installed: wiped checkedout
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared/include/ $(INCLUDE_DIR)/$(NAME)
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared/lib/lib$(NAME).so $(LIB_DIR)/lib$(NAME).so
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared/lib/lib$(NAME).so $(LIB_DIR)/lib$(NAME).so.7
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared/lib/lib$(NAME)_hl.so $(LIB_DIR)/lib$(NAME)_hl.so
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared/lib/lib$(NAME)_hl.so $(LIB_DIR)/lib$(NAME)_hl.so.7
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared/lib/libsz.so.2 $(LIB_DIR)/libsz.so.2
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)-$(KERNEL_STR)-shared/lib/libsz.so.2.0.0 $(LIB_DIR)/libsz.so.2.0.0 
	touch installed

wiped: Makefile
	make wipe
	touch wiped

clean:
	rm -rf $(BUILD_DIR) $(INCLUDE_DIR) $(LIB_DIR)
	rm -f checkedout wiped installed

wipe: clean

