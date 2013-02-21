all: installed

BUILD_DIR=$(shell pwd)/build
INCLUDE_DIR=$(shell pwd)/include

NAME=tclap
VERSION=1.2.1

checkedout:
	mkdir -p $(BUILD_DIR) $(INCLUDE_DIR)
	cd $(BUILD_DIR) &&\
	wget -O $(NAME)-$(VERSION).tar.gz http://sourceforge.net/projects/tclap/files/latest/download?source=files && tar xf $(NAME)-$(VERSION).tar.gz
	touch checkedout

installed: wiped checkedout
	ln -s $(BUILD_DIR)/$(NAME)-$(VERSION)/include/tclap/ $(INCLUDE_DIR)/$(NAME)
	touch installed

wiped: Makefile
	make wipe
	touch wiped

clean:
	rm -rf $(BUILD_DIR) $(INCLUDE_DIR)
	rm -f checkedout wiped installed

wipe: clean

