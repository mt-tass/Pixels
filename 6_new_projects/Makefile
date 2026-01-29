# Specifying which compiler to use
CC = g++

# Defining the name of the object that will be created after using make
PROJECT = expecto_patronus

#defining source files
SRC = main.cpp $(wildcard src/*.cpp)

# Include directories and compiler flags
CFLAGS = -std=c++11 $(shell pkg-config --cflags opencv4) -Iinclude/

# Linker flags for OpenCV
LIBS = $(shell pkg-config --libs opencv4)

# Check if link is set
ifeq ($(SRC), $(link),)
build:
	$(error "SRC is not set")
else
build:
	@echo "Building..."
	@$(CC) $(SRC) $(link) -o $(PROJECT) $(CFLAGS) $(LIBS)
endif	

# if folder is not set, clean all build files all subfolders
.PHONY: clean
clean:
	@echo "Cleaning..."
	@rm -rf $(PROJECT)