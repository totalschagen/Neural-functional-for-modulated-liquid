# compiler and flags
cc = g++
cflags = -Wall  -O3 -std=c++17
parallel_flags = -Wall  -O3 -fopenmp -std=c++17
debug_flags = -Wall -g -O0 -std=c++17
ldflags = -lm -std=c++17 -lstdc++fs


# source files directory
src_dir_lin = src/Linear
src_dir_par = src/Parallel

# source files
src_files_lin = $(wildcard $(src_dir_lin)/*.cpp)
src_files_par = $(wildcard $(src_dir_par)/*.cpp)


# object files
obj_files_lin = $(src_files_lin:.cpp=.o)
obj_files_par = $(src_files_par:.cpp=.o)

# executable name
EXECUTABLE_lin = build/linear/linear
EXECUTABLE_par = build/parallel/parallel
EXECUTABLE = $(EXECUTABLE_lin) $(EXECUTABLE_par)

# default target
all: clean $(EXECUTABLE)

# target to build with debug flags
debug: cflags = $(debug_flags)
debug: ldflags = $(debug_flags)
debug: all
	@echo "Building with debug flags"



# rule to compile linear source file into object file
$(src_dir_lin)/%.o: $(src_dir_lin)/%.cpp
	$(cc) $(cflags) -c $< -o $@
	@echo "Compiled $< to $@"

# rule to compile parallel source file into object file
$(src_dir_par)/%.o: $(src_dir_par)/%.cpp
	$(cc) $(parallel_flags) -c $< -o $@
	@echo "Compiled $< to $@"

# rule to link object files into executable
$(EXECUTABLE_lin): $(obj_files_lin)
	$(cc) $(cflags) -o $@ $^ $(ldflags)
	@echo "Linked object files to create $@"

#rule to link object files into parallel executable
$(EXECUTABLE_par): $(obj_files_par)
	$(cc) $(parallel_flags) -o $@ $^ $(ldflags)
	@echo "Linked object files to create $@"






# rule to clean up object files and executable
clean:
	rm -f $(obj_files) $(EXECUTABLE)
	@echo "Cleaned up object files and executable"

# rule to clean up /Output folder
clean_output:
	rm -rf Output/*
	@echo "Cleaned up /Output folder"

# rule to clean up /Density_profiles folder
clean_density_profiles:
	rm -rf Density_profiles/*
	@echo "Cleaned up /Density_profiles folder"

# update clean rule to include cleaning the folders as well
clean_all: clean clean_output clean_density_profiles
	@echo "Cleaned up everything (object files, executable, and output folders)"