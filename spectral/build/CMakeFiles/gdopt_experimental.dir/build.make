# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/linus/Projects/GDOPT/spectral

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/linus/Projects/GDOPT/spectral/build

# Include any dependencies generated for this target.
include CMakeFiles/gdopt_experimental.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/gdopt_experimental.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gdopt_experimental.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gdopt_experimental.dir/flags.make

CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o: CMakeFiles/gdopt_experimental.dir/flags.make
CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o: ../src/integrator.cpp
CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o: CMakeFiles/gdopt_experimental.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linus/Projects/GDOPT/spectral/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o -MF CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o.d -o CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o -c /home/linus/Projects/GDOPT/spectral/src/integrator.cpp

CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linus/Projects/GDOPT/spectral/src/integrator.cpp > CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.i

CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linus/Projects/GDOPT/spectral/src/integrator.cpp -o CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.s

CMakeFiles/gdopt_experimental.dir/src/main.cpp.o: CMakeFiles/gdopt_experimental.dir/flags.make
CMakeFiles/gdopt_experimental.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/gdopt_experimental.dir/src/main.cpp.o: CMakeFiles/gdopt_experimental.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linus/Projects/GDOPT/spectral/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/gdopt_experimental.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gdopt_experimental.dir/src/main.cpp.o -MF CMakeFiles/gdopt_experimental.dir/src/main.cpp.o.d -o CMakeFiles/gdopt_experimental.dir/src/main.cpp.o -c /home/linus/Projects/GDOPT/spectral/src/main.cpp

CMakeFiles/gdopt_experimental.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gdopt_experimental.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linus/Projects/GDOPT/spectral/src/main.cpp > CMakeFiles/gdopt_experimental.dir/src/main.cpp.i

CMakeFiles/gdopt_experimental.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gdopt_experimental.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linus/Projects/GDOPT/spectral/src/main.cpp -o CMakeFiles/gdopt_experimental.dir/src/main.cpp.s

# Object files for target gdopt_experimental
gdopt_experimental_OBJECTS = \
"CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o" \
"CMakeFiles/gdopt_experimental.dir/src/main.cpp.o"

# External object files for target gdopt_experimental
gdopt_experimental_EXTERNAL_OBJECTS =

gdopt_experimental: CMakeFiles/gdopt_experimental.dir/src/integrator.cpp.o
gdopt_experimental: CMakeFiles/gdopt_experimental.dir/src/main.cpp.o
gdopt_experimental: CMakeFiles/gdopt_experimental.dir/build.make
gdopt_experimental: CMakeFiles/gdopt_experimental.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/linus/Projects/GDOPT/spectral/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable gdopt_experimental"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gdopt_experimental.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gdopt_experimental.dir/build: gdopt_experimental
.PHONY : CMakeFiles/gdopt_experimental.dir/build

CMakeFiles/gdopt_experimental.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gdopt_experimental.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gdopt_experimental.dir/clean

CMakeFiles/gdopt_experimental.dir/depend:
	cd /home/linus/Projects/GDOPT/spectral/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/linus/Projects/GDOPT/spectral /home/linus/Projects/GDOPT/spectral /home/linus/Projects/GDOPT/spectral/build /home/linus/Projects/GDOPT/spectral/build /home/linus/Projects/GDOPT/spectral/build/CMakeFiles/gdopt_experimental.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gdopt_experimental.dir/depend
