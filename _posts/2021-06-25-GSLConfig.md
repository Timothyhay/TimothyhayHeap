---
layout: blogpage
title: Using GSL on Windows
comments: true
tags: Note Skill
---

This is a tutorial about how to install and use GSL on Windows in the most simplest way(After I tried so many approaches).


## 1. Download MSYS2 ##

Jump to their [official website](https://www.msys2.org/) to downlad MSYS2. And install a MinGW-w64 at the same time. Since MinGW is no longer supported(And its developer has a bad temper), MinGW-w64 and MSYS2 will be a better choice to complie a x64 application.

After installing, please first `cd` to your MSYS2 directory, then run:


    ./msys2.exe

    
## 2. Install GSL and MinGW-w64 ##

MSYS2 is a light GNU environment, an extension of MSYS, a upgraded version of Cygwin integrated with MinGW-w64 and pacman.

If you are not sure have you installed a MinGW-w64 or not, continue to run this on the MSYS2 bash terminal:

    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make mingw-w64-x86_64-gdb mingw-w64-x86_64-binutils


Then install the GSL:

    pacman -S mingw-w64-x86_64-gsl

## 3. Config ToolChain and CMakeList ##

Here I am using the JetBrain's C/C++ IDE, CLion. 

Open the master branch of the Source-LDA project, and generate a CMakeList with the `add_executable` for files in `src` folder, like:

    add_executable(Source_LDA_master
            src/conceptlda.cpp
            src/conceptlda.h
            src/conceptlda_parallel.cpp
            src/conceptlda_parallel.h
            src/eda.cpp
            src/eda.h
            src/eda_parallel.cpp
            src/eda_parallel.h
            src/gtpoints.cpp
            src/gtpoints.h
            src/lda.cpp
            src/lda.h
            src/main.cpp
            src/srclda.cpp
            src/srclda.h
            src/srclda_parallel.cpp
            src/srclda_parallel.h
            src/utility.cpp
            src/utility.h)

 Make sure to set your default toolchain to the particular one with GSL on [Settings > Build, Execution and Deployment > Toolchain].

For those who manually download and build the GSL in another directory, remember to specify the library(GSL) in the CMakeList by adding the following lines:

    include_directories(<your INCLUDE FILES for GSL>\include)
    link_directories(your LIB FILES for GSL\lib)
    link_libraries(gsl)

These lines above should be added brfore 'add_executable' to avoid unreference error.

Finally add

    target_link_libraries(Source_LDA_master gsl)

to your CMakeList and save it. Now the project should be runnable.


