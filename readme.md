# Genetic Algorithm to Solve Scheduling Problem

## The Problem

1. A `job` (or `Task`) is sent to an Access Point;
2. The access point **schedules** the job among possible SEO satellites.


# Building the Project

## Requirements

- python
- pybind11
- CMake

If you're running inside a docker container:

```shell
sudo apt-get update
sudo apt-get install -y python3 cmake
sudo apt install python3-pybind11
```

## Build

1. Make a `build` directory and move inside it:

```
mkdir build && cd build
```

2. Build the python module and copy it on the project directory:

```
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
cmake --build .
cp scheduler*.so ..
```

The `scheduler*.so` file is the module that can be used by Python.