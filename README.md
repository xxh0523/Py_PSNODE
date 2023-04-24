# Py_PSNODE
General neural ODE and DAE modules for power system dynamic modeling. 

The PyTorch-based ODE solver is developed based on [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

Samples are generated using [Py_PSOPS](https://github.com/xxh0523/Py_PSOPS).

# Environment
-[Windows 7, 8, 10]

-[Linux]

-[Python 3.6, 3.7, 3.8]

# Initialization
1.  Clone / Pull the codes.

2.  Try building neural dynamic models for power system dynamic components such as generator unit, loads, stations, distribution networks, regulators, etc. 

# Directory
## neural_00_ODE_01_no_encode.py
Regular neural ODE module with external inputs.

## neural_00_ODE_02_direct_encode.py
Autoencoder-based neural ODE module with external inputs.

## neural_00_DAE_01_no_encode.py
Regular neural DAE module.

## neural_00_DAE_02_direct_encode.py
Autoencoder-based neural DAE module.

## utils
Logger class.

# Data Files

A sample data file can be downloaded at Baidu web storage:

URL: https://pan.baidu.com/s/1P24h0AQqNS9SAwG1K4wW6g

Code：tiiy

# TODO
1. Currently, the C++-witten PSOPS can support PyTorch C++ API. However, the python API Py_PSOPS of PSOPS can only successfully load PSOPS.dll/PSOPS.so without PyTorch C++ API. We need to find a way to deal with the violation between Python and PyTorch C++ API when using PyTorch C++ API-integrated PSOPS.dll/PSOPS.so. 

2. Add comment.

3. Develop more general modules. 

4. Improve performance of the general module and the trained models. 

# References
[1] **T. Xiao**, Y. Chen*, T. He, and H. Guan, “Neural ODE and DAE Modules for Power System Dynamic Component Modeling,” [arxiv](https://arxiv.org/abs/2110.12981).

[2] **T. Xiao**, Y. Chen*, J. Wang, S. Huang, W. Tong, and T. He, “Exploration of AI-Oriented Power System Transient Stability Simulations,” [arxiv](http://arxiv.org/abs/2110.00931).
