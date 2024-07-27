## To run:
```
# in a build directory:
$ cmake -DDEAL_II_DIR=<path-to-deal-ii> <path-to-plasticitylab>
$ make release
$ make -j 8 &&  $HOME/share/bin/mpirun -n 18 ./PlasticityLab

# The triangulation is configured in PlasticityLabProgDrivers.cpp in run()
# The material is configured in main.cpp
# The timestep is configured in PlasticityLabProg.h
```

## Citation:
If you use this code as part of your work, please cite:
Hamed, M.M.O., McBride, A.T. & Reddy, B.D. An ALE approach for large-deformation thermoplasticity with application to friction welding. Comput Mech 72, 803–826 (2023).
https://doi.org/10.1007/s00466-023-02303-0

