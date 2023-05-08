# Streamlines for Embroidery

Let the streamlines flow from your favorite photos to embroidery patterns you can sew on your hoodies using an automatic embroidery machine!

## Environment Setup

The code is not well tested and optimized, but it does its job hopefully.

```
conda env create -f environment.yml
```

Installation in a virtual environment via `pip` is recommended as some packages (e.g. `pyembroidery`) are not available on conda.

Packages: 
```
pyembroidery numpy scipy numba matplotlib scikit-learn scikit-image jupyter networkx triangle labelme
```

## Run the script

`examples/run_multiple_patches.py` 

`examples/run_interactive_regularization.py`

`examples/run_analytical_field.py`

## Licensing

The project is licensed under MPL2.0, notice can be found at the beginning of each source file in the project, and here.