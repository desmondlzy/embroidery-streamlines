![teaser image, mashup of three photos of  embroidery patterns designed by our algorithm](./images/mashup.jpg)

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

### `examples/run_multiple_patches.py` 
Run the pipeline on the density and direction fields extracted from each segment of an image. Generate a preview of the embroidery pattern, and `.emb` files that can be used on the Bernina B590 machine.

`examples/run_interactive_regularization.py`
Demo of the interactive regularization. An example of a horizontal direction field and a linearly increasing density field is used here. A matplotlib popup will appear for the interactive demo.

`examples/run_analytical_field.py`
Run the pipeline on some analytical fields, please check the source code to choose from the fields available, or make your own fields.

## Licensing

The project is licensed under MPL2.0, notice can be found at the beginning of each source file in the project, and [here](./LICENSE).