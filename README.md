# Multispot Paper Notebooks

This repository contains the notebooks used for the analysis the paper:

*Multi-spot single-molecule FRET: towards high-throughput analysis of freely diffusing molecules
Ingargiola et al. (2016)*

You can navigate the notebooks online starting from [index.ipynb](http://nbviewer.jupyter.org/github/tritemio/multispot_paper/blob/master/index.ipynb),
or you can reproduce the full analysis by following the steps below.

## How to use?

### Install FRETBursts
In order to execute these notebooks you need to install
FRETBursts first. If you have already installed python through `conda` just type:

    conda install fretbursts -c conda-forge

Otherwise see the instruction on the [FRETBursts manual](http://fretbursts.readthedocs.io/en/latest/getting_started.html).

### Download notebooks and data

Download the ZIP archive from [this link](https://github.com/tritemio/multispot_paper/archive/master.zip)
and extract the archive.

You also need to download two datasets from figshare. Put files from the
[single-spot dataset](https://doi.org/10.6084/m9.figshare.1098961) in the in the `data/singlespot` folder,
and the files from the [multi-spot dataset](https://doi.org/10.6084/m9.figshare.1098962)
in the `data/multispot` folder.

### Use the notebooks

The notebook [index.ipynb](http://nbviewer.jupyter.org/github/tritemio/multispot_paper/blob/master/index.ipynb) contains links
to all the other notebooks (with a brief explanation of what each notebook does).
It also contains links the result data files (CSV format, stored in the [results](results) folder).

Running index notebook you will re-execute the full analysis and regenerate the
[output notebooks](out_notebooks)], the [figures](figures) and all the
[numeric results](results).

### Reproducibility
For now, you should be able to re-run the analysis on the current scientific python stack.

In the future, new version some used library can introduce incompatibilities that break the code
posted here. In this case, you can recreate the original conda environment used for running this
analysis using the file `environment.yml`.

## License

All the text and documentation are released under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
All code is under the MIT license (see LICENSE.txt).

Copyright 2016 The Regents of the University of California, Antonino Ingargiola
