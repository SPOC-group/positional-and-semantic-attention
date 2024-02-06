# Numerics and Experiments

## Two algorithms for solving the histogram task

- The experiments for this section from the paper are can be run using `run_histogram.py`, and plotted using `01-histogram.ipynb`


## A model of positional and semantic embeddings

- The results are visualized in `02-toy_model_plots.ipynb` using the raw csv's available in the repo. The raw data is created in separated files for theory and empirics.
- Theory:
    - The numerical solution of equations (7-8) of Result 4.2 is implemented in theory/theory.ipynb. The notebooks uses the [quadpy](https://pypi.org/project/quadpy/) package for 2d numerical integration. Note that this method is not adaptative, and depending on the accuracy of the selected scheme, the iterative resolution of (7-8) may present instabilities or inaccuracies, notably if too low of an order is selected.

- Empirics: 
    - The module `src.empirics_mixed_teacher_softmax` is used for the student with rank 1, for the experiments presented in the main and the appendix
    - The module `src.empirics_mixed_teacher_softmax_r2` is used for the student with rank 2, for the experiments presented in the appendix
    - The experiments frequired for all figures can be run using the scripts
        - `run_<exp_name>.sh`
    - The original data is quite large, so it can be processed to csvs using the `convert_data.ipynb`
    - The csvs are available in the repository, while the raw data is available upon request to the authors

