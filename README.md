# jexercise-data-analysis

Developed by *Khoa Ngoc Le* and *Herman Myrbråten Karlsen* through pair programming, hence all the code being committed by one contributor.

**The code in this repository was developed for our Master's thesis.**

## Jupyter notebooks
The Jupyter notebook files (.ipynb extension) are primarily used for data exploration and experimentation.
The two most notable notebooks are:

### Bokeh.ipynb
The Bokeh notebook contains the tool we developed to assist with data analysis and visualization. As the name suggests,
we based the tool on the Bokeh framework ([Bokeh documentation](https://bokeh.pydata.org/en/latest/)). Bokeh is fully compatible
with Jupyter notebooks and integrates seamlessly. The tool enables reading DataFrames (stored in CSV files) and easily plotting the
chosen Series. The tool scales the data automatically before plotting and provides tooltips to inspect data points.
When clicking on a data point, the source code snapshot from that time stamp is displayed in the element next to the plot window.

### Classify struggling.ipynb
This notebook contains the 10 samples which we inspected during our experiments. The results from running our classification algorithm
on these samples are displayed in this notebook, in the form of plots and tables. Each sample is also prefaced with some information about
which grade the student achieved and comments on how the student approached the exercise.

## Python modules

### raw_jexercise_data.py
The raw_jexercise_data module contains all the functions necessary for reading the .ex (XML) files in to pandas DataFrames.
ElementTree is used to parse the .ex files, and the resulting DataFrames are saved as CSV files to make it easier to save/load with pandas.

### preprocessing_data.py
The preprocessing_data module contains a number of different functions. Functions for:

- Filling (replacing) NaN values
- Aggregating (grouping) several columns
- Creating relative time, where any time gap longer than 10 minutes in the data is shortened to 10 minutes
- 'Patching' the StoredString series together, producing the complete source code
- Calculating the line (or character) difference between two source code snapshots. To do this we use SequenceMatcher and Differ from the
Difflib package (https://docs.python.org/3.5/library/difflib.html)

The logic for calculating the indicators and performing the classification is also found in this module.
The function `run_algorithm(assignment, hash_id, exercise, smoothing_window=5, save=True)` performs all the necessary preprocessing,
runs the algorithm and optionally saves the result in a CSV file.

## Recreating the Python environment using conda
Requires **conda**: https://conda.io/docs/

the `./conda/` folder contains the environment.yml file which can be used to recreate the conda environment we used during experimentation:
`conda env create -f environment.yml`