# jexercise-data-analysis

Developed by *Khoa Ngoc Le* and *Herman Myrbråten Karlsen*.

**The code in this repository was developed for our Master's thesis.**

## Structure
The Jupyter notebook files (.ipynb extension) are primarily used for data exploration and experimentation.

### Bokeh notebook
The Bokeh notebook contains the tool we developed to assist with data analysis and visualization. As the name suggests,
we based the tool on the Bokeh framework ([Bokeh documentation](https://bokeh.pydata.org/en/latest/)). Bokeh is fully compatible
with Jupyter notebooks and integrates seamlessly. The tool enables reading DataFrames (stored in CSV files) and easily plot the
chosen Series. The tool scales the data automatically before plotting and provides tooltips to inspect data points.
When clicking on a data point, the corresponding source code snapshot is loaded in the element next to the plot window.

### raw_jexercise_data.py
The raw_jexercise_data module

### preprocessing_data.py
The preprocessing_data module
