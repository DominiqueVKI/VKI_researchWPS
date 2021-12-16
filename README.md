This repository is structured as follows:
We have two folders grouping the two main data-driven regression tools used to infer wall pressure spectral models:

1) GEP 
2) NN

The GEP folder has a script that show you an example run of the GEP on the goody model.
GEP_LocalOpt_V2_Goody is a script that uses the modified version of the geppy package to perform a symbolic regression of the Goody model based on a synthetic dataset.

The NN folder contains three scripts that show you how to build an ANN model, run the model and evaluate its uncertainty using an ensemble approach
ANN_built is a script that shows you how to build an ANN to model WPS
ANN_display is a script that loads a model and displays its prediction for a set of boundary layer parameters
ANN_uncertainty is a script that evaluate the ANN uncertainty using an ensemble approach


** Geppy **

Initial python implementation of GEP 

https://geppy.readthedocs.io/en/latest/

** DEAP **

Geppy was built on the deap package

https://deap.readthedocs.io/en/master/

** TENSORFLOW **

The Neural network was built on the TensorFlow package 

https://www.tensorflow.org/install/pip?hl=fr

** Methodology **

See section 3.2 of the Journal of Sound and Vibration paper:

Dominique, J., Christophe, J., Schram, C., & Sandberg, R. D. (2021). Inferring empirical wall pressure spectral models with Gene Expression Programming. Journal of Sound and Vibration, 506, 116162.
https://doi.org/10.1016/j.jsv.2021.116162

See section 5 of the being reviewed Physics of Fluids paper:

"Artificial Neural Networks Modelling of Wall Pressure Spectra BeneathTurbulent Boundary Layers" 
Dominique Joachim, van den Berghe Jan, Schram Christophe, Mendez A Miguel
