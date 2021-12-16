This respository is structured as followed:
We have two folder grouping the two main data-driven regression tools used to infer wall pressure spectral models:

1) GEP 
2) NN

The GEP folder has a scripts that show you an example run of the GEP on the goody model
GEP_LocalOpt_V2_Goody is a script that use the modified version of the geppy package to perform a symbolic regression of the Goody model based on a synthetic dataset.

The NN folder contain 3 scripts that show you how to build a ANN model, run the model and evaluate its uncertainty using an ensemble approach
ANN_built is a sript that shows you how to build a ANN to model WPS
ANN_display is a script that load a model and display its prediction for a set of boundray layer parameters
ANN_uncertainty is a sript that evaluate the ANN uncratinty using an esemble approach


** Geppy **

Initial python implementation of GEP 

https://geppy.readthedocs.io/en/latest/

** DEAP **

Geppy was build on the deap package

https://deap.readthedocs.io/en/master/

** Methodology **

See section 3.2 of the Journal of Sound and Vibration paper:

Dominique, J., Christophe, J., Schram, C., & Sandberg, R. D. (2021). Inferring empirical wall pressure spectral models with Gene Expression Programming. Journal of Sound and Vibration, 506, 116162.
https://doi.org/10.1016/j.jsv.2021.116162

See section 5 of the being reviewed Physics of Fluids paper
"Artificial Neural Networks Modelling of Wall Pressure Spectra BeneathTurbulent Boundary Layers" 
Dominique Joachim, van den Berghe Jan, Schram Christophe, Mendez A Miguel
