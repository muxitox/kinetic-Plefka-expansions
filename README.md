# A unifying framework for mean field theories of asymmetric kinetic Ising systems

Code reproducing the models used in the article Aguilera, M, Moosavi, SA & Shimazaki H (2021). [A unifying framework for mean-field theories of asymmetric kinetic Ising systems](https://www.nature.com/articles/s41467-021-20890-5). _Nature Communications_ 12:1197; [https://doi.org/10.1038/s41467-021-20890](https://doi.org/10.1038/s41467-021-20890).

## Abstract

Kinetic Ising models are powerful tools for studying the non-equilibrium dynamics of complex systems. As their behaviour is not tractable for large networks, many mean-field methods have been proposed for their analysis, each based on unique assumptions about the system's temporal evolution. This disparity of approaches makes it challenging to systematically advance mean-field methods beyond previous contributions. Here, we propose a unified framework for mean-field theories of asymmetric kinetic Ising systems from an information geometry perspective. The framework is built on Plefka expansions of a system around a simplified model obtained by an orthogonal projection to a sub-manifold of tractable probability distributions. This view not only unifies previous methods but also allows us to develop novel methods that, in contrast with traditional approaches, preserve the system's correlations. We show that these new methods can outperform previous ones in predicting and assessing network properties near critical regimes. 

## Description of the code

The code reproducing the approximation methods in the paper is contained in the 'plefka/' folder:
* 'plefka_expansions.py', calls the functions to perform each approximation
* 'plefka_functions.py', contains additional functions computing the equations behind each approximation

The results in the paper can be reproduced by running the scripts in the 'results/' folder as follows:
* Simulation data of the asymmetric SK model to test against the models can be generated using 'generate_data.py', which uses 'kinetic_ising/kinetic_ising.py' to update a kinetic Ising model.
* The forward Ising problem is computed with 'forward-Ising-problem.py', which compares the result with the simulated data. The figure containing the results can be generated using 'forward-Ising-problem-results.py'
* The inverse Ising problem is computed with 'inverse-Ising-problem.py', which estimates the model parameters from the simulated data. The figure containing the results can be generated using 'inverse-Ising-problem-results.py'
* Simulation data for testing phase reconstruction can be computed using 'generate_data_transition.py'.
* The phase transition reconstruction results can be reproduced running 'reconstruction-Ising-problem.py', which computes the forward Ising problem for the original models (with more values of β) and the models from solving the inverse problem. The figure containing the results can be generated using 'reconstruction-Ising-problem-results.py'
* Finally, a comparison of execution times (Supplementary note), can be computed using 'execution-time-results.py'

Note that some steps are computationally quite costly. Specifically, the results of  'generate_data.py', 'generate_data_transition.py' and 'reconstruction-Ising-problem.py' were obtained parallelizing the code in a computer cluster. Alternatively, the data used for reproducing the exact results in the paper can be accessed in the Zenodo repository [https://zenodo.org/record/4318983](https://zenodo.org/record/4318983) or downloaded directly using the script 'download-datasets.py'. After this, any of the steps above can be reproduced directly.

Data generated by the different scripts is stored in the 'results/data/' folder.

Figures in the paper are generated in the folder 'results/img/' which contains the figures in the paper plus vectorial files for Fig. 1 and Fig. 2.

### For visualization install the following packages:

sudo apt install cm-super
sudo apt-get install dvipng
