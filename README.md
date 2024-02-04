#  Release notes (final version 4.1):
-  Simplification of the problem: Examining the evolution of distributed approaches in a context where validation of each one is not necessary, as cross-validation with all medical data was performed when determining the optimal artificial neural network (ANN) architecture.

# General description of the project:

This work involves implementing a use case, the Heart Disease dataset, comparing various data decentralization architectures, such as Federated Learning, Ring All-Reduce, and Gossip Learning.

# Related previous work. Data preparation and analysis.
Raw initial data: heart.csv <br>
Data Preparation and Exploratory Data Analysis (EDA): dataset5_heart_DP&EDA.ipynb <br>
Output of the last notebook: heart_ConditionalMeanImputation.csv <br>

# Current repository structure. AI issue (final version 4.1):
## Notebook to find the optimal ANN architecture:
OptimalArchitecture_learningPriority_v4.1.ipynb <br>
## Notebooks to develop the distributed learning architectures:
Federated Learning: Test_FL_v4.1_noValidation.ipynb <br>
Ring All_Reduce: Test_RAR_v4.1_noValidation.ipynb <br>
Conditional Gossip Learning: Test_GL_fixed_v4.1_noValidation.ipynb <br>
Random Gossip Learning: Test_GL_random_v4.1_noValidation.ipynb <br>
Customized architecture: Test_customized_1_v4.1_noValidation.ipynb <br>
## Output of the architectures in pickle format (test metrics and model weights, same as in version 4):
results_Test_FL_v4_noValidation.pkl <br>
results_Test_RAR_v4_noValidation.pkl <br>
results_Test_GL_fixed_v4_noValidation.pkl <br>
results_Test_GL_random_v4_noValidation.pkl <br>
results_Test_customized_1_v4_noValidation.pkl <br>

## Analysis of the results:
###  At the final of notebooks of the distributed approaches:
-  Evolution of test metrics (loss, accuracy, AUC) per client
-  Weight divergence between pairs of clients for round 50, where loss metric converges.
### Notebook of weighted averages of the distributed architectures:
-  Average test metrics <br>
-  Average weight divergence <br>
analysisResults_v4.1_noValidation.ipynb <br>

# Requirements
## Dependencies
-  Python version: 3.8.10
-  NumPy version: 1.23.4
-  Pandas version: 2.0.3
-  Matplotlib version: 1.23.4
-  Scikit-learn version: 1.3.2
-  TensorFlow version: 2.11.0

Note: Specific imported modules are shown at the beggining of each notebook.

## Hardware Specifications

- **GPU:** NVIDIA Tesla T4
- **RAM:** 16 GB
- **Platform:** AI4EOSC*

(*) AI4EOSC Platform

AI4EOSC is a platform designed to harness artificial intelligence (AI), deep learning (DL), and machine learning (ML) technologies within the European Open Science Cloud (EOSC) framework. The platform facilitates the utilization of advanced AI techniques for research and innovation, offering users tools and services to effectively work with large distributed datasets within the EOSC framework. More info: https://ai4eosc.eu/

# Other versions:
## Version 1:
###  Files:
-  initialTest_architecture_privacyPriority.ipynb
-  initialTest_architecture_learningPriority.ipynb
-  initialTest_GossipLearning_fixed.ipynb
-  initialTest_GossipLearning_random.ipynb
-  initialTest_RingAllReduce.ipynb
-  initialTest_FL_SMA.ipynb
-  initialTest_FL_WMA.ipynb
###  Release notes:
-  Initial search of optimal ANN architecture.
-  The optimal architecture was found by cross-validation and classification metrics on test subset. Two variants:
"..._privacyPriority" considering only the client with more quantity of data.
"..._learningPriority" considering the medical data of all the clients.
-  Initial version of the distributed architectures
-  Each notebook is explained with pseudocodes at the beginning.
-  Test metrics was measured at the final round
-  Model weights were programmed to be saved in a .h5 format. 
-  FL architecture shows a good performance and a correct implementation of the code.
-  The validation metrics evolution of other architecures appears no to have the correct behavior. 

## Version 2:
###  Files:
-  Test_customized_1_v2.ipynb
-  Test_RAR_v2.ipynb
-  results_test_RAR_v2.pkl
-  Test_GL_fixed_v2.ipynb
-  Test_GL_random_v2.ipynb

###  Release notes:
-  Problem reduced to 4 artificial clients, waiting an improve of the performance and the behaviour of the metrics. 
-  Code improved to accept complex distributed architectures where a client receives more than one model weights of different clients.
-  Corrected code of client-to-client distributed architectures, having only one fitting per round and client.
-  Test metrics are not quite good but they presents the correct behavoir, consequence of the corrected code of the distributed architectures.
-  Corrected code of plotting for validation and test to show the evolution of each client in the rounds.

## Version 3:
### Files:
-  Test_RAR_v3_validation.ipynb
-  Test_RAR_v3_noValidation.ipynb
-  Test_customized_1_v3_noValidation.ipynb
-  Test_customized_1_v3_validation.ipynb
-  Test_GL_fixed_v3_noValidation.ipynb
-  Test_GL_fixed_v3_validation.ipynb
-  Test_GL_random_v3_noValidation.ipynb
-  Test_GL_random_v3_validation.ipynb
-  results_test_RAR_v3_validation.pkl
-  results_test_RAR_v3_noValidation.pkl
### Release notes:
-  Now each architecture have a version with validation and without validation. Validation it is not necessary because cross-validation with all the medical data was carried out when determining the optimal ANN architecture.
-  Corrected lack of data scaling, causing a great improve in metrics values.
-  An attempt to recover the study of the five hospitals in some architectures.
-  Beta version of calculation of weight divergence metric in the RAR architecture.
-  Saving models for each client each 5 rounds and test metrics (and validation each exists) for all the rounds. The format elected was pickle.

## Version 4:
### Files:
-  Test_RAR_v4_validation.ipynb
-  Test_RAR_v4_noValidation.ipynb
-  Test_customized_1_v4_noValidation.ipynb
-  Test_customized_1_v4_validation.ipynb
-  Test_GL_fixed_v4_noValidation.ipynb
-  Test_GL_fixed_v4_validation.ipynb
-  Test_GL_random_v4_noValidation.ipynb
-  Test_GL_random_v4_validation.ipynb
-  results_Test_customized_1_v4_validation.pkl
-  results_Test_customized_1_v4_noValidation.pkl
-  results_Test_GL_fixed_v4_validation.pkl
-  results_Test_GL_fixed_v4_noValidation.pkl
-  results_Test_GL_random_v4_validation.pkl
-  results_Test_GL_random_v4_noValidation.pkl
-  results_Test_RAR_v4_validation.pkl
-  results_Test_RAR_v4_noValidation.pkl
### Release notes:
-  Adjust of limits of all the plots to facilitate comparison of results.
-  Calculation of weight_divergence for all the client-to-client architectures at round 50 to facilitate comparison of results.
-  Code of FL architecture.
-  Implementation of final comments and descriptions in the notebooks.
