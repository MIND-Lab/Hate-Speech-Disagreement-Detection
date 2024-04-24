# Is a bunch of words enough to detect disagreement in hateful content?

This repository contains the code developed for "Is a bunch of words enough to detect disagreement in hateful content?". 
This repository contains several scripts to reproduce the results presented in the paper. 
The scripts allow both to estimate the Constituent Disagreement Scores for each constituent in the training dataset and to estimate Sentence Disagreement Scores (SDS) according to the four proposed strategies: Sum, Mean, Median, and Minimum. 

### Datasets
The datasets are reserved for the participants of Task 11 @SemEval2023 on Learning with Disagreements (LeWiDi), 2nd edition. Please refer to the [paper](https://aclanthology.org/2023.semeval-1.314/) for additional information.
To request the datasets, please refer to the [official LeWiDi page](https://le-wi-di.github.io/).
The proposed code involves positioning the dataset in the main directory. To execute the code, please place the datasets there or change the reference paths.

### Running
The scripts (*.py) allow the reproducibility of the results presented in the paper.
Before running the code, the Datasets should be downloaded (see "Dataset" section) and organized in a "Data" folder.
The scripts allow to execute the entire procedure described in the paper and should be executed according to the following order:
- *1_dataProcessing_tokenSelection.py*: allows preprocessing the data and selecting the constituents from the training and validation dataset.
- *2_TestProcessing.py*: allows preprocessing the data and selecting the constituents from the test dataset.
- *3_evaluateDev_ParameterEstimation.py*: evaluate the four disagreement estimation procedures on the validation set and estimate the best thresholds.
- *4_evaluation.py*: evaluate the four disagreement estimation procedures on the test set.
  
Before executing the above-described scripts, all requirements must be met. Installation can be fulfilled manually or via `pip install -r requirements.txt`.

