# Is a bunch of words enough to detect disagreement in hateful content? 

This repository contains the code developed for "Is a bunch of words enough to detect disagreement in hateful content?". 
This repository contains several scripts to reproduce the results presented in the paper. 
The scripts allow to estimate the Disagreement Scores for each constituent in the training dataset and to estimate the final sentence label according to the four proposed strategies: Sum, Mean, Median, and Minimum. 

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

### Baselines Models
Within the paper, the proposed approach has been compared with several basilines models:
- **G-minimum** : the best approach presented in [Unraveling Disagreement Constituents in Hateful Speech](https://link.springer.com/chapter/10.1007/978-3-031-56066-8_3) ([code available for reproducidibility](https://github.com/MIND-Lab/Unrevealing-Disagreement-Constituents-in-Hateful-Speech))
- **mBERT**: state-of-the-art model by [Devlin et al. (2018)](https://arxiv.org/abs/1810.04805) (Huggingface model `bert-base-multilingual-cased`)
- **Llama-2**: state-of-the-art model by [Touvron et al. (2023)](https://arxiv.org/abs/2302.13971)  (Huggingface model `Llama-2-7b-chat-hf`)
- **Mistral-7B**: state-of-the-art model by [Jiang et al., (2023)](https://arxiv.org/abs/2310.06825)  (Huggingface model `Mistral-7B-Instruct-v0.3`)
- **Llama-3.2**: state-of-the-art model by [Dubey et al. (2024)](https://arxiv.org/abs/2407.21783)  (Huggingface model `Llama-3.2-3B-Instruct`)
- **Phi-3.5**: state-of-the-art model by [Haider et al. (2024)](https://arxiv.org/abs/2404.14219)  (Huggingface model `Phi-3.5-mini-instruct`).

The selected state-of-the-art baselines include generative LLMs. While mBERT has been fine-tuned for the classification task by concatenating a final classification layer, generative LLMs have been instruction-tuned to adapt their generative capabilities for the specific classification task.



## Citation
If you found our work useful, please cite our papers:
[Is a bunch of words enough to detect disagreement in hateful content?](https://aclanthology.org/2025.comedi-1.1/)

```
@inproceedings{rizzi2025bunch,
  title={Is a bunch of words enough to detect disagreement in hateful content?},
  author={Rizzi, Giulia and Rosso, Paolo and Fersini, Elisabetta},
  booktitle={COLING 2025, Proceedings of Context and Meaning: Navigating Disagreements in NLP Annotation},
  pages={1--11},
  year={2025}
}
```
