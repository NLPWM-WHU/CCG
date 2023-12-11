# CCG

## Overview
![image](https://github.com/Double1203/CCG/blob/main/figure/CCG_framework.png)

## Setup

### Install dependencies
Please install all the dependency packages using the following command:
```
pip install -r requirements.txt
```
### Download and preprocess the datasets
Our experiments are based on three datasets: SemEval, RE-Attack, ACE2005. Please find the links and pre-processing below:
* SemEval: We provide the processed SemEval dataset in the "semeval" folder.
* RE-Attack: We provide the processed RE-Attack dataset in the "re-attack" folder, note that the train and dev sets are follow semeval.
* ACE2005: We can't provide the ACE2005 dataset directly because of copyright, but the dataset can be downloaded in https://catalog.ldc.upenn.edu/LDC2006T06. And we use the preprocessing code from [DyGIE repo](https://github.com/luanyi/DyGIE/tree/master/preprocessing). Please follow the instructions to preprocess the ACE05 datasets.

## Quick Start
We release our code of three modules in "code" folder with corresponding names.
