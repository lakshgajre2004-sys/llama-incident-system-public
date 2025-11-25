[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17683498.svg)](https://doi.org/10.5281/zenodo.17683498)


\# LLaMA-Incident-System (Reproducible Research Release)



A reproducible pipeline for evaluating LLaMA-based AIOps incident prediction, alert understanding, and anomaly reasoning.

This repository provides all datasets, logs, scripts, and evaluation procedures required to reproduce experiments from the research work.



---------------------------------------------------------------------



\# Repository Structure



gh-public/

|

|-- datasets/

|     \\-- sample\_dataset\_1000.csv

|

|-- sample\_logs/

|     |-- sample\_aggregated.log

|     |-- sample\_api.log

|     |-- sample\_auth.log

|     |-- sample\_db.log

|     |-- sample\_scheduler.log

|     \\-- sample\_worker.log

|

|-- results/

|     |-- sample\_eval.txt

|     \\-- sample\_model.joblib

|

|-- src/

|     |-- generate\_sample\_logs.py

|     |-- train\_small.py

|     \\-- eval\_small.py

|

\\-- README.md



---------------------------------------------------------------------



\# 1. Overview



This repository provides a fully reproducible AIOps incident prediction baseline, including:



\- Synthetic reproducible dataset  

\- Synthetic multi-component logs  

\- Train/test split  

\- Baseline ML model (Logistic Regression)  

\- Evaluation pipeline  

\- End-to-end instructions  



Designed for transparency, replicability, and conference submission requirements.



---------------------------------------------------------------------



\# 2. Dataset



The dataset used here is a 1000-row public subset located at:



datasets/sample\_dataset\_1000.csv



Features:

\- 20 numeric metrics (f0 to f19)

\- Binary target label (0 or 1)

\- Stratified sampling



Load using:



import pandas as pd

df = pd.read\_csv("datasets/sample\_dataset\_1000.csv")



---------------------------------------------------------------------



\# 3. Synthetic Logs



Synthetic application logs are provided under sample\_logs/.  

These simulate logs from:



\- auth service

\- api service

\- database

\- worker

\- scheduler



To regenerate logs:



python src/generate\_sample\_logs.py



---------------------------------------------------------------------



\# 4. Baseline Model



The baseline model used:



\- Logistic Regression

\- StandardScaler

\- Liblinear solver

\- 80/20 train-test split

\- Accuracy approximately 0.835



Train using:



python src/train\_small.py



---------------------------------------------------------------------



\# 5. Evaluation



Run:



python src/eval\_small.py



This prints a classification report and stores results in:



results/sample\_eval.txt



---------------------------------------------------------------------



\# 6. Reproducibility Summary



This repository includes:



\- Dataset: Yes  

\- Training Script: Yes  

\- Evaluation Script: Yes  

\- Synthetic Logs: Yes  

\- Saved Model: Yes  

\- Instructions: Yes  



Reproduce results:



python src/train\_small.py

python src/eval\_small.py



---------------------------------------------------------------------



\# 7. Citation



@software{llama\_incident\_system\_2025,

&nbsp; author = {Laksh Gajre},

&nbsp; title = {LLaMA-Incident-System Public Reproducibility Release},

&nbsp; year = {2025},

&nbsp; publisher = {GitHub},

&nbsp; url = {https://github.com/lakshgajre2004-sys/llama-incident-system-public}

}



(You can add a DOI later if you create a Zenodo release.)



---------------------------------------------------------------------



\# 8. License



Recommended:

\- MIT License

\- Apache 2.0 License



If you need a LICENSE file, ask and it will be generated.



---------------------------------------------------------------------



\# 9. Contact



Laksh Gajre,Venkatraman Naik,Amruth MS  

Email: lakshgajre.bs23@bmsce.ac.in
       venkatraman.bs24@bmsce.ac.in 
       amruthams.bs23@bmsce.ac.in

GitHub: https://github.com/lakshgajre2004-sys



---------------------------------------------------------------------



End of README.md



