\# Incident Detection Baseline System (Reproducible Release)



This repository provides a complete, reproducible implementation of a \*\*log-based incident detection baseline\*\* using classical machine learning. It accompanies the research paper \*“Intelligent Predictive Incident Management for Modern IT Ecosystems.”\*



The system evaluates incident classification on both \*\*synthetic logs\*\* and the \*\*HDFS Public Log Corpus\*\*, using a lightweight Logistic Regression model.



---



\## 📁 Repository Structure

llama-incident-system-public/

│

├── data/ # Synthetic + HDFS datasets

├── models/ # Saved baseline models (.joblib)

├── results/ # Confusion matrix, metrics, evaluation outputs

├── sample\_logs/ # Example multi-component logs

└── src/ # Preprocessing, training, evaluation scripts



yaml

Copy code

---



\## ⚙️ 1. Overview

This repository includes:



\- Synthetic dataset  

\- HDFS log preprocessing  

\- Structured feature extraction  

\- Logistic Regression baseline  

\- Complete training + evaluation pipeline  

\- Reproducible outputs (confusion matrix, metrics)  



The implementation focuses on \*\*interpretability, reproducibility, and reliability\*\*.



---



\## 📊 2. Model Performance



\### \*\*Synthetic Dataset\*\*

\- Accuracy: \*\*94.8%\*\*



\### \*\*HDFS Public Log Corpus\*\*

\- Accuracy: \*\*85.5%\*\*

\- Confusion matrix: `\[\[169, 31], \[27, 173]]`



These match results reported in the research paper.



---



\## 🛠️ 3. Training



```bash

python src/train.py

📈 4. Evaluation

bash

Copy code

python src/evaluate.py

Outputs stored in:



bash

Copy code

results/evaluation.txt

results/publish\_evaluation.txt

results/confusion\_matrix.png

🔮 5. Future Extensions

This baseline establishes a foundation for upcoming improvements including:



Transformer-based log embeddings



LLaMA-driven semantic reasoning



Multi-modal log/metric fusion



Sequence-learning (LSTM/GRU) architectures



📚 6. Citation

bibtex

Copy code

@software{incident\_system\_2025,

&nbsp; author    = {Laksh Gajre and Venkatraman Naik and Amruth M S},

&nbsp; title     = {Incident Detection Baseline System - Public Reproducibility Release},

&nbsp; year      = {2025},

&nbsp; url       = {https://github.com/lakshgajre2004-sys/llama-incident-system-public}

}

📞 Contact

Laksh Gajre

Venkatraman Naik

Amruth M S



Email: lakshgajre.bs23@bmsce.ac.in

GitHub: https://github.com/lakshgajre2004-sys

