This is the official repository for the paper "Classification of health product defect reports by deep learning".

# Getting Started

## Prerequisites

1. Version of software:

- Anaconda version: 2.3.1

- Python version: 3.7.15

- CUDA version: 11.0.167

2. The packages, dependencis, and version information required to run the provided notebook are included in requirements.txt file.


## Code

The Product_defects_notebook.ipynb file contains the full pipeline of this study. 

The pipeline includes the following components:
1. Data preprocessing
2. Model training (loading, fine-tuning, and prompt-tuning)
3. Model evaluation
4. Interpretability analyses
5. Performance analyses 


## Model

The two model weights can be accessed and downloaded from [this link](https://drive.google.com/drive/folders/1wqiBd_-5pn3tRm5W27kZlB9wztk41F5U?usp=drive_link):
1. Bert-base fine-tuned model: [MedDefects-BERT](https://drive.google.com/drive/folders/1AI7sttjr67IcwaFA0Z0XHyA8BnSCAOej?usp=drive_link)
2. Bert-base deep-prompt-tuned model: [MedDefects-DPT-BERT](https://drive.google.com/file/d/1bm-D33-vFT0ArKTxsF1hXAsLXDaXNX-U/view?usp=drive_link)


# Demo for testing the software

Demo scripts and test datasets have been provided (15-Sep-2023) in the Demo folder. Please refert to the Readme.md file of that folder for more details and follow the instruction in the guide to run the scripts and reproduce the results.




# Contact
If there are any questions, please contact: vicente.enrique@synapxe.sg
