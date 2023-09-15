## Demonstration of the Software

The following guide contains instructions to install and test the MedDefects-BERT software developed in this study.

The Demo contains two scripts:
1. “Demo_Inference.py” takes as input a single report or a batch of reports in a dataset (with or without known labels) and outputs the predicted label(s) and their corresponding confidence scores.
2. “Demo_Shap.py” takes as input a single report or a batch of reports in a dataset (with or without known labels) and outputs the Shapley values of the text in a gradient map and a force plot as described in the Methods section and illustrated in Figure 3 of the paper.

Follow the provided instructions step by step in order to reproduce the provided test results and test the software with your own test dataset.

## Installation
The following Demo can be run on a Windows 7 and above Operating System. The Demo runs faster in a GPU enabled machine. 
However, it can be run on a machine without GPU. Add logit in script: 
```bash
if not GPU:
  os.environ = [‘-1’]
```

Before running the demo ensure that the following applications are installed:
-	Python 3.7.15 or above
-	Windows 7 or above
-	Anaconda 2.3.1 or above

### Install steps:

1. Open an anaconda prompt terminal, create an environment for this Demo with python version 3.7.15,  and install the packages specified in the requirements.txt file.
3. Create a folder in your local machine where the code, the model weights, and the test datasets of this Demo will be copied.
4. Navigate to the Demo folder and clone this repository or copy the scripts “Demo_Inference.py”  and “Demo_Shape.py”.
5. Create a subfolder named “models” in the Demo directory and download on it the weights of the MedDefects-BERT model from the link here. Unzip the file inside the models subfolder. The model weights and configuration will be located in a subfolder named “MedDefects-BERT” in the “models” folder.
6. If the repository is not cloned, create a subfolder named “data” and copy the csv files “Demo_test_dataset.csv” and “Demo_test_dataset_without_labels.csv” containing the test datasets. If the repository is cloned, the folder with the datasets will be automatically created.
7. If the repository is not cloned, create a subfolder named “results” and (optionally) copy the csv files from the same folder in the repository. If the repository is cloned, the folder with the datasets will be automatically created. This folder contains the results of the test runs for reference. Additional test runs will store the results in this folder, with a user provided output file name.


### Notes about the csv files

We provide two csv files for testing the model:
1. “Demo_test_dataset.csv” contains 3 columns. Columns Title and Desc contain the title of the report and description of the report in text format. Column MedDRA contains the annotated label of the report (ground truth) also in text format.
2. “Demo_test_dataset_without_labels.csv” is identical to the first one but does not contain the column MedDRA.

If you want to test your own dataset, you need to provide your reports in the same format with the same column names. Columns Title and Desc are mandatory as they are required to generate the input for the model. Column MedDRA is optional. If present, this column is displayed along with the model predictions for reference. If not present, only the model predictions are displayed.


## Inference

To run the script “Demo_Inference.py” for inference follow these steps:
1. Open an anaconda terminal, activate the Demo environment that you have created, and navigate to the Demo directory
2. Execute the script “Demo_Inference.py” by typing the following command:
```bash
python Demo_Inference.py
```
3. The script will ask whether the input is a single report or a batch of reports. Enter <T> for single or <F> for batch:
```bash
Is the test data a single case (T/F)?
If F is selected a csv file with a batch os cases is expected: 
```
4. Next the script will ask for the name of the output file. Provide a name without blank spaces and without extension, as shown below. By default, the output file will be saved in the “results” directory with csv format.
```bash
Please enter the name of the output prediction file (without blank spaces): 
```
  4a. If a single report is selected in 2, the script will sequentially ask for the Title and the Description of the report. Copy and paste the Title and Description of the report as shown below:
```bash
Please provide the case Title: 
```
```bash
Please provide the case Description: 
```
  4b. If a batch of reports is selected in 2, the script will ask for the path to the csv file containing the batch of the reports. Enter a valid path such as <./data/Demo_test_dataset_without_labels.csv> as shown below. If an invalid path is entered, the script will use the default path <./data/Demo_test_dataset.csv> to the test dataset containing labels (ground truth).
```bash
# valid path
Please enter the path to the CSV dataset.
If no valid path is provided the default dataset will be used: ./data/Demo_test_dataset.csv
```
```bash
# invalid path
Please enter the path to the CSV dataset.
If no valid path is provided the default dataset will be used: Demo_test_dataset
```

Once the input data is passed to the system, the script will load the model and perform inference on the report/s provided. After execution, the script will display the predicted class for each input report along with its confidence score. The script will display the results in one of the two formats shown below:
If a single record was selected:

![image](https://github.com/hytting/Product-defect/assets/93244335/8576bc56-9043-40b3-82f3-22c6832d4b02)

If a batch of records was selected:

![image](https://github.com/hytting/Product-defect/assets/93244335/89da3989-a0f0-4fbf-8e63-5d94b104e1e1)

The results are also saved in a csv file with the name provided by the user in the “results” folder.

We have run and stored the test results of three different inference use-cases in the “results” subfolder of the repository for reference. The use cases include:
1. Inference on a single report as illustrated in step 4a. The results are saved in the file “Inference_1_1.csv” in the “results” folder.
2a. Inference on a batch of reports with an invalid path as illustrated in step 4b invalid path. The default path to <./data/Demo_test_dataset.csv> is used. The results are saved in the file “Inference_2a_1.csv” in the “results” folder.
2b. Inference on a batch of reports with a valid path as illustrated in step 4b valid path. The results are saved in the file “Inference_2b_1.csv” in the “results” folder.


## Interpretability analysis

To run the script “Demo_Shap.py” for interpretability analysis follow these steps:
1. Open an anaconda terminal, activate the Demo environment that you have created, and navigate to the Demo directory
2. Execute the script “Demo_Shap.py” by typing the following command:
```bash
python Demo_Shap.py
```
3.The script will ask whether the input is a single report or a batch of reports. Enter <T> for single or <F> for batch:
```bash
Is the test data a single case (T/F)?
If F is selected a csv file with a batch os cases is expected: 
```
5. Next the script will ask for the name of the output file. Provide a name without blank spaces and without extension, as shown below. By default, the output file will be saved in the “results” directory with html format.
```bash
Please enter the name of the output shap textplot file (without blank spaces):
```
4a. If a single report is selected in 2, the script will sequentially ask for the Title and the Description of the report. Copy and paste the Title and Description of the report as shown below:
```bash
Please provide the case Title: 
```
```bash
Please provide the case Description: 
```
4b. If a batch of reports is selected in 2, the script will ask for the path to the csv file containing the batch of the reports. Enter a valid path such as <./data/Demo_test_dataset_without_labels.csv> as shown below. If an invalid path is entered, the script will use the default path <./data/Demo_test_dataset.csv> to the test dataset containing labels (ground truth).
```bash
# valid path
Please enter the path to the CSV dataset.
If no valid path is provided the default dataset will be used: ./data/Demo_test_dataset.csv
```
```bash
# invalid path
Please enter the path to the CSV dataset.
If no valid path is provided the default dataset will be used: Demo_test_dataset
```

Once the input data is passed to the system, the script will load the model and perform interpretability analysis on the report(s) provided. After execution, the script will save the results in an interactive html file with the name provided by the user in “results” folder. Open the html file to review the results of the interpretability analysis, as shown below:
- If a single record was selected:
<img width="1271" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/72fb7b31-1296-4b25-ac2b-a6daf5e46543">


- If a batch of records was selected:
<img width="1261" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/ecbabbdb-5a43-4631-957d-f9724e4b2d8b">



By default, each input record will highlight the highest confidence class, and also shows case title + description. The gradient map and the force plot for the class with the confidence for each record will be shown when related label is chosen by click. You can view the gradient map and the force plot for other classes with lower confidence by hovering over the list of possible classes and selecting the desired one.

We have run and stored the test results of three different interpretability analysis use-cases in the “results” subfolder of the repository for reference. The use cases include:
1. Interpretability on a single report as illustrated in step 4a. The results are saved in the file “Shap_1_1.csv” in the “results” folder.
   2a. Interpretability on a batch of reports with an invalid path as illustrated in step 4b invalid path. The default path to <./data/Demo_test_dataset.csv> is used. The results are saved in the file “Shap_2a_1.csv” in the “results” folder.
  2b. Interpretability on a batch of reports with a valid path as illustrated in step 4b valid path. The results are saved in the file “Shap_2b_1.csv” in the “results” folder.

