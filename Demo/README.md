## Demonstration of the Software

The following guide contains instructions to install and test the MedDefects-BERT software developed in this study.

The Demo contains two scripts:
1. “Demo_Inference.py” takes as input a single report or a batch of reports in a dataset (with or without known labels) and outputs the predicted label(s) and their corresponding confidence scores.
2. “Demo_Shap.py” takes as input a single report or a batch of reports in a dataset (with or without known labels) and outputs the Shapley values of the text in a gradient map and a force plot as described in the Methods section and illustrated in Figure 3 of the paper.

Follow the provided instructions step by step in order to reproduce the provided test results and test the software with your own dataset.

## Installation
The following Demo can be run on a Windows 7 and above Operating System. The Demo runs faster in a GPU enabled machine. 
However, it can be run on a machine without GPU. For CPU-only machines, uncomment line 14 of both scripts to disable GPU.

Before running the demo ensure that the following applications are installed:
-	Python 3.7.15 or above
-	Windows 7 or above
-	Anaconda 2.3.1 or above

### Installation steps:

1. Open an anaconda prompt terminal, create an environment for this Demo with python version 3.7.15, and install the packages specified in the requirements.txt file.
2. Create a folder in your local machine where the code, the model weights, and the test datasets of this Demo will be copied.
3. Navigate to the Demo folder and clone this repository or copy the scripts “Demo_Inference.py”  and “Demo_Shape.py” in the working directory.
4. Create a subfolder named “models” in the Demo directory and download on it the weights of the MedDefects-BERT model from this [link](https://drive.google.com/drive/folders/1wqiBd_-5pn3tRm5W27kZlB9wztk41F5U?usp=drive_link). Unzip the file inside the models subfolder. The model weights and configuration will be located in a subfolder named “MedDefects-BERT” in the “models” folder.
5. If the repository is not cloned, create a subfolder named “data” and copy on it the csv files “Demo_test_dataset.csv” and “Demo_test_dataset_without_labels.csv” containing the test datasets. If the repository is cloned, the folder with the datasets will be automatically created.
6. If the repository is not cloned, create a subfolder named “results” and (optionally) copy on it the csv files from the results.rar file located in the "results" folder of the repository. If the repository is cloned, the folder with the datasets will be automatically created. This folder contains the results of our test runs as reference. Additional test runs will store the results in this folder, with a user provided output file name.


### Notes about the csv files

We provide two csv files for testing the model:
1. “Demo_test_dataset.csv” contains 3 columns. Columns "Title" and "Desc" contain the title of the report and description of the report in text format, respectively. Column MedDRA contains the annotated label of the report (ground truth) also in text format.
2. “Demo_test_dataset_without_labels.csv” is identical to the first one but does not contain the column "MedDRA".

If you want to test your own dataset, you need to provide your reports in the same format with the same column names. Columns "Title" and "Desc" are mandatory as they are required to generate the input for the model. Column "MedDRA" is optional. If present, this column is displayed along with the model predictions for reference. If not present, only the model predictions are displayed.


## Inference

To run the script “Demo_Inference.py” for inference follow these steps:
1. Open an anaconda terminal, activate the Demo environment that you have created, and navigate to the Demo directory
2. Execute the script “Demo_Inference.py” by typing the following command:
```bash
python Demo_Inference.py
```
3. The script will ask whether the input is a single report or a batch of reports. Enter "T" for single or "F" for batch:

![image](https://github.com/hytting/Product-defect/assets/93244335/f5c29b2c-fe93-4183-bb04-6150c86614f1)


5. Next, the script will ask for the name of the output file. Provide a name without blank spaces and without extension, as shown below. By default, the output file will be saved in the “results” directory with csv format.
<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/581deba3-3d40-4a13-8e27-12b08391758b">


- 4a. If a single report is selected in 2, the script will sequentially ask for the Title and the Description of the report. Copy and paste the Title and Description of the report as shown below:
<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/5066c7dd-ee2c-4458-a75f-4e64eea16188">

```bash
The Therapeutic Goods Administration (TGA) has tested a product labelled The Rock and found that it contains the undeclared substances sulfosildenafil and hydroxyhomothiosildenafil an analogue of sildenafil.
```

<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/f384f8bc-7e83-4489-9831-712dc988b60c">

```bash
Safety advisory TGA has tested a product labelled The Rock and found that: ?€?it contains the undeclared substances sulfosildenafil and hydroxyhomothiosildenafil an analogue of sildenafil. ?€?consumers are advised that both hydroxyhomothiosildenafil and sulfosildenafil are prescription-only medicines. The supply of The Rock capsules is illegal. The Rock capsules have not been assessed by the TGA for quality, safety or efficacy as required under Australian legislation, and the place of manufacture is not approved by the TGA. TGA investigations have shown that a number of people in Australia have bought the product online.
```

- 4b. If a batch of reports is selected in 2, the script will ask for the path to the csv file containing the batch of the reports. Enter a valid path such as <./data/Demo_test_dataset_without_labels.csv> as shown below. If an invalid path is entered, the script will use the default path <./data/Demo_test_dataset.csv> to the test dataset containing labels (ground truth).

```bash
# valid path
./data/Demo_test_dataset.csv
```

<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/9ea945d6-76e6-4b8f-afae-0ead6525efd1">

```bash
# invalid path
Demo_test_dataset
```

<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/eb18e87d-a8b7-4d4c-af0e-d277fe6ba626">


5. Output:
Once the input data is passed to the system, the script will load the model and perform inference on the report/s provided. After execution, the script will display the predicted class for each input report along with its confidence score. The script will display the results in one of the two formats shown below:

- If a single record was selected:

![image](https://github.com/hytting/Product-defect/assets/93244335/8576bc56-9043-40b3-82f3-22c6832d4b02)

- If a batch of records was selected:

![image](https://github.com/hytting/Product-defect/assets/93244335/89da3989-a0f0-4fbf-8e63-5d94b104e1e1)



The results are also saved in a csv file with the name provided by the user in the “results” folder.

We have run and stored the test results of three different inference use-cases in the “results” subfolder of the repository for reference. The use cases include:
- “Inference_1_1.csv”: Inference on a single report as illustrated in step 4a. 
- “Inference_2a_1.csv”: Inference on a batch of reports with an invalid path as illustrated in step 4b invalid path. The default path to <./data/Demo_test_dataset.csv> is used. 
- “Inference_2b_1.csv”: Inference on a batch of reports with a valid path as illustrated in step 4b valid path. 


## Interpretability analysis

To run the script “Demo_Shap.py” for interpretability analysis follow these steps:
1. Open an anaconda terminal, activate the Demo environment that you have created, and navigate to the Demo directory
2. Execute the script “Demo_Shap.py” by typing the following command:
```bash
python Demo_Shap.py
```

3. The script will ask whether the input is a single report or a batch of reports. Enter "T" for single or "F" for batch:

![image](https://github.com/hytting/Product-defect/assets/93244335/89cd51ff-7fa6-4122-b2dc-cd85b2726cb6)


5. Next, the script will ask for the name of the output file. Provide a name without blank spaces and without extension, as shown below. By default, the output file will be saved in the “results” directory with html format.
<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/9d8a7585-dbee-4e27-80b7-e03f1aaec773">

- 4a. If a single report is selected in 2, the script will sequentially ask for the Title and the Description of the report. Copy and paste the Title and Description of the report as shown below:
<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/b7c5d82a-221a-4eca-821b-53cbee3c022e">

```bash
The Therapeutic Goods Administration (TGA) has tested a product labelled The Rock and found that it contains the undeclared substances sulfosildenafil and hydroxyhomothiosildenafil an analogue of sildenafil.
```
<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/fde8f06b-88cf-4df2-a837-2783b6556cee">

```bash
Safety advisory TGA has tested a product labelled The Rock and found that: ?€?it contains the undeclared substances sulfosildenafil and hydroxyhomothiosildenafil an analogue of sildenafil. ?€?consumers are advised that both hydroxyhomothiosildenafil and sulfosildenafil are prescription-only medicines. The supply of The Rock capsules is illegal. The Rock capsules have not been assessed by the TGA for quality, safety or efficacy as required under Australian legislation, and the place of manufacture is not approved by the TGA. TGA investigations have shown that a number of people in Australia have bought the product online.
```

- 4b. If a batch of reports is selected in 2, the script will ask for the path to the csv file containing the batch of the reports. Enter a valid path such as <./data/Demo_test_dataset_without_labels.csv> as shown below. If an invalid path is entered, the script will use the default path <./data/Demo_test_dataset.csv> to the test dataset containing labels (ground truth).

```bash
# valid path
 ./data/Demo_test_dataset.csv
```
<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/9670b62b-e207-4f09-9d42-745abffc104d">

```bash
# invalid path
Demo_test_dataset
```
<img width="2500" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/f20e85dd-ea17-49f3-8fb9-f1bec0653b2a">


5. Output:
Once the input data is passed to the system, the script will load the model and perform interpretability analysis on the report(s) provided. After execution, the script will save the results in an interactive html file with the name provided by the user in the “results” folder. Open the html file to review the results of the interpretability analysis, as shown below:
- If a single record was selected:
<img width="1277" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/2dfb8e9e-d032-4120-b9c0-e0c18a4fa58a">


- If a batch of records was selected:
<img width="1265" alt="image" src="https://github.com/hytting/Product-defect/assets/93244335/914d0b72-584b-4e28-9e91-642b32648c71">



By default, when the html file is opened, the tool will highlight the class label with the highest confidence score along with the plain text for each report (title plus description). To view the gradient map and the force plot for each report, hover the mouse over the list of class labels and click on the highlighted one. You can also view the gradient map and the force plot for other classes with lower confidence by hovering over the list of possible classes and selecting the desired one.

We have run and stored the test results of three different interpretability analysis use-cases in the “results” subfolder of the repository for reference. The use cases include:

- “Shap_1_1.html”: Interpretability on a single report as illustrated in step 4a.
- “Shap_2a_1.html”: Interpretability on a batch of reports with an invalid path as illustrated in step 4b invalid path. The default path to <./data/Demo_test_dataset.csv> is used.
- “Shap_2b_1.html”: Interpretability on a batch of reports with a valid path as illustrated in step 4b valid path.

