#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DefaultDataCollator
from datasets import Dataset
import shap
import re
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # uncomment this line if GPU is not available

from utils import ID2LABEL, LABEL2ID, PATTERN

class Demo:
    
    """
    Class to demonstrate how to use interpretability tool of the MedDefects-BERT model.
    The class includes functions to:
    1. Load the model and associated artifacts
    2. Load test dataset from a directory or by direct input into the command line
    3. Pre-process the data
    4. Generate Shapley textplots from the input data, and save the results as a html file in a local directory
    """
    
    def __init__(self,
                 path = None,
                ):
        
        self.path = path
        self.id2label = ID2LABEL
        self.label2id = LABEL2ID
        self.checkpoint = "./models/MedDefects-BERT"
        
        self.tokenier = None
        self.model = None
        self.data_collator = None
        self.labels = None
        
        self.df = None
        
    ## function to load tokenizer, model, and data collator
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            id2label = self.id2label,
            label2id = self.label2id)
        self.data_collator = DefaultDataCollator(return_tensors="tf")
        self.labels = sorted(self.model.config.label2id, key = self.model.config.label2id.get)
        print('Model is loaded')
    
    ## function to load dataset as pandas dataframe
    def load_dataset(self, input_text: list = None):
        if not input_text:
            if not self.path:
                self.path = './data/Demo_test_dataset.csv'
            self.df = pd.read_csv(self.path)
            self.df['text'] = self.df['Title'] + ". " + self.df['Desc']
        else:
            self.df = pd.DataFrame({'Title': [input_text[0]], 'Desc': [input_text[1]]})
            self.df['text'] = self.df['Title'] + ". " + self.df['Desc']
        print('Dataset is loaded')
            
    ## function to pre-process data
    def pre_process(self):
        ## remove sentences with irrelevant information
        pattern = PATTERN
        self.df = self.df.assign(text = lambda x: x['text'].str.replace(pattern, "", case = False, regex = True))
        print('Pre-processing done')

    ## function to generate shap textplot
    def get_shap_values(self, output_name):
        def f(x):
            df = pd.DataFrame(x, columns = ['text'])
            ds = Dataset.from_pandas(df)
            ds = ds.map(lambda x: self.tokenizer(x['text'], max_length = 512, padding = 'max_length', truncation = True), batched = True)

            X = ds.to_tf_dataset(
                columns = ['attention_mask', 'input_ids', 'token_type_ids'],
                shuffle = False,
                collate_fn = self.data_collator,
                batch_size = 4)

            logits = self.model.predict(X).logits
            scores = tf.nn.softmax(logits, axis = -1)
            return scores
        
        explainer = shap.Explainer(f, self.tokenizer, output_names = self.labels)
        ds_test  = Dataset.from_pandas(self.df)
        print('Generating  SHAP textplot...')
        shap_value = explainer(ds_test[:])
        
        with open(f'./results/{output_name}.html','w',encoding = 'utf-8') as file:
            file.write(shap.plots.text(shap_value,display = False))
        print(f'Shap textplot is saved in the results directory as {output_name}.html')
    
def main():
    
    """
    Function to perform interpretability of a test dataset.
    The function includes the following steps:
    
    1. Check whether the input data is a single report or a batch
    2. Request for the name of the output file to save the results
    3.a If the input data is a single report, request the report Title and Description from user
    3.b If the input data is a batch of reports, request for the path to the csv file containing the reports
    4. Instantiate the Inference Class by creating an object of the Class
    5. Load input data and apply pre-processing on text
    6. Load model and associated artifacts
    7. Generate shap textplots and save the results
    """

    single_case = input('Is the test data a single case (T/F)?\nIf F is selected a csv file with a batch os cases is expected: ')
    if single_case in ['T','t']:
        output_name = input('Please enter the name of the output shap textplot file (without blank spaces): ')
        title = input('Please provide the case Title: ')
        desc = input('Please provide the case Description: ')
        Demo_shap = Demo()
        Demo_shap.load_model()
        Demo_shap.load_dataset([title, desc])
        
    elif single_case in ['F', 'f']:
        output_name = input('Please enter the name of the output shap textplot file (without blank spaces): ')
        path = input('Please enter the path to the CSV dataset.\nIf no valid path is provided the default dataset will be used: ')
        if os.path.isfile(path):
            Demo_shap = Demo(path)
        else:
            print('Invalid path, the default dataset will be loaded')
            Demo_shap = Demo()
        Demo_shap.load_model()
        Demo_shap.load_dataset()
        
    else:
        print('Invalid input, exiting the app')
        return
    Demo_shap.pre_process()
    Demo_shap.get_shap_values(output_name)
    
if __name__ == '__main__':
    main()