#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import tensorflow as tf
from pickle import load
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification
from datasets import Dataset
import re
from IPython.display import display
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # uncomment this line if GPU is not available

## format setting for result
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', None)

from utils import ID2LABEL, LABEL2ID, PATTERN

class Demo_pred:
    
    """
    Class to demonstrate how to use MedDefects-BERT in inference mode.
    The class includes functions to:
    1. Load the model and associated artifacts
    2. Load test dataset from a directory or by direct input into the command line
    3. Pre-process the data
    4. Predict MedDRA labels of the test dataset, display the results at the endpoint, and save the results as a csv file in a local directory
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
        
        self.df = None
    
    ## function to load tokenizer, model, and data collator
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.checkpoint)
        self.data_collator = DefaultDataCollator(return_tensors="tf")
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
        if 'MedDRA' in self.df.columns:
            self.df['labels'] = self.df['MedDRA'].map(self.label2id)
        print('Pre-processing done')
   
    ## function to predict MedDRA labels of test dataset
    def model_predict(self, output_name):
        ds  = Dataset.from_pandas(self.df)
        
        ## data embedding
        ds = ds.map(lambda x: self.tokenizer(x['text'], padding="max_length", truncation=True), batched=True)
        if 'MedDRA' in self.df.columns:
            X = ds.to_tf_dataset(
                columns=["attention_mask", "input_ids", "token_type_ids"],
                label_cols=["labels"],
                shuffle=False,
                collate_fn=self.data_collator,
                batch_size=4)
        else:
            X = ds.to_tf_dataset(
                columns=["attention_mask", "input_ids", "token_type_ids"],
                shuffle=False,
                collate_fn=self.data_collator,
                batch_size=4,    
            )
        ## model prediction
        print('Generating prediction...')
        logits = self.model.predict(X)
        y_score = tf.nn.softmax(logits[0], axis=-1)
        y_prob = np.max(y_score, axis = 1)
        y_pred = np.argmax(y_score, axis = 1)

        if 'MedDRA' in self.df.columns:
            results = pd.DataFrame({
                'Predicted Class': np.vectorize(self.id2label.get)(y_pred),
                'Confidence': y_prob,
                'Real Class': self.df['MedDRA']
            })
        else:
            results = pd.DataFrame({
                'Predicted Class': np.vectorize(self.id2label.get)(y_pred),
                'Confidence': y_prob,
            })
        
        display(results)
        
        ## results saving
        results.to_csv(f'./results/{output_name}.csv',index = False)
        print(f'Inference results are saved in the results directory as {output_name}.csv')
        
def main():
    
    """
    Function to perform inference of a test dataset.
    The function includes the following steps:
    
    1. Check whether the input data is a single report or a batch
    2. Request for the name of the output file to save the results
    3a. If the input data is a single report, request the report Title and Description from user
    3b. If the input data is a batch of reports, request for the path to the csv file containing the reports
    4. Instantiate the Inference Class by creating an object of the Class
    5. Load input data and apply pre-processing on text
    6. Load model and associated artifacts
    7. Generate model predictions, display and save the results
    """
    
    single_case = input('Is the test data a single case (T/F)?\nIf F is selected a csv file with a batch of cases is expected: ')
    if single_case in ['T','t']:
        output_name = input('Please enter the name of the output prediction file (without blank spaces): ')
        title = input('Please provide the case Title: ')
        desc = input('Please provide the case Description: ')
        Demo_pred_val = Demo_pred()
        Demo_pred_val.load_model()
        Demo_pred_val.load_dataset([title, desc])
        
    elif single_case in ['F', 'f']:
        output_name = input('Please enter the name of the output prediction file (without blank spaces): ')
        path = input('Please enter the path to the CSV dataset.\nIf no valid path is provided the default dataset will be used: ')
        if os.path.isfile(path):
            Demo_pred_val = Demo_pred(path)
        else:
            print('Invalid path, the default dataset will be loaded')
            Demo_pred_val = Demo_pred()
        Demo_pred_val.load_model()
        Demo_pred_val.load_dataset()
    else:
        print('Invalid input, exiting the app')
        return

    Demo_pred_val.pre_process()
    Demo_pred_val.model_predict(output_name)    
    
if __name__ == '__main__':
    main()
