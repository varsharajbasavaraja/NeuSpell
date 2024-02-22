
# **Enhancing Language Precision Using Neural Context-Aware Spelling Correction Toolkit**

## Steps to implement NeuSpell: A Neural Spelling Correction Toolkit

###### Note: Entire Implementation is done using Google Colab where each step is explained in detail and along with code and test cases, this is available in Jupyter notebook named ```ECE570NeuSpell.ipynb```,  the notebook can be used for in detail Implementation instructions.

###  Contents
---
1. Installation
   
    1.1 Cloning the NeuSpell GitHub repository by mounting Google Drive
    
    1.2 Installing extra requirements and Resolving Installation Errors by changing the torch version
2. Download Checkpoints
3. Download Datasets
4. Implementation
  
    4.1 Defining the listed Models in NeuSpell

    4.2 Installing Correctors

      4.2.1 SC-LSTM Corrector

      4.2.2 Nested-LSTM Corrector
    
      4.2.3 CNN-LSTM Corrector
    
      4.2.4 BERT Corrector
 
5. Commmand line Interface
  
    5.1 SC-LSTM Checker

    5.2 Nested-LSTM Checker

    5.3 CNN-LSTM Checker

    5.4 BERT Checker

 6. Testing the Neuspell Corrector Modules SclstmChecker, NestedlstmChecker, CnnlstmChecker, BertChecker (JFLEG) dataset 

 7. Performance of Neuspell models (Discussion and Conclusion)


# 1. Installation

### 1.1 Cloning the NeuSpell GitHub repository by mounting Google Drive

```bash
from google.colab import drive
drive.mount('/content/drive')
root = '/content/drive/MyDrive/Colab Notebooks/ECE570/Varsha'
import os
if not os.path.isdir(root):
  os.mkdir(root)
os.chdir(root)
print(f'\nChanged CWD to "{root}"')

```

```bash
# Clone repository and pull latest changes.
!git clone https://github.com/neuspell/neuspell

# Change to the cloned directory
%cd neuspell

# Install the package using pip
!pip install -e .
```
### 1.2 Installing extra requirements and Resolving Installation Errors by changing torch version

```bash
!python -m pip install --upgrade pip
```
```bash
!mkdir -p ~/.pip
!echo "[global]" > ~/.pip/pip.conf
!echo "root_user_action = ignore" >> ~/.pip/pip.conf
```
Install the torch version to be compatible with all the depencies (Restart runtime if needed)
```bash
!pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 torchdata==0.3.0 torchtext==0.1.1
!pip install allennlp
```
 spacy models can be downloaded as:

```bash
python -m spacy download en_core_web_sm
```

# 2. Download Datasets

1. Download data from Google Drive and keep in this folder ```neuspell/data/traintest``` https://drive.google.com/drive/folders/1ejKSkiHNOlupxXVDMg67rPdqwowsTq1i

2. Alternatively, run code ECE570NeuSpell.ipynb under Download Datasets section to download all the datasets


# 3. Download Checkpoints
Download the required models of neuspell from the Google Drive link below
https://drive.google.com/drive/folders/1jgNpYe4TVSF4mMBVtFh4QfB2GovNPdh7?usp=sharing

Extract it and place it in the “neuspell/data/checkpoints” folder.
Make sure to replace any existing model files with the new one.

You can find the name of model and checkpoint to download from the drive in the following table

---


| Spell Checker                       | Class               | Checkpoint name             | Disk space (approx.) |
|-------------------------------------|---------------------|-----------------------------|----------------------|
| ```CNN-LSTM```                      | `CnnlstmChecker`    | 'cnn-lstm-probwordnoise'    | 450 MB               |
| ```SC-LSTM```                       | `SclstmChecker`     | 'scrnn-probwordnoise'       | 450 MB               |
| ```Nested-LSTM```                   | `NestedlstmChecker` | 'lstm-lstm-probwordnoise'   | 455 MB               |
| ```BERT```                          | `BertChecker`       | 'subwordbert-probwordnoise' | 740 MB               |
| ```SC-LSTM plus ELMO (at input)```  | `ElmosclstmChecker` | 'elmoscrnn-probwordnoise'   | 840 MB               |
| ```SC-LSTM plus BERT (at input)```  | `BertsclstmChecker` | 'bertscrnn-probwordnoise'   | 900 MB               |
| ```SC-LSTM plus BERT (at output)``` | `SclstmbertChecker` | 'scrnnbert-probwordnoise'   | 1.19 GB              |
| ```SC-LSTM plus ELMO (at output)``` | `SclstmelmoChecker` | 'scrnnelmo-probwordnoise'   | 1.23 GB              |


---
Note: Each Model will take 5-10mins to upload depending on the size


# 4. Implementation

## List of neural models implemented:

- [```SC-LSTM```](https://drive.google.com/file/d/1OvbkdBXawnefQF1d-tUrd9lxiAH1ULtr/view?usp=sharing)
It corrects misspelt words using semi-character representations, fed through a bi-LSTM network. The semi-character representations are a concatenation of one-hot embeddings for the (i) first, (ii) last, and (iii) bag of internal characters
- [```Nested-LSTM```](https://drive.google.com/file/d/19ZhWvBaZqrsP5cGqBJdFPtufdyBqQprI/view?usp=sharing)
The model builds word representations by passing its individual characters to a bi-LSTM. These representations are further fed to another bi-LSTM trained to predict the correction
- [```CNN-LSTM```](https://drive.google.com/file/d/14XiDY4BJ144fVGE2cfWfwyjnMwBcwhNa/view?usp=sharing)
Similar to the previous model, this model builds word-level representations from individual characters using a convolutional network.
- [```BERT```](https://huggingface.co/transformers/bertology.html)
The model uses a pre-trained transformer network. We average the sub-word representations to obtain the word representations, which are further fed to a classifier to predict its correction

## 4.1 Defining the listed Models in NeuSpell

1. Hidden size of the bi-LSTM network in all models is set to to 512 and  {50,100,100,100} sized convolution filters with lengths {2,3,4,5} respectively are used in CNNs. 
2. A dropout of 0.4 is used on the bi-LSTM’s outputs and modes are trained using cross-entropy loss.
3. The BertAdam5 optimizer is for models with a BERT component and Adam (Kingma and Ba, 2014) optimizer is used for the remainder. These optimizers are used with default parameter settings. 
4. A batch size of 32 examples is used, and train with a patience of 3 epoch

## 4.2 Installing Correctors
Defining class Corrector

  ### 4.2.1 SC-LSTM Corrector
  Definining class SclstmChecker
  ### 4.2.2 Nested-LSTM Corrector
  Definining class NestedlstmChecker
  ### 4.2.3 CNN-LSTM Corrector
  Definining class CnnlstmChecker
  ### 4.2.4 BERT Corrector
  Definining class BertChecker

Run the code snippets in corresponding sections in ECE570NeuSpell.ipynb to define corrector classes

# 5. Commmand line Interface
## Loading each Neural Model and Testing for context-aware spelling correction via a unified command line

## 5.1 SC-LSTM Checker
Select and load a spell checker as follows:

```python
from neuspell import SclstmChecker

checker_sclstm = SclstmChecker(n_epochs=5)
checker_sclstm.from_pretrained()
```

## 5.2 Nested-LSTM Checker
Select and load a spell checker as follows:

```python
from neuspell import NestedlstmChecker

checker_nestedlstm = NestedlstmChecker()
checker_nestedlstm.from_pretrained()
```

## 5.3 CNN-LSTM Checker
Select and load a spell checker as follows:

```python
from neuspell import CnnlstmChecker

checker_cnnlstm = CnnlstmChecker()
checker_cnnlstm.from_pretrained()
```

## 5.3 BERT Checker
Select and load a spell checker as follows:

```python
from neuspell import BertChecker

checker_bert = BertChecker()
checker_bert.from_pretrained()
```

## The following test cases can be run across all the checkers using the snippets to create a command line interface

change the name of the checker based on what model needs to be tested (an example is shown for BERT Checker)


 ### 1. Test for rectification of misspellings in individual sentences

```python
checker_bert.correct("This luks like a gud project")
```
```
Output: This looks like a good project
```
 ### 2. Test for rectification of misspellings in sentences with homophones
```python
checker_bert.correct_strings(["I feel week this hole week. I think i mite get fiver"])
```
```
Output: ['I feel weak this whole week . I think I might get fever']
```

 ### 3. Test for rectification of misspellings in a list of sentences
```python
checker_bert.correct_strings(["Thee wors are often used together. You can go to the defition of spellig or the defintion of mistae. Or, see other combintions with mistke.", ])
```
```
Output: ['These words are often used together. You can go to the definition of spelling or the definition of mistake . Or , see other combinations with mistake .']
```
 ### 4. Test for rectification of misspellings in a file

```python
checker_bert.correct_from_file(src=f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt",
                              dest=f"{TRAIN_TEST_DATA_PATH}/sample_prediction.txt")
checker_bert.evaluate(f"{TRAIN_TEST_DATA_PATH}/sample_clean.txt", f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt")
```
```
Output: 
###############################################
data size: 41
11it [00:18,  1.64s/it]
total inference time for this data is: 18.065975 secs
saving results at: /content/drive/MyDrive/Colab Notebooks/ECE570/Varsha/neuspell/neuspell/../data/traintest/sample_prediction.txt
 /content/drive/MyDrive/Colab Notebooks/ECE570/Varsha/neuspell/neuspell/../data/traintest/sample_clean.txt /content/drive/MyDrive/Colab Notebooks/ECE570/Varsha/neuspell/neuspell/../data/traintest/sample_corrupt.txt
41it [00:00, 60423.92it/s]
41it [00:00, 290975.40it/s]
loaded tuples of (corr,incorr) examples from 
###############################################
data size: 41
11it [00:16,  1.47s/it]
Epoch None valid_loss: 0.13959307433106005
total inference time for this data is: 16.204485 secs
###############################################
total token count: 811
_corr2corr:754, _corr2incorr:6, _incorr2corr:36, _incorr2incorr:15
accuracy is 0.9741060419235512
word correction rate is 0.7058823529411765
###############################################
```

### 5. Evaluation of SC-LSTM Checker with BEA-322 dataset

```python
checker_bert.evaluate(clean_file=f"{TRAIN_TEST_DATA_PATH}/test.bea322", corrupt_file=f"{TRAIN_TEST_DATA_PATH}/test.bea322.noise")
```
```
Output: 
 /content/drive/MyDrive/Colab Notebooks/ECE570/Varsha/neuspell/neuspell/../data/traintest/test.bea322 /content/drive/MyDrive/Colab Notebooks/ECE570/Varsha/neuspell/neuspell/../data/traintest/test.bea322.noise
322it [00:00, 299526.70it/s]
322it [00:00, 752404.39it/s]
loaded tuples of (corr,incorr) examples from 
###############################################
data size: 322
81it [01:29,  1.10s/it]
Epoch None valid_loss: 0.5250554458226686
total inference time for this data is: 89.023216 secs
###############################################
total token count: 5432
_corr2corr:4941, _corr2incorr:168, _incorr2corr:213, _incorr2incorr:110
accuracy is 0.948821796759941
word correction rate is 0.6594427244582043
###############################################
```

# 6. Testing the Neuspell Corrector Modules SclstmChecker, NestedlstmChecker, CnnlstmChecker, BertChecker across three (Synthetic, Natural and Ambiguous) different datasets

6.1  Synthetic(WORD-TEST): A word is swapped with its noised counterpart from a prebuilt lookup table. 109K misspelled - correct word pairs were collected for 17K popular English words from various public sources.

6.2 Natural (JFLEG): The JHU FLuency-Extended GUG Corpus (JFLEG) dataset is a collection of essays written by English learners with different first languages. This dataset contains 2K spelling mistakes (6.1% of all tokens) in 1601 sentences.

6.3 Ambiguous (BEA-322): Manually prune down the list to 322 sentences, with one ambiguous mistake per sentence.

The test cases are run in the notebook section 6

# 7. Performance of Neuspell models

 Performance of the implemented four different spelling correction models—CNN-LSTM, BERT, SC-LSTM, and Nested-LSTM was assessed across three diverse datasets: Synthetic(WORD-TEST), Natural(JFLEG), and Ambiguous(BEA-322).


| **Model**               | **Dataset**          | **Accuracy** | **Word Correction Rate** | **Time per Sentence (ms)** |
|-------------------------|----------------------|--------------|---------------------------|-----------------------------|
| CNN-LSTM                | Synthetic(WORD-TEST) | 96.99%       | 88.02%                    | 5.0                         |
| BERT                    | Synthetic(WORD-TEST) | 98.84%       | 95.27%                    | 7.8                         |
| SC-LSTM                 | Synthetic(WORD-TEST) | 97.56%       | 90.45%                    | 4.7                         |
| Nested-LSTM             | Synthetic(WORD-TEST) | 97.80%       | 91.14%                    | 6.1                         |
| CNN-LSTM                | Natural (JFLEG)       | 97.56%       | 80.11%                    | 6.2                         |
| BERT                    | Natural (JFLEG)       | 97.86%       | 84.98%                    | 13.3                        |
| SC-LSTM                 | Natural (JFLEG)       | 97.90%       | 81.92%                    | 6.0                         |
| Nested-LSTM             | Natural (JFLEG)        | 97.74%       | 81.58%                    | 6.9                         |
| CNN-LSTM                | Ambiguous (BEA-322)   | 90.76%       | 57.28%                    | 6.0                         |
| BERT                    | Ambiguous (BEA-322)    | 92.52%       | 71.83%                    | 8.1                         |
| SC-LSTM                 | Ambiguous (BEA-322)    | 94.88%       | 65.94%                    | 5.5                         |
| Nested-LSTM             | Ambiguous (BEA-322)   | 91.37%       | 63.16%                    | 5.4                         |


## Discussion

1. Accuracy: BERT consistently demonstrated the highest accuracy across all datasets, outperforming other models. SC-LSTM and Nested-LSTM also showcased competitive accuracy.
2. Word Correction Rate: BERT exhibited superior word correction rates, emphasizing its effectiveness in correcting spelling errors. SC-LSTM and Nested-LSTM demonstrated commendable performance as well.
3. Time per Sentence: The evaluation also considered the time each model took per sentence. SC-LSTM demonstrated efficiency, especially on Synthetic(WORD-TEST) and Ambiguous BEA-322 datasets.


## Conclusion

BERT excelled in accuracy and word correction rate, while SC-LSTM showcased efficiency in terms of time per sentence. The choice of a model should be context-dependent, considering the specific requirements of the application.


Switch to ```ECE570NeuSpell.ipynb``` notebook for in detail Implementaion r in detail Implementaion instructions.
