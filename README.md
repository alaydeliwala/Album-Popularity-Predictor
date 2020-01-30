# Album-Popularity-Predictor
A project for CSC 422 Automated Learning & Data Analysis at NC State.
## Introduction
We hoped to be able to predict an albums popularity on the year end Billboard top charts based on various acoustic features. Our models assumed an album was popular if the rank was <html> &le;<html/> 25  or not popular if the rank  was <html> &gt;<html/> 25.

In order to assess whether or not an album is popular, we utilized different machine learning models:
* Naive-Bayes
* Decision Tree (utilizing Information Gain and Entropy)
* Support Vector Machine
* Deep Neural Networks

## Dataset
* Full Album Data with Acoustic Features ([Link to Dataset](https://drive.google.com/open?id=1Hl2DEB99cL0VxdnPEudylAfoX3_lEVgh))

Created using data from:
  * [Acoustic and meta features of albums and songs on the Billboard 200](https://components.one/datasets/billboard-200/)
  * The Billboard Year End Top Albums List

*The first dataset was used for the acoustic features and the the Top Albums List was scraped for the album name*


## Traning & Testing
We performed a 70/30 training testing spilt and standardized the data
* [Link to Training Data](https://drive.google.com/open?id=1BqxIklysgCJXq1SJsX-Jlw98e2mg56wf)
* [Link to Testing Data](https://drive.google.com/open?id=1h8agdS6_3DLGEzrue7rsxR7Z8kr4A8Gz)

## Model Results

|            Model             | Accuracy |
| :--------------------------: | :------: |
| Naive Bayes Model (Gaussian) |  74.9%   |
|  Decision Tree Model (Gini)  |  86.5%   |
|          SVM Model           |  85.3 %  |
|      2-NN + 10-Fold CV       |  85.58%  |
|     Deep Neural Network      |  85.58%  |
|     Deep Neural Network      |          |

## Setup
### Auto Installation using pip!

1. Make sure you have installed virtualenv, or if not then run `pip3 install virtualenv`
2. Create the python three virtual environment `virtualenv venv`
3. Start the environment `source venv/bin/activate`
4. Automatically install all relevant dependencies using the following command `pip install -r requirements.txt`
### Download Testing and Training Data
Allow `dataset_download.sh` permission to execute by running
```shell
$ chmod +x dataset_download.sh
```
Download the data byt running
```shell
$ ./dataset_download.sh
```
The training and testing data should be available in `data/`

## Usage

In the root folder of the program run this command to start the virtual environment
```shell
$ source venv/bin/activate
```
After the virtual environment has started run this command to start the program
```shell
$ python models/decision_tree.py
$ python models/knn_model.py
$ python models/naive_bayes_model.py
$ python models/neural_net.py
$ python models/svm_model.py
```

## Additional Resources
[Data Science for Hit Song Prediction](https://towardsdatascience.com/data-science-for-hit-song-prediction-32370f0759c1)

[Song Popularity Predictor](https://towardsdatascience.com/song-popularity-predictor-1ef69735e380)


