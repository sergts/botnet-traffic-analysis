## Python dependencies
Check that you have python dependencies installed listed in `requirements.txt` file

## Download the dataset
Use `download_data.py` script

Then extract rar files into respective `{device_name}/gafgyt_attack/` and `{device_name}/mirai_attack/` directories

## Run the training and evaluation script
`train.py`

You can specify if you just want to use just top N features

`train.py 5`


## Run just the test for the existing model
As input give it the top number of features to use and the name of the model to load

`test.py 5 'model_5.h5'`


---------------
## Results


For all features
Trained for 20 epochs for 42 mins.
Results are

Model evaluation
Loss, accuracy
[0.0010332345978360195, 0.9998208877454652]

SO Accuracy on test set is 0.99982


Confusion matrix
Benign     Gafgyt     Mirai
[[111369    114      2]
 [    34 567253      7]
 [     9     87 733647]]


For top 5 features, trained for 5 epochs
Loss, accuracy 
[0.004696070021679117, 0.9991879772492039]

Confusion matrix
Benign     Gafgyt     Mirai
[[111436     40      9]
 [   647 566647      0]
 [    63    388 733292]]


For top 3 features, trained for 5 epochs, 99s per epoch
Loss, accuracy 
[0.0032474066202494442, 0.9994704507257232]

Confusion matrix
benign  gafgyt  mirai
[[111439     40      6]
 [   460 566834      0]
 [    80    162 733501]]


For top 2 features, trained for 5 epochs, 99s per epoch
Loss                   Accuracy
[0.33539704368039586, 0.8430219139947991]
Confusion matrix
benign  gafgyt  mirai
[[ 58989  52297    199]
 [   523 436574 130197]
 [   486  38033 695224]]



