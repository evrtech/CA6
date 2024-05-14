# A sample ML classification problem

Get data from the cloud using `dvc pull data_raw.csv`, It must be uploaded by you or group member in the past.  

Process the data by converting the non-numeric data to numeric

`python process_data.py`

Create a training script for ML, This script will run on the pulled data. 

`python train.py`

# Lets practice different ML methods for improving the model

In this part we want to see the effect of the new method by comparing the results across different branches.

## DVC pipeline 

DVC pipeline provides a way to achieve it easily. Lets create a stage of dvc

`dvc stage add  -n get_data -d get_data.py -o data_raw.csv --no-exec python get_data.py`

Create rest of the stages manualy named, "process" and "train" and run it 

`dvc repro`

Commit the code to github to store the snapshot of the current development.

Lets add github actions to the repro to atuomate the above process. 




