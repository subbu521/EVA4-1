# S5 Step 1

* Target: To make the model lighter
* Results
* Parameters: 10,790
* Best Train Accuracy: 99.00
* Best Test Accuracy: 98.72
* Analysis
* The model performing good
* The model is not overfitting

# S5 Step 2

* Target: Add Batch-norm to increase model efficiency.
* Results
* Parameters: 10,970
* Best Train Accuracy: 99.76
* Best Test Accuracy: 99.18
* Analysis
* The model overfitting

# S5 Step 3

* Target: Add Dropouts.
* Results
* Parameters: 10,970
* Best Train Accuracy: 99.27
* Best Test Accuracy: 99.17
* Analysis
* After adding dropouts, The overfitting is reduced
* The dorpout rate is 0.25. I shoudl reduce the dropout rate


# S5 Step 4

* Target: Add GAP and remove the last BIG kernel.
* Results
* Parameters: 8,212
* Best Train Accuracy: 99.32
* Best Test Accuracy: 99.25
* Analysis
* After adding the GAP and removing the last BIG kernel the perameters are reduced to 8,212
* The acuracy is increased
* THe dropout rate is reduced to 0.15 which helps to reduce the overfitting. 


# S5 Step 5

* Target: .
* Results
* Parameters: 6,070
* Best Train Accuracy: 98.58
* Best Test Accuracy: 97.77
* Analysis
* After 



## Team
K Bhargava Kiran 
bhargav.kiran@gmail.com
M V Subbarao
subbu.521@gmail.com
