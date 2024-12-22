# Naive Bayes with NBA Players Dataset

## Lets use Naive Bayes to classify players as "high scoring" "mid scorinng" or "low scoring"!
#### Samantha Morello

### Lets begin!

Import important libraries.


```python
#hw 1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder # for encoding categorical features from strings to number arrays
from sklearn.naive_bayes import MultinomialNB, CategoricalNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

```

### Read in and quick look at the data

We use data set all_seasons.csv, from Kaggle: https://www.kaggle.com/datasets 

Let's explore the data set!


```python
df = pd.read_csv('all_seasons.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12844 entries, 0 to 12843
    Data columns (total 22 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Unnamed: 0         12844 non-null  int64  
     1   player_name        12844 non-null  object 
     2   team_abbreviation  12844 non-null  object 
     3   age                12844 non-null  float64
     4   player_height      12844 non-null  float64
     5   player_weight      12844 non-null  float64
     6   college            12844 non-null  object 
     7   country            12844 non-null  object 
     8   draft_year         12844 non-null  object 
     9   draft_round        12844 non-null  object 
     10  draft_number       12844 non-null  object 
     11  gp                 12844 non-null  int64  
     12  pts                12844 non-null  float64
     13  reb                12844 non-null  float64
     14  ast                12844 non-null  float64
     15  net_rating         12844 non-null  float64
     16  oreb_pct           12844 non-null  float64
     17  dreb_pct           12844 non-null  float64
     18  usg_pct            12844 non-null  float64
     19  ts_pct             12844 non-null  float64
     20  ast_pct            12844 non-null  float64
     21  season             12844 non-null  object 
    dtypes: float64(12), int64(2), object(8)
    memory usage: 2.2+ MB



```python
print(df.shape)
df.head()
```

    (12844, 22)





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>player_name</th>
      <th>team_abbreviation</th>
      <th>age</th>
      <th>player_height</th>
      <th>player_weight</th>
      <th>college</th>
      <th>country</th>
      <th>draft_year</th>
      <th>draft_round</th>
      <th>...</th>
      <th>pts</th>
      <th>reb</th>
      <th>ast</th>
      <th>net_rating</th>
      <th>oreb_pct</th>
      <th>dreb_pct</th>
      <th>usg_pct</th>
      <th>ts_pct</th>
      <th>ast_pct</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Randy Livingston</td>
      <td>HOU</td>
      <td>22.0</td>
      <td>193.04</td>
      <td>94.800728</td>
      <td>Louisiana State</td>
      <td>USA</td>
      <td>1996</td>
      <td>2</td>
      <td>...</td>
      <td>3.9</td>
      <td>1.5</td>
      <td>2.4</td>
      <td>0.3</td>
      <td>0.042</td>
      <td>0.071</td>
      <td>0.169</td>
      <td>0.487</td>
      <td>0.248</td>
      <td>1996-97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Gaylon Nickerson</td>
      <td>WAS</td>
      <td>28.0</td>
      <td>190.50</td>
      <td>86.182480</td>
      <td>Northwestern Oklahoma</td>
      <td>USA</td>
      <td>1994</td>
      <td>2</td>
      <td>...</td>
      <td>3.8</td>
      <td>1.3</td>
      <td>0.3</td>
      <td>8.9</td>
      <td>0.030</td>
      <td>0.111</td>
      <td>0.174</td>
      <td>0.497</td>
      <td>0.043</td>
      <td>1996-97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>George Lynch</td>
      <td>VAN</td>
      <td>26.0</td>
      <td>203.20</td>
      <td>103.418976</td>
      <td>North Carolina</td>
      <td>USA</td>
      <td>1993</td>
      <td>1</td>
      <td>...</td>
      <td>8.3</td>
      <td>6.4</td>
      <td>1.9</td>
      <td>-8.2</td>
      <td>0.106</td>
      <td>0.185</td>
      <td>0.175</td>
      <td>0.512</td>
      <td>0.125</td>
      <td>1996-97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>George McCloud</td>
      <td>LAL</td>
      <td>30.0</td>
      <td>203.20</td>
      <td>102.058200</td>
      <td>Florida State</td>
      <td>USA</td>
      <td>1989</td>
      <td>1</td>
      <td>...</td>
      <td>10.2</td>
      <td>2.8</td>
      <td>1.7</td>
      <td>-2.7</td>
      <td>0.027</td>
      <td>0.111</td>
      <td>0.206</td>
      <td>0.527</td>
      <td>0.125</td>
      <td>1996-97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>George Zidek</td>
      <td>DEN</td>
      <td>23.0</td>
      <td>213.36</td>
      <td>119.748288</td>
      <td>UCLA</td>
      <td>USA</td>
      <td>1995</td>
      <td>1</td>
      <td>...</td>
      <td>2.8</td>
      <td>1.7</td>
      <td>0.3</td>
      <td>-14.1</td>
      <td>0.102</td>
      <td>0.169</td>
      <td>0.195</td>
      <td>0.500</td>
      <td>0.064</td>
      <td>1996-97</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Each row (i.e. observation) represents a single player. The "pts" variable displays the number of average points each player has scored. 

### Goal
The goal is to use Naive Bayes method to classify players as "high scoring" "mid scoring" or "low scoring".

### Data Munging and Cleaning

The only variables needed to classify a player as "high scoring" "mid scoring" or "low scoring" are kept, the rest are dropped.


```python
# keep only the following columns of data
columnsKept = ['player_name','player_height', 'player_weight','pts','reb','ast','season']

# filter to include only data from 2020 and newer
df = df[df['season'] >= '2020'][columnsKept]

df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>player_height</th>
      <th>player_weight</th>
      <th>pts</th>
      <th>reb</th>
      <th>ast</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11160</th>
      <td>Freddie Gillespie</td>
      <td>205.74</td>
      <td>111.130040</td>
      <td>5.6</td>
      <td>4.9</td>
      <td>0.5</td>
      <td>2020-21</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>Gary Trent Jr.</td>
      <td>195.58</td>
      <td>94.800728</td>
      <td>15.3</td>
      <td>2.6</td>
      <td>1.4</td>
      <td>2020-21</td>
    </tr>
    <tr>
      <th>11162</th>
      <td>Gary Payton II</td>
      <td>190.50</td>
      <td>88.450440</td>
      <td>2.5</td>
      <td>1.1</td>
      <td>0.1</td>
      <td>2020-21</td>
    </tr>
    <tr>
      <th>11163</th>
      <td>Gary Harris</td>
      <td>193.04</td>
      <td>95.254320</td>
      <td>9.9</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2020-21</td>
    </tr>
    <tr>
      <th>11164</th>
      <td>Gary Clark</td>
      <td>198.12</td>
      <td>102.058200</td>
      <td>3.1</td>
      <td>2.9</td>
      <td>0.8</td>
      <td>2020-21</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12839</th>
      <td>Joel Embiid</td>
      <td>213.36</td>
      <td>127.005760</td>
      <td>33.1</td>
      <td>10.2</td>
      <td>4.2</td>
      <td>2022-23</td>
    </tr>
    <tr>
      <th>12840</th>
      <td>John Butler Jr.</td>
      <td>213.36</td>
      <td>86.182480</td>
      <td>2.4</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>2022-23</td>
    </tr>
    <tr>
      <th>12841</th>
      <td>John Collins</td>
      <td>205.74</td>
      <td>102.511792</td>
      <td>13.1</td>
      <td>6.5</td>
      <td>1.2</td>
      <td>2022-23</td>
    </tr>
    <tr>
      <th>12842</th>
      <td>Jericho Sims</td>
      <td>208.28</td>
      <td>113.398000</td>
      <td>3.4</td>
      <td>4.7</td>
      <td>0.5</td>
      <td>2022-23</td>
    </tr>
    <tr>
      <th>12843</th>
      <td>JaMychal Green</td>
      <td>205.74</td>
      <td>102.965384</td>
      <td>6.4</td>
      <td>3.6</td>
      <td>0.9</td>
      <td>2022-23</td>
    </tr>
  </tbody>
</table>
<p>1684 rows × 7 columns</p>
</div>



See whether there are missing data in some columns with this line!


```python
df.isnull().sum(axis=0)  #sum along first dimension/axis, so, indexed by 0 (axis=0)
```




    player_name      0
    player_height    0
    player_weight    0
    pts              0
    reb              0
    ast              0
    season           0
    dtype: int64



Remove all observations (rows/players) who have at least one data missing with this! 


```python
df.dropna(inplace=True) #remove missing values
df.shape
```




    (1684, 7)



We can see that there are still 1684 rows × 7 columns, the same as before, which means there are no rows with missing values in any of the columns!

Lets define "high scoring" "mid scoring" and "low scoring" and see the proportions.


```python
def classify_scoring(points):
    if points >= 20:
        return 'High Scoring'
    elif points <= 10:
        return 'Low Scoring'
    else:
        return 'Mid Scoring'  


df['scoring_class'] = df['pts'].apply(classify_scoring)

# filter out the other values
#dfFiltered = df[df['scoring_class'] != 'Mid Scoring']

print(df['scoring_class'].value_counts())

# low scoring 1142, high scoring 141
```

    Low Scoring     1142
    Mid Scoring      401
    High Scoring     141
    Name: scoring_class, dtype: int64


## Fit Naive Bayes on train, predict on test data


Lets perform the next three steps:
- split the original data into train and test datasets
- make a model and fit it on the train data
- predict on the test data


```python
# randomize the dataset
data_randomized = df.sample(frac=1, random_state=1)

# calculate index for split - take first 80% of the data for test set
training_test_index = round(len(data_randomized) * 0.8)

# split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)
```

    (1347, 8)
    (337, 8)


#### Sanity check

If everything is okay, our training_set and test_set should look structurally the same as the original data frame! Lets plot them for so called sanity check!
This allows us to see whether everything looks okay at least at first sight, without diving much into details!


```python
#sanity check
type(training_set) 
training_set.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>player_height</th>
      <th>player_weight</th>
      <th>pts</th>
      <th>reb</th>
      <th>ast</th>
      <th>season</th>
      <th>scoring_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lonzo Ball</td>
      <td>198.12</td>
      <td>86.182480</td>
      <td>13.0</td>
      <td>5.4</td>
      <td>5.1</td>
      <td>2021-22</td>
      <td>Mid Scoring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lester Quinones</td>
      <td>193.04</td>
      <td>94.347136</td>
      <td>2.5</td>
      <td>0.8</td>
      <td>0.5</td>
      <td>2022-23</td>
      <td>Low Scoring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dereon Seabron</td>
      <td>195.58</td>
      <td>81.646560</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>2022-23</td>
      <td>Low Scoring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Troy Brown Jr.</td>
      <td>198.12</td>
      <td>97.522280</td>
      <td>4.3</td>
      <td>3.1</td>
      <td>1.0</td>
      <td>2021-22</td>
      <td>Low Scoring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nikola Vucevic</td>
      <td>208.28</td>
      <td>117.933920</td>
      <td>17.6</td>
      <td>11.0</td>
      <td>3.2</td>
      <td>2022-23</td>
      <td>Mid Scoring</td>
    </tr>
  </tbody>
</table>
</div>



Looks good!

This will calculate and display the proportion of each category within our scoring_class column for both the training and test sets! 
It shows the percentage of each category relative to the total number of entries in the dataset.


```python
training_set['scoring_class'].value_counts(normalize=True)
```




    Low Scoring     0.684484
    Mid Scoring     0.233853
    High Scoring    0.081663
    Name: scoring_class, dtype: float64




```python
test_set['scoring_class'].value_counts(normalize=True)
```




    Low Scoring     0.652819
    Mid Scoring     0.255193
    High Scoring    0.091988
    Name: scoring_class, dtype: float64



## Creating the Naive Bayes model


Here we create pandas data frame trainX and pandas series trainy. The data frame trainX consists of only features/predictors of the training data. Trainy is the the corresponding labels (column scoring_class) of the training data.

Then we do another sanity check and print out first couple of observations for each of the two objects, to check whether everything is okay.


```python
trainX = training_set.iloc[:,:-1]
trainy = training_set['scoring_class']

colnames = trainX.columns

trainX.head()

trainy.head()
```




    0    Mid Scoring
    1    Low Scoring
    2    Low Scoring
    3    Low Scoring
    4    Mid Scoring
    Name: scoring_class, dtype: object




```python
test_set.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>player_height</th>
      <th>player_weight</th>
      <th>pts</th>
      <th>reb</th>
      <th>ast</th>
      <th>season</th>
      <th>scoring_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malik Beasley</td>
      <td>193.04</td>
      <td>84.821704</td>
      <td>12.7</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>2022-23</td>
      <td>Mid Scoring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake LaRavia</td>
      <td>200.66</td>
      <td>106.594120</td>
      <td>3.0</td>
      <td>1.8</td>
      <td>0.6</td>
      <td>2022-23</td>
      <td>Low Scoring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kevin Knox II</td>
      <td>200.66</td>
      <td>97.522280</td>
      <td>3.9</td>
      <td>1.5</td>
      <td>0.5</td>
      <td>2020-21</td>
      <td>Low Scoring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mark Williams</td>
      <td>213.36</td>
      <td>108.862080</td>
      <td>9.0</td>
      <td>7.1</td>
      <td>0.4</td>
      <td>2022-23</td>
      <td>Low Scoring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R.J. Hampton</td>
      <td>193.04</td>
      <td>79.378600</td>
      <td>6.4</td>
      <td>1.9</td>
      <td>1.1</td>
      <td>2022-23</td>
      <td>Low Scoring</td>
    </tr>
  </tbody>
</table>
</div>



Here we create pandas data frame testX and pandas series testy. The data frame testX consists of all the features/predictors values from the test data and without the output scoring_class. The vector testy consists of output values from the test data.


```python
testX = test_set.iloc[:,:-1] #select all rows & all columns except last one "-1"
testy = test_set['scoring_class'] 
```

### Encoding and training

Lets prepare for training by we encoding train labels (trainy) into 0-1 codes (Bernoulli random variable). The 0-1 coded values are stored in the variable trainBrnli!


```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() #rename function for simpler writing

#0-1 encoding train labels (think of it as a Bernoulli variable)
trainBrnli = le.fit_transform(trainy)

trainBrnli[:5] #print first 5 of train Bernoulli (check that No=0, Yes=1)
```




    array([2, 1, 1, 1, 2])



#### Naive Bayes with mixed variables

Because we have both discrete and continuous features and submodule sklearn.naive_bayes does not handle this situation automatically, we group its values into bins, for each numerical feature. Each bin is represented by a single number from 0, 1, ... , n, where n is the number of bins (categories) for this feature.


```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()  #rename the function for easier writing

trainX = enc.fit_transform(trainX)  #the output is numpy array (ndarray)

trainX = pd.DataFrame(trainX, columns=colnames) #convert ndarray to pandas data frame 
                                                #not required but good for printing 

trainX.head()  #sanity check
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>player_height</th>
      <th>player_weight</th>
      <th>pts</th>
      <th>reb</th>
      <th>ast</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>459.0</td>
      <td>9.0</td>
      <td>24.0</td>
      <td>129.0</td>
      <td>54.0</td>
      <td>51.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>455.0</td>
      <td>7.0</td>
      <td>42.0</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>168.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>684.0</td>
      <td>9.0</td>
      <td>49.0</td>
      <td>42.0</td>
      <td>31.0</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>542.0</td>
      <td>13.0</td>
      <td>89.0</td>
      <td>172.0</td>
      <td>103.0</td>
      <td>32.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating and fitting the model

Then we use CategoricalNB() from sklearn.naive_bayes submodule, which is a classifier based on the Naive Bayes algorithm for categorical data.


```python
model = CategoricalNB()  #create model object
model.fit(trainX,trainBrnli) # fit on train data
```



#### Predicting

Here, we make predictions for each training observation!


```python
yhattrain = model.predict(trainX) # predict on train data
yhattrain
```




    array([2, 1, 1, ..., 1, 1, 0])



#### Confusion Matrix

Now lets look at our confusionmatrix to assess performance.


```python
## confusion matrix using pandas method crosstab
pd.crosstab(yhattrain, trainy)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>scoring_class</th>
      <th>High Scoring</th>
      <th>Low Scoring</th>
      <th>Mid Scoring</th>
    </tr>
    <tr>
      <th>row_0</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>106</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>915</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>6</td>
      <td>310</td>
    </tr>
  </tbody>
</table>
</div>



We could also do this!


```python
#alternative way, which requires encoded labels, i.e. trainBrnli
confusion_matrix(yhattrain, trainBrnli)

```




    array([[106,   1,   2],
           [  1, 915,   3],
           [  3,   6, 310]])



#### Accuracy Score


```python
accuracy_score(yhattrain, trainBrnli)
```




    0.9881217520415738



Now lets repeat some parts of the procedure above, but for the test data!


```python
#0-1 encoding test labels (think of it as a Bernoulli variable)
testBrnli = le.transform(testy)

#transform testX into a pandas data frame with encoded values for all features
testX = enc.fit_transform(testX)
testX = pd.DataFrame(testX, columns=colnames)

#create vector (numpy array) of predictions based on the feature values from the test data
yhattest = model.predict(testX)

yhattest[:9]
```




    array([2, 1, 1, 1, 1, 1, 2, 2, 2])



Then we compute the confusion matrix, for test data, and store it as a variable confM!


```python
## confusion matrix using pandas method crosstab
pd.crosstab(yhattest, testy)


#alternative way, which requires encoded labels, i.e. trainBrnli
confM = confusion_matrix(yhattest, testBrnli)
confM
```




    array([[  8,   1,   4],
           [  1, 213,  28],
           [ 22,   6,  54]])



#### Accuracy Score

Finally we compute accuracy score, the proportion of the correctly predicted outputs!


```python
acc = accuracy_score(yhattest, testBrnli)
acc
```




    0.8160237388724035



The accuracy score is a measure of the proportion of correct predictions made by the model out of all predictions. Our accuracy of 81.6% means that the model correctly predicted the scoring category (high, mid, or low) for about 81.6% of the players in our test dataset. 

I would say that this is a relatively high score, which indicates that our model is performing well! Although there is room for improvement. 

Additionally, looking at the data, it is obvious that there is a bit of an imbalance with a significant majority of instances classified as "Low Scoring" and notably fewer instances classified as "High Scoring", which is the smallest class by proportion. Considering this distribution, we could potentially implement techniques like SMOTE to enhance our model's overall performance. This is something I would consider doing in our second project to further enhance the efficiency of this one!

In conclusion, this project has provided valuable insights into the classification of NBA players based on their scoring abilities. Our model has demonstrated strong performance with an accuracy score of 81.6%. I've gained significant knowledge throughout this process and am looking forward to refining this model further as we continue to explore new techniques throughout the semeseter! 
