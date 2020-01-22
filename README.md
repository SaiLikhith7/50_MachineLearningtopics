># 50 Important topics or tools to know more about Machine Learning

- ###  Python 

Lists, Tuples, Regular Expressions

- ###  Mathematics 

Training, Model Complexity, Estimate right confidence parameter, Linear Algebra, Probability and Statistics 

- ###  Exploratory Data Analysis

    It is valuable to explore Data, check co-reraltion between rows and columns, it's an approach how we understood the data

- ###  Data PreProcessing

   <ol>
    <li>Missing Data- It will show what all are missing values that we require to perform the operations.</li> 
    <li> Categorical data - Gender - M/F , it should be in the form of 0 or 1</li> 
    <li> Outliers - differe greatly from each others</li>
    <li> Feature Selection- Millions of rows and columns, select only few columns, select only few which are co-related to each other</li>
</ol>

## Supervised Learning
- ### Linear Regression
    Single input variable and Single output variable.
    Multiple linear regression is also there where multiple inputs will correspond to the outputs in linear phase.
    We try to plot a line which best fits all the points, generally, it's linear
- ### Ridge and Lasso model
    They will help in regularising the difference between the output and input predictions.
    - Ridge Method- **L2 regularization** adds penality which is square of the absolute of the difference between the output and input coefficients
    - Lasso Method-  **L1 regularization** adds penality which is absolute magnitude of the co-efficients  
- ### Polynomial Regression
    It provides data points in the curve shape. Polynomial regression is in the form of the linear regression in which the relation between the x and corresponding y in polynomial relation
     #### Advantages: 
     - Broad range of functions can be fit under it. It has wide range of the curvature. It provides best approximation of the dependent and independant variables.
    #### Disadvantges: 
    - Too sensitive to outliers 
- ### Support Vector Machines
    For classification and regression, it can solve the linear and non-linear as well. It creates a line or hyper plane which seperates the data into classes. It takes the decision in such a way that the supression in the two classifications is as wide as possible so that we can easily distinguish them.
- ### Decision Trees
    They can solve both regression and classifictaion problems, they use the tree representation to solve the problem. 
    The attributes are represented on the internal nodes of the tree.
    The leaf represents the output of the class labels
    Decision tree classifies example from root to leaf node
- ### Logistic Regression:
    When dependant varibale is categorical like yes or no. It is a sigmoid function. It can mode any value into 0 and 1. 
    There are 3 types of logistic regressions:
    <ol>
    <li>
        Binary Logistic Regression
    </li>
    <li>
        Multinomial Logistic Regression - 3 or more categories eg: cities
    </li>
    <li>
        Ordinal Logistic regression - 3 or more with numerical values eg: movie rating
    </li>
    </ol>
- ### Naive baye's classifier
    Every pair of feature that is independant. It will be classified using Baye's theorm, given that P(A/B). 
    It is used in Recommendor system, Sentiment Analysis, Spam Filtering
- ### K Nearest Neighbours
    It deals with both regression and classification
    The prediction is based on the mean or median of the K most similar instances.
    When Kmin can be used as classification, we can calculate output from class with highest frequency and k most near instances, each instances votes for their class, class with most votes taken as prediction. Kmin assumes that similar things exist in close proximity. Selecting Right K value is important, brute force k values and take k which reduces lot of the output errors.
    
- ### Ensemble Learning
    That combine several base models in order to produce one optimal producing model. This is better than one single producing model. It is used in several hackathons like Netflix hackathons, Kaggle hackathons. It is used to decrease the variance
    Types:
    - Random Forest - uses multiple decision trees for predictions
    - Bagging or bootstrap aggregating
    
- ### Performance Matrix
    Using regression are Mean Avsolute error, RMS, R^2 etc.,
    Using classification are confusion matrix, classification accuracy, area under ROC curve 

## Unsupervised Learning

- ### K Means Clustering
    It looks for fixed number of k clusters in a dataset. Cluster is group of elements with similar properties forms a cluster.
    
- ### Hirearchial Clustering
     It groups similar objects called clusters. Each observation as separate cluster. It repetetively executes the two steps. It identifies the clusters closest to each other, it merges the clusters together. It uses Euclidean distance to merge the similar clusters

- ### Recommendation systems
    - Apriori method- Frequent item pairs are identified. It identifies minimum occuring items which are frequent and uses them to group and then it should have the frequent items as well. It has support, confidence and lift as characteristics and based on this, you will get the next item recommendation. This will be majorly used in the E-commerce websites. It uses BFS as search purpose. This doesn't involve any repetitive scanning of data to find any individual support values.
    - Eclat Method- Its is basically Equivalence class clustering and bottom of lattice travelled cell. It is one of the popular method of the association rules. We use DFS of graph as a vertical search for faster purpose.

## Reinforcement Learning

- ### Upper Confidence bound (UCB)
    It is a multi-m bandit problem. It is used to represent a similar kind of problem and find a good strategy to solve the problem. Bandit is like one who steals our money, it's like a bandit in casino where you pull a lever and get rewards, so, a multi-m bandit is a computed-slot machine, where in one level is pulled rather than the several levers, the task is to find out the probability to pull the which lever to pull to get the maximum reward.

- ### Thompson Sampling
    It is an algorithm for decision problems where actions are taken sequentially in manner that must have expectation what is known to maximum immediated performance and enhance the accumulation of the experience in order to improve the future performance. It addresses a broad range of the problems. 

- ### Time Series Forecasting
    It is used in prediction problems which involves the time components. Why because in classical series, time series must be understood to understand the underlying causes. Forecasting means predicting the future data based on the present data.
    
- ### Natural Language Processing
    It is practically used to perceive the language leaarning. Eg: Google uses to understand speech synthesis.
    Google Translator, Youtube comments under community policies or not.
    Some techniques used in the Natural Language Processing are : 
    - Removing puntucation
    - Tokenization
    - Remove stop words
    - Stemming
    - Limitizing
    - Vectorizing data
    - Backuping data
    - Validating data
## Deep Learning
- ### Artificial Neural Networks
    There are lot of advances in the artifical neural networks.The basic idea of ANN based on belief that working of human brain, by using the correct connections. Human brain imitation considering the semiconductors like neurons in our brain.
    The nodes take the value of the previous nodes and thus build a connection between them this imitating the connections of the brain. 
    Types of Neural Networks:
    
- ### Convolution Neural Network
     It is a deeplearning algorithm which can take input image and assign learnable weights to varous aspects of image such that we can distingusih from one other. While in primitives methods, filters are hand engineered. While in CNN, they can learn this filters or characteristics. These are inspired by the organization of the visual cortex in human brain. Visual neurons respond to stimual in a restrcit field known as the visual field. It has input layers, max pulling convolution. 
- ### Recurrent Neural Networks
     It's decisions are influenced by what it has learned in the past. Basic models only remembers what they are trained, while RNN in addition, they remembers things from the past while dealing with prior inputs to generate the prior outputs. From one or more input vectors and produce one or more outputs vectors and the o/ps depends on not only applied weights but also based on the state vector repressenting the context of the inputs and outputs. So, the same input vectors depending on the prior state, may give the different outputs as well.

- ### Recursive Neural Network: 
    Kind of the deep learning Neural Networks produced while applying the same set of weights recursively over the structured input to predict a structured input over a scalar prediction on it. They are successful in learning trees structured inputs which are sequential in nature.

## Libraries or Packages

- ### NumPy 
    NumPy is a python programming library for multi-dimensional arrays along with lot of mathematical functions as well. It can be used to create vector, sparse matrix, applying operations to elements, calculating min, max, variance, transposing a matrix, find rank or determenant of matrix, clalculate the trace of matrix, eigen values and eigen vectors, inverting the matrix etc.,

- ### Pandas 
    It is also mostly u8sed with pythona dn for data analaysis. Primary object types are dataframes and data series. Can read csv files, excel files, links etc., 

- ### matplotlib 
    It is used for plotting the plots. We can have scatter plots, histograms, bar graphs, diverging bars, area chart, to check the compositions of the data we can go with waffle chart, pie chart, bar chart. To check the distributions, use line plot, histogram, box plot, violin plot.

- ### Scikit learn 
    It features various algorithms like SVM, random forests, K near plots etc., Scikit learn is created to make machine laerning easier. We process data.

- ### Seaborn 
    It is a python visulaization library using matplotlib, provides high quality graphics. It has similar kind of plots of atplotlib + additional plots like cat plots etc.,

- ### Tensorflow 
    It bundles together the Machine Learning and deep learning algorithms. It used python to provide convinient front end api for building applucatiin using the frameworks, using them in high level performance C++. They can run Deep Neural Networks uses those hand writing recognition algorithms, NLP etc, these applications can be run on either android or local machine or google cloud etc., 

- ### Keras
    Keras contains implementations of the commonly used NN building blocks like layers, optimizers etcs., to make working with image and text data easiers

- ### NLTK 
    It can be used as NL tool kit. It has parsing,streaming, tokenizing has tools for almost all NLP tasks. It supports lot of 3rd party extensions.


    
    



```python

```

