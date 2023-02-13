# Machine-Learning_LinearRegression
 Linear regression is an effective model used for predicting from one or more variables (Independent) in Machine learning. It is a Supervised Algorithm which learns to predict a value based on past data.
 
The general Equation is:
			**Y = mx + c**
			
Where **Y** is the dependent variable									
      **X** is the independent variable
The given Data set is  monet.CSV

Using the attached dataset to develop, train, and evaluate a group of linear regression models to predict the price (dependent variable) of a Monet painting from a few of its features (independent variables). Create your model in Python.

Tasks: 1. Create at least two simple linear regression models, each of them has one different independent variable (you may transform the raw independent variable into different formats, such as to conduct a logarithmic transformation or combine two variables into a new variable such as Size = width * height). You may consider one variable as Size, and another one as Width. Create a scatter plot for showing the relationship between the independent variable and the dependent variable for each model and showing the linear regression line in the same plot. Calculate the error of the prediction with test data. 

2. Create a multivariate linear regression model. You may need to consider the normalization of the raw data. Calculate the error of the prediction with test data. 

MODEL-1
Platform: Jupyter Notebook

![image](https://user-images.githubusercontent.com/54754462/218348195-d70e6f24-a80e-4a87-a166-5ca72b56f125.png)

Here I have imported the required libraries for the Tasks
Pandas: By definition. It can be used to perform data manipulation and analysis 

NumPy: It is for Array concepts and has functions to work in the field of Mathematics such as linear algebra and Matrices etc.

Matplotlib: which is used as a Visualize the given dataset.

Seaborn: which is used to Visualize the random distributions (Statistical Graphs)

Reading the Dataset into pandas. The dataset is in Comma separated values and will use pd.read_csv which reads the CSV files.

The pandas provide a dataframe to the dataset.
