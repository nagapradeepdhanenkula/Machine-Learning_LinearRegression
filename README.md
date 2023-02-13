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

**Pandas**: By definition. It can be used to perform data manipulation and analysis 

**NumPy**: It is for Array concepts and has functions to work in the field of Mathematics such as linear algebra and Matrices etc.

**Matplotlib**: which is used as a Visualize the given dataset.

**Seaborn**: which is used to Visualize the random distributions (Statistical Graphs)

Reading the Dataset into pandas. The dataset is in Comma separated values and will use pd.read_csv which reads the CSV files.

The pandas provide a dataframe to the dataset.

![image](https://user-images.githubusercontent.com/54754462/218348328-c5557f5a-403f-4eb3-8c57-41ed046e538e.png)

We need to check the shape of the dataset which tells us how many rows and columns were present in the dataset.

![image](https://user-images.githubusercontent.com/54754462/218348349-066e3782-ad08-4fde-aafe-97c32d6b9e15.png)

From the data set, we have a total of 6 features and 430 rows

![image](https://user-images.githubusercontent.com/54754462/218348381-813b6051-a95f-470d-aff9-40cbf9d0ba04.png)

We can see that the monet.csv file doesn’t have any missing values

![image](https://user-images.githubusercontent.com/54754462/218348394-cd1ab9c0-b350-471c-927f-d880d6c5a605.png)

This shows the descriptive statistics of the dataset like mean, standard deviation and quartile ranges etc.

![image](https://user-images.githubusercontent.com/54754462/218348415-4496e9fc-8284-4722-9fcb-1a6c4829ca34.png)

From the above Correlation matrix using Heatmap between attributes of the dataset.

The strong correlation has seen between the features “Height” Vs “Width” with a value of 0.5 which is a positive correlation means if one variable increases the other variable also increases

![image](https://user-images.githubusercontent.com/54754462/218348446-35e4e860-3073-42bc-9a0f-211b10de0c14.png)

According to the Task-1, a new variable is created by multiplying the width X Height.

![image](https://user-images.githubusercontent.com/54754462/218348477-5f94bc82-b012-42d3-b018-6ebebb2065dd.png)

Here Splitting the data usually **80**% is for Training data and **20**% is for Testing the data. importing the Linear regression to train the dataset

![image](https://user-images.githubusercontent.com/54754462/218348492-af9bf920-baee-4863-b4f0-93582b7bfa1e.png)

Here we have used reshape () function which changes the shape of an array without changing the data in dataset.

![image](https://user-images.githubusercontent.com/54754462/218348539-c054a70c-b359-4dea-9663-49fb50b3df4f.png)

We have imported the required libraries like train_test_split to split the data in desired ratio, linear regression to train the data, mean_squared_error and r2_score for the evaluating the model.

Here, height is the independent variable and price is the dependent variable and as X_train and Y_train.

**Mean -Squared error (MSE):**
![image](https://user-images.githubusercontent.com/54754462/218348575-896f2ca9-73f4-40f0-bb9d-2ad7c3a9921d.png)

**R-squared:**
![image](https://user-images.githubusercontent.com/54754462/218348594-0d12afdd-c728-4ffe-b8f3-c9983ad30567.png)

If r2_score = 1 that means the model is perfect for prediction otherwise doing worse (negative value).

From the plot, we can see that most of the readings (red dots) are around the line(Blue).so, the model is good for prediction.

**Model 2:**
In this lr model 2, we will use different attribute and will be repeating the steps by tweaking the code with “**Size**” instead of “**Height**”.

We will import all the required libraries to split the data, linear regression and model evaluation like: MSE and R-squared.

![image](https://user-images.githubusercontent.com/54754462/218348655-e331257d-29f4-4cc5-aa68-c0411e049550.png)

![image](https://user-images.githubusercontent.com/54754462/218348663-a38d0089-e561-4355-b898-c1076610c250.png)

Here Splitting the data usually 80% is for Training data and 20% is for Testing the data. importing the Linear regression to train the dataset.

Here we have used reshape () function which changes the shape of an array without changing the data in dataset.

We have imported the required libraries like train_test_split to split the data in desired ratio, linear regression to train the data, mean_squared_error and r2_score for the evaluating the model.

Here, “size” is the independent variable and “price” is the dependent variable and as X_train and Y_train.

If r2_score = 1 that means the model is perfect for prediction otherwise doing worse (negative value).

From the plot, we can see that most of the readings (blue dots) are around the line (Black).so, the model is good for prediction.

The linear regression model means its between the two variables and multivariate linear regression model means 2 or more independent variables.
As for the raw data from monet.csv. we have total of 6 features.

In this model, we will be considering height, width, signed, picture and house as independent variables (X_train) and Price as the dependent variable (Y_train).
As per the Task 2, we need to consider the normalization of the raw data.

There are 4 ways to normalize and make the data on a similar scale. Namely: MinMaxScaler, RobustScaler, StandardScalar and normalizer.

![image](https://user-images.githubusercontent.com/54754462/218348787-968a0056-287f-4828-8ad1-b7c10106ffa0.png)

Hence, will be using “StandardScalar” to normalize the data on a similar scale.

![image](https://user-images.githubusercontent.com/54754462/218348808-eb51840c-119c-4a35-b4ff-730e84bcacb9.png)

![image](https://user-images.githubusercontent.com/54754462/218348819-52f370ce-773d-4153-b6d4-f4cb2f5914ef.png)

In this Multivariate linear regression model, we will be dropping the “size” variable from the dataframe as we created this by multiplying with height*width.
From the seaborn.regplot(), we are able to see most of the data points close to the best fit line.

![image](https://user-images.githubusercontent.com/54754462/218348838-1821ba68-2ab7-4b3b-ba1f-adbd5e869e92.png)










