# Marine Snails Age Prediction Model
This work contains a machine learning model using all the four types of Regression, which is used to find the age of marine snails called as Abalones. Customarily, the age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope. But it is a tedious and time-consuming task. Therefore, other measurements, which are easier to obtain, are used to predict age.<br>
___
**Attributes:** We have given eight attributes which includes
* Length : Longest shell measurement (in mm).
* Diameter : Diameter of the shell calculated as perpendicular to length (in mm).
* Height : Height of the shell with meat in shell (in mm).
* Whole weight : Weight of whole abalone (in grams).
* Shucked weight : Weight of meat (in grams).
* Viscera weight : Gut-weight after bleeding (in grams).
* Shell weight : Weight of the shell after being dried (in grams).
* Rings : Number of rings in a shell. (Adding 1.5 to the number of rings gives the age of abalone in years).<br>
*__Rings is the target attribute others are features used for predicting rings.__*<br>

#### Simple Linear Regression Model:
- The accuracy for this model is around 74.681% for training data and around 74.859%.

  ![Screenshot 2024-02-15 111921](https://github.com/Priyanshu8887/Determining-Snail-s-AGE/assets/112472808/ef61ebb2-f4a2-4e25-9764-94994ecfa717)


#### Multivariate Linear Regression Model:
- The accuracy for this model is around 77.802% for training data and around 77.393%.

#### Polynomial Regression Model:
- For different value of degree of polynomial there is different accuracy.
- To find the accurate value of p, I have used *__Elbow method__* which gives p = 5, so the maximum accuracy is for p = 5..
- The accuracy for trainind data is 75.301% and for testing data 75.488%

  ![Screenshot 2024-02-15 111947](https://github.com/Priyanshu8887/Determining-Snail-s-AGE/assets/112472808/b24f2548-0cff-4fb9-860c-eae6911811c7)


#### Multivariate Polynomial Regression Model:
- For this still p = 5 is the best degree of polynomial, so the maximum accuracy is for p = 5..
- The accuracy for this model is around 83.784% for training data and around 90.923%.

  ![Screenshot 2024-02-15 120702](https://github.com/Priyanshu8887/Determining-Snail-s-AGE/assets/112472808/5a3ede41-c18e-442a-99b2-62d8ee7144ce)

- From the figure we can see after p = 5 accuracy with p is is almost constant, so p = 5 is the best degree of polynomial.
