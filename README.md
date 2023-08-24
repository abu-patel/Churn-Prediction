# Churn-Prediction
Abu Patel
 # Data Science Internship - Churn Preditiction Using Multi Time Slicing Approach in B2B Setting 

 *Machine Learning, B2B, Time Slicing*
 
  This Churn Analysis aims to contribute to both the theoretical and empirical body of knowledge in the non-contractual B2B customer churn prediction domain. In particular, we explore: 
  - Whether it is possible to use a single common source of business data (i.e., invoice data) to devise predictive models capable of reliably identifying churners in real-world settings, 
  - The effects of using different amounts of historical data for devising features on the performance of resulting models
  - whether using alternative churn definitions could yield models that perform well enough to serve as foundations for discussing new potential retention activities.

Finally, by leveraging a recently proposed approach to training data set creation and comparing it with the approach used traditionally, we aim to evaluate whether it generalizes to different case data.

# Churn Definition:

 Churn can be defined in multiple ways however the first step is the timeline to consider. Let’s say one is interested in predicting quarterly churn then churn at a customer level can be defined as customers who did not make any purchase in this quarter but did in previous quarter. Then this customer can be addressed with a value of 1(Churn) otherwise 0 or non-churn. Similarly, one can define churn yearly, monthly and at times even weekly depending on business needs. Our model is capable of handling churn at any time level, and this is only one of its robust capabilities

# Multiple Time Slicing Technique:
This is really the heart of the project:

 ![Single Time Slice Technique](image.png)

 Multi Slice shown below:
 ![Multiple Time Slice Tecnique](image-2.png)

 ## Advantages of Time Slicing 

 - It is robust enough to account for existing seasonality
 - The multislice model is neither an over sample nor an under-sample approach when it comes to imbalanced response distribution. Because of multiple training slices the model simply represent the data for what it is and ultimately tackles the class imbalance issue
 - Instead of out of sample testing the multislice model considers out of period testing that is testing on a more recent time slice which is much more realistic than random sampling.
 - Increase the training size drastically. Depending on number of years of data at hand we have methodologies in place to determine the right number of training slices. This dramatically increase the training size leveraging as much data as possible
 - Missing Trend values are implicitly calculated by the model as the different training slices are a result from shift in time and therefore eliminates the need of explicitly calculating trends


## 2X5 Nested CV Technique

In order to maintain no data leakage between multiple training slices and test set we have developed a methodology that uses 2X5 nested cross validation where the inner loop tunes the hyper parameter along with a wrapper method for feature selection and the outer loop delivers performance metrics on the whole set. Personally, an achievement that we are very proud of is that only by using invoice data mainly we can come up with a set of 52 feature space that can be applied to literally any distributor and when integrated with the power of multi-slice approach and statistically sound machine learning techniques we can significantly bridge the gap between current limitations in non-contractual B2B setting.

Please see the Markdown file for complete documentation on this:

single CV 
![CV](image-3.png)

Multi Training set CV 
![CV](image-4.png)

## Feature Space 

We developed 54 Feature from scratch and increased the training size significantly using Multi Time Slice Technique

![Feature Space](image-5.png)

## Feature Selection

We used a wrapper method along with multiple other feature selection technique, however **Boruta Selection Method**  yielded fairly similar results : 

![Baruta](image-6.png)

## Results 

ROC Curve of **0.82**:
![ROC ](image-7.png)

Confusion Matrix:
![CF](image-8.png)

- Over **1 Million Dollar Savings** for only a small to medium size distributor in a quarter
- Established Strategies with Sales team to prevent customer that has higher probability of Churning 

## Deployment 

- First AI model to be deployed at this company 
- Increased revenue by 4.2% in less than a year

![Churn KPIS](image-9.png)

