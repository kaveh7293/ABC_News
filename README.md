<h2> News Reliability Determination Using Natural Language Processing</h2>
<h3> Overview</h3>
<p> In this project, I used a dataset containing more than 20000 rows of data to develop a classfication model which will be helpful for identifying the relible news. Each row of the dataset contains information including news id, news title, news author, news text and the corresponding label (i.e, 0 or 1). I used the news title and news text to make a classification model. I did not use the news author column because the number of unique authors were huge so if I add that as a feature the corresponding transformed data corresponding to authors became so large (e.g., if onehot encoding were used). Note that, I made a pipeline (as recommended) to facilitate the use of model after training for the users. Further, the dataset is balance because about a half of the labels are 0 and the rest are 1.</p>
<p> The steps that have been used to solve this problem is as follows:<br>
<ol>
<li> I imported the required libraries and modules for fitting the model </li>
<li> I created a <strong>class to handle the missing values</strong> in the news title and news text columns. Note that this class contaied three important methods (i.e., fit, fit_transform and transform)  so that it can be used in as an element in a pipeline. Also, this class inhere from BaseEstimator, TransformerMixin classes (the same format were used for all the defined classes which will be used in the pipeline). The <strong>missing values </strong> are handled as follows:
<ul>
<li> For the text column, the missing values were replaced with the string 'empty'. I did so because I think being empty is also important when reliability of news is evaluated.</li>
<li> For the title column, the missing values were replaced with the string 'not specified'. Simialrly, I believed that this approach is better than simply dropping the rows because it adds additional information which will be helpful during model developement. </li>
</ul> 
</li>
<li> I created a class (with the required features so that can be used as a pipeline element) for preprocessing the text. To do so, I used the following preprocessing:
  <ul><li> I defined a method in the class so that clean the text by removing special characters (i.e., #$% etc.) and also replacing punctuation forms with the original form of the words</li>
    <li> I <strong>lemmatized </strong> the words in the texts and titles. To be more efficient, I defined a class to <strong>determine the parts of speech </strong> of the words so that a more efficient lemmatizing can be used (e.g., the word following can be lemmatized to follow if it is a verb but it will not change if it is a noun).</li>
      
    
  </ul>
  </li>
  <li> I defined a class so that vectorize the columns text and title and then stack them together. Note that, I did not want to use a single tfidf object for both of the text and title columns. I prefer to first vectorize each column and then stack them together. </li>
  <li>Finaly, <strong> different classifiers (i.e., Gaussian naive Bayes classifier, multinomial naive Bayes classifier and support vector machine classifier) with their hyperparameters are specified</strong> and a randomized search cross-validation were implemented to determine the optimal classfier with its corresponding hyperparameters (<a href='https://github.com/kaveh7293/ABC_News/blob/main/Take_Home_Assignment_HyperParameterTuning.py'><strong>HyperParameter Tuning Code</strong></a>). Note that a better model could have been xgboost, but since its hyperparameter tunning were very time-consuming I could not find appropriate parameters using that. Note that even though I chose simple models, the process of hyperparameter tuning took 54 minutes. I ran hyper-parameter tunning when a xgboost model is used, but after about 5 hours I stopped the kernel because of the limitted time. 
   </li>
  <li>After finding the appropriate hyperparameter, pipeline containg all the steps was used  and the quality of the model predictions for the test data set were determined (<a href='https://github.com/kaveh7293/ABC_News/blob/main/Take_Home_Assignment_FittingModel.py'>see the code here</a>). 
  <li>
 </ol>
 <h3> Advantages of the code</h3>
 <p> This code is written in a <strong> pipeline </strong> format so that the final model accepts the pandas dataframe as an input. The model-user should not even choose the corresponding columns which are used for inputs of the model. The only thing that model-user should be careful is that their dataframe should have column names 'title' and 'text'.
 <h3> Limitation and Recommendation </h3>
 <p> The current project that I have done have the following limitations and can be improved further:
 <ol>
 <li>
  I used a tfidf for vectorizing the text data in this study. The disadvantage of this method is that it does not account for the order of the words in sentences. A more accurate method can be used using <strong>word embedding </strong> which could be used from tensorflow.keras module (see my other repository <a href='https://github.com/kaveh7293/Spotify-Reviews-'> here</a> which used this method for classifying the text) and use transfer learning.  
 </li>
  <li> A deep learning model could also be used for classfication. Since the f1-score of this code is relatively small, I would like to do so, but because of time limit I did not do that. </li>
  <li> 
 </ol>
