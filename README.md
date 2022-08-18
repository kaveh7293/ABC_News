<h2> News Reliability Determination Using Natural Language Processing</h2>
<h3> Overview</h3>
<p> In this project, I used a dataset containing more than 20000 rows of data to develop a classfication model which will be helpful for identifying the relible news. Each row of the dataset contains information including news id, news title, news author, news text and the corresponding label (i.e, 0 or 1). I used the news title and news text to make a classification model. I did not use the news author column because the number of unique authors were huge so if I add that as a feature the corresponding transformed data corresponding to authors became so large (e.g., if onehot encoding were used). Note that, I made a pipeline (as recommended) to facilitate the use of model after training for the users. </p>
<p> The steps that have been used to solve this problem is as follows:<br>
<ol>
<li> I imported the required libraries and modules for fitting the model </li>
<li> I created a class to handle the missing values in the news title and news text columns. Note that this class contaied three important methods (i.e., fit, fit_transform and transform)  so that it can be used in as an element in a pipeline. Also, this class inhere from BaseEstimator, TransformerMixin classes (the same format were used for all the defined classes which will be used in the pipeline). The missing values are handled as follows:
<ul>
<li> For the text column, the missing values were replaced with the string 'empty'. I did so because I think being empty is also important when reliability of news is evaluated.</li>
<li> For the title column, the missing values were replaced with the string 'not specified'. Simialrly, I believed that this approach is better than simply dropping the rows because it adds additional information which will be helpful during model developement. </li>
</ul> 
</li>
<li> I created a class (with the required features so that can be used as a pipeline element) to handle preprocess the text. To do so, I used the following preprocessing:
  <ul><li> I defined a method in the class so that clean the text by removing special characters (i.e., #$% etc.) and also replacing punctuation forms with the original form of the words</li>
    <li> I lemmatized the words in the texts and titles. To be more efficient, I defined a class to determine the parts of speech of the words so that a more efficient lemmatizing can be used (e.g., the word following can be lemmatized to follow if it is a verb but it will not change if it is a noun).</li>
      
    
  </ul>
  </li>
  <li> I defined a class so that vectorize the columns text and title and then stack them together. Note that, I did not want to use a single tfidf object for both of the text and title columns. I prefer to first vectorize each column and then stack them together. </li>
  <li>Finaly, different classifiers with their hyperparameters are specified and a randomized search cross-validation were implemented to determine the optimal classfier with its corresponding hyperparameters. I also chose the value of max_features in the text vectorizer using the crossvalidation. 
   </li>
  <li>The fitted pipeline were tested on the test data set and an f1-score of ... were obtained. 
  
 </ol>
 
 <h3> Limitation and Recommendation </h3>
 <p> The current project that I have done have the following limitations and can be improved further:
 <ol>
 <li>
  I used a tfidf for vectorizing the text data in this study. The disadvantage of this method is that it does not account for the order of the words in sentences. A more accurate method can be used using word embedding which could be used from tensorflow.keras module (see my other repository <a href='https://github.com/kaveh7293/Spotify-Reviews-'> here</a> which used this method for classifying the text).  
 </li>
 </ol>
