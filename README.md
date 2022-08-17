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
<li> 
