# SMS Spam Detection 

In this project, we are going to detect the whether the received SMS to the mobile is a normal or a spam SMS. In this we are using a dataset provided by the Kaggle for the training of the model. This dataset consists of 5559 SMS received by a mobile which contains both spam and genuine SMS. The data is split in the ratio of 80:20 and is used to train and test the model.

<img src=".\week 1\results\randomforest.png">

The above image is a graphical representation of the spam and genuine SMS in the given dataset. The model used is the multinomial naive bayes classifier model. Multinomial Naive Bayes is a specialized version of Naive Bayes that is designed more for text documents. Whereas simple naive Bayes would model a document as the presence and absence of particular words, multinomial naive bayes explicitly models the word counts and adjusts the underlying calculations to deal with in. It is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. 

The testing of the model is done and the results are stored in the "y_preds" variable. This data is used to find the accuracy and the outcome of the model and its prediction efficiency.


#The final results and the accuracy of the detection of the SMS

The following images give the accuracy and the classification report of the model.

<img src=".\week 1\results\randomforest.png">

As we can see the accuracy score is ~0.9901 which means that the model  had detected 99% of the messages correctly.





