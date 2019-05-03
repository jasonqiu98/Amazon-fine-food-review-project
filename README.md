# Amazon-fine-food-review-project
Course Project Amazon Fine Food Review

This is a course project for ERG3020 by three students from CUHK(SZ). The dataset used in this project is revised from the original Kaggle dataset, which can be found from the link [here](https://www.kaggle.com/snap/amazon-fine-food-reviews).

The source dataset can be downloaded after logging in to Kaggle. Those data should be put in "data" folder, and the "data" folder should be put in the project folder.

## Project Organization

* Data preprocessing (read_data.ipynb)
* Naive Bayes implementation (lib.py and naive_bayes.ipynb)
* SVM practice (lib_svm.py and svm.ipynb)
* k-NN practice (knn.ipynb)

## Project Conclusion

* The project conducts sentiment analysis of Amazon Fine Food Reviews dataset with Naïve Bayes, SVM, and k-NN algorithms. Naïve Bayes algorithm is thoroughly implemented and other algorithms are applied using tools from sklearn. Naïve Bayes is easy to implement and more functions are provided in addition to the calculation of evaluation measures. Our sentiment analysis indicates SVM algorithm has nearly the best performance. The k-NN algorithm generates high recall of positive examples and low overall accuracy due to the effect of local data. Future improvements include data preprocessing and model improvements.

* In semantic level, Naïve Bayes can provide abundant amount of useful information. Some words, such as "hears", "perfect", "antioxidents", "winter" and "dances", indicate positive sentiment. Some words, such as "uneatable". "implicated", "refundable", "unhelpful", and "poorest", indicate negative sentiment. Correlation analysis using chi-squared score is conducted. Results show that the words, such as "not", "great", "disappointed", "bad" and "love", etc., are highly correlated to the sentiment labels. Furthermore, five reviews are selected and analyzed. Naive Bayes tends to lower the conditional probability for words given the sentiment label is positive. This is due to the asymmetry of the dataset regarding sentiment labels. In addition, some quasi-positive words, such as "like", "better", "help", "new" and "improved", actually denote negative sentiment. It may provide more insights for further sematic analysis.

## References
* [SAILORS 2017 NLP project](https://github.com/abisee/sailors2017)
* [Amazon-Food-Reviews-Using-Linear-SVC](https://www.kaggle.com/amitrajitbose/amazon-food-reviews-using-linear-svc/notebook)
* [Amazon fine food reviews analysis using KNN](https://www.kaggle.com/premvardhan/amazon-fine-food-reviews-analysis-using-knn/notebook)
