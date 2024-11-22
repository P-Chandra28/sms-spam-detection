import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("C:/spammsgs.csv")
print(df.shape)
df.head()
df.describe()
spam_messages = df[df["type"]=="spam"]
spam_messages.head()
spam_messages.describe()
sns.countplot(data = df, x= df["type"]).set_title("Amount of spam and no-spam messages")
plt.show()
x_train, x_test, y_train, y_test = train_test_split(df.text,df.type,test_size=0.2,random_state=0) 
print("data_train, labels_train : ",x_train.shape, y_train.shape)
print("data_test, labels_test: ",x_test.shape, y_test.shape)
vectorizer = CountVectorizer() 
x_train_count = vectorizer.fit_transform(x_train)
x_test_count  = vectorizer.transform(x_test)
clf = MultinomialNB()
clf.fit(x_train_count, y_train)
y_preds = clf.predict(x_test_count)
print ("accuracy_score : ", accuracy_score(y_test, y_preds))
#print ("confusion_matrix : \n", confusion_matrix(y_test, y_preds))
print (classification_report(y_test, y_preds))