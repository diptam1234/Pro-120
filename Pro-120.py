from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 

#------------------------------------------------------------------------

wine = datasets.load_wine()

print("-------------------------------------------------------")
print("Features ---> " , wine.feature_names)

print("-------------------------------------------------------")
print("Labels --->" , wine.target_names )

print("-------------------------------------------------------")

#---------------------------------------------------------------------------------------------------
X = wine.data
Y = wine.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 109)


gnb = GaussianNB()

gnb.fit(X_train,Y_train)

y_predict = gnb.predict(X_test)

Accuracy = accuracy_score(Y_test,y_predict)

print("The Accuracy --->",Accuracy)




