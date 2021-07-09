import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from flask import *
learn = Flask(__name__)

@learn.route('/')
def result():
        return render_template("diabetes.html")
@learn.route('/result', methods=['POST'])
def result1():
    if request.method == "POST":
        res=request.form
        preg=request.form["preg"]
        glu = request.form["glu"]
        bp = request.form["bp"]
        st = request.form["st"]
        ins = request.form["Ins"]
        bmi = request.form["bmi"]
        dpf = request.form["dpf"]
        age = request.form["age"]
        p = diabetesPrediction(preg,glu,bp,st,ins,bmi,dpf,age)
        return render_template("hello.html",res=p)
@learn.route("/result1",methods=["POST"])
def result2():
    if request.method == "POST":
        res1=request.form
        male=request.form["male"]
        age = request.form["age"]
        cs = request.form["cs"]
        cigar = request.form["cigar"]
        bpmeds = request.form["bpmeds"]
        stri = request.form["str"]
        hy = request.form["hy"]
        diab = request.form["diab"]
        chol = request.form['chol']
        sysbp = request.form['sysbp']
        diabp = request.form['diabp']
        bmi = request.form['bmi']
        hrate = request.form['hrate']
        glu = request.form['glu']
        p = heartDiseasePrediction(male,age,cs,cigar,bpmeds,stri,hy,diab,chol,sysbp,diabp,bmi,hrate,glu)
        return render_template("hello.html",res=p)


#------------------------------------diabetes prediction using svm and knn----------------------------------------------
def diabetesPrediction(preg,glu,bp,st,ins,bmi,dpf,age):
    data = pd.read_csv('diabetes.csv')
    X = data.iloc[:, :8].values
    y = data.iloc[:, 8].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2897, random_state=4)
    #-----------------------diabetes prediction using K-nearest neighbors---------------------------------------------------
    model = KNeighborsClassifier(n_neighbors=17)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred1=model.predict([[preg,glu,bp,st,ins,bmi,dpf,age]])
    print("diabetes prediction accuracy using KNN :",accuracy_score(y_test,y_pred)*100)
    #---------------------diabetes prediction using SVM-------------------------------------------------------------------
    from sklearn.svm import SVC
    model_svm=SVC()
    model_svm.fit(X_train,y_train)
    y_pred_svm=model_svm.predict(X_test)
    y_pred_svm1=model_svm.predict([[preg,glu,bp,st,ins,bmi,dpf,age]])
    acc = accuracy_score(y_test, y_pred_svm) * 100
    print("diabetes prediction accuracy using SVM:",acc)

    #-----------------------------------comparing accuracies of model and predicting disease with the best model--------------------------------------------------------------------------------
    if y_pred_svm1[0]==1:
        return "You are infected with diabetes as per the model whose accuracy is "+str(acc)
    else:
        return "You are not infected with diabetes as per the model whose accuracy is "+str(acc)


#-----------------------------------------heart disease detection using svm and knn---------------------------------------------
def heartDiseasePrediction(male,age,cs,cigar,bpmeds,stri,hy,diab,chol,sysbp,diabp,bmi,hrate,glu):
    data = pd.read_csv('framingham1.csv')
    X1 = data.iloc[:, :14].values
    y1 = data.iloc[:, 14].values

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.312, random_state=1)
    ndata = [[male,age,cs,cigar,bpmeds,stri,hy,diab,chol,sysbp,diabp,bmi,hrate,glu]]
    #-----------------------heart disease prediction using K-nearest neighbors---------------------------------------------------
    model_knn=KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X_train1,y_train1)
    y_pred1=model_knn.predict(X_test1)
    print("heart disease prediction accuracy using KNN :",accuracy_score(y_test1,y_pred1)*100)
    #-----------------------------heart disease prediction using svm------------------------------------------------------------
    from sklearn.svm import SVC
    model_svm=SVC()
    model_svm.fit(X_train1,y_train1)
    y_pred_svm1=model_svm.predict(X_test1)
    y_pred_svm11=model_svm.predict(ndata)
    acc1 = accuracy_score(y_test1,y_pred_svm1)*100
    print(y_pred_svm11)
    print("heart disease prediction accuracy using SVM :",acc1)
    if y_pred_svm11[0]==1:
        return "You are infected with heart disease as per the model whose accuracy is "+str(acc1)
    else:
        return "You are not infected with heart disease as per the model whose accuracy is "+str(acc1)

if __name__ == "__main__":
    learn.run(debug=True)