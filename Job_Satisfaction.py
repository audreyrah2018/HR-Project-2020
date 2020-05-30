import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
W  = '\033[0m'  # white 
R  = '\033[31m' # red
G  = '\033[32m' # green
O  = '\033[33m' # orange
B  = '\033[34m' # blue
P  = '\033[35m' # purple

print(P+"*********************************************************")
print("                                                         ")  
print(O+"         THE PREDICT EMPLOYEES SATISFACTION              ")
print("                                                         ") 
print(P+"*********************************************************"+W)
print("                                                         ")

#Preprocessing
HR = pd.read_csv("HR_Project.csv", header=0)
HR.drop(['Attrition','BusinessTravel'], axis=1,inplace=True)
HR.replace("Male",1, inplace=True)
HR.replace("Female",2, inplace=True)
HR.replace("Single",1, inplace=True)
HR.replace("Married",2, inplace=True)
HR.replace("Divorced",3, inplace=True)
HR_JOB_LEVEL=HR['JobLevel']

#  Making Features and Target
HRx = HR.drop('JobSatisfaction', axis=1)
HRy = HR['JobSatisfaction']

#  Training & testing
Knn=KNeighborsClassifier()
Knn.fit(HRx,HRy)

x_train, x_test, y_train, y_test= train_test_split(HRx,HRy, test_size=0.20,
                                                   random_state=42 , stratify=HRy)
neighbors=np.arange(1,6)
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))

for d,f in enumerate(neighbors):   
    knn_model=KNeighborsClassifier(n_neighbors=f)
    knn_model.fit(x_train,y_train)
    train_accuracy[d]=knn_model.score(x_train, y_train)
    test_accuracy[d]=knn_model.score(x_test, y_test)

test_accuracy = test_accuracy*100
test_accuracy = test_accuracy.astype(int)

plt.scatter(HR_JOB_LEVEL,HRy)
plt.xlabel("Job Level")
plt.ylabel("Satisfaction")
plt.show()

k=0
for q,r in enumerate(test_accuracy):
    if test_accuracy[q] > k :
        k=test_accuracy[q]
print(P+"The ACCURACY rate is about : "+O+"%",k)
print(B)       

n_features = np.array(['Age','Distance From Home (Kilometer)',
                       'Gender(Male=1 , Femail=2)',
                       'Marital Status (Single=1, Married=2, Divorced=3)', 
                       'Job Level as the list above', 'Monthly Income'])
X_New_employee=np.empty(len(n_features), dtype=int)
for i,j in enumerate(n_features):
    if n_features[i]=="Job Level as the list above":
        print(G+"-------------------------")
        print(G+"Manager High Level   = 5 ")
        print(O+"Manager Medium Level = 4 ")
        print(G+"Specialist Educated  = 3 ")
        print(B+"Employee Technician  = 2 ")
        print(W+"Employee clerk       = 1 ")
        
    print("Enter your ",j," :")
    X=input()
    X_New_employee[i]=(X)
        
X_New=np.reshape(X_New_employee,(1,-1))
Y_New=Knn.predict(X_New)
 
if Y_New==5:
    Satis="a very much Satisfaction 😃😃😃😃😃"
elif Y_New==4:
    Satis="a Goog level of Satisfaction 😃😃😃"
elif Y_New==3:
    Satis="a Moderate Satisfaied 😃"
elif Y_New==2:
    Satis="a DISATISFACTION 😞  "
elif Y_New==1:
    Satis="a Very DISATISFACTION 😞😞😞😞 "  
    wait=input("Press Enter 4")   
print("-------------------------------------------------------------------------")
print("                                                         ") 
print(R+"*********************************************************************")  
print(W+"New Employee's job level will be "+O+"", Satis)
print(R+"*********************************************************************")
print("                                                         ") 
print("                                                         "+W)

wait=input("END OF PROGRAM , Press enter to finish program ")
