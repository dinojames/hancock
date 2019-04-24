import numpy as np

x = np.array([[-3, -2],[-1, -0.7],[2,1],[-2.6,-1.2],[1.5,2],[3,3.8],[-0.3,-2]])
y = np.array([1,1,2,1,2,2,1])

for i in range(len(x)):
    print('Value', x[i], 'belongs to class ', y[i])
    
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x,y)

print('Enter value to be classified: ')
classifier = [int(z) for z in input().split()]
print('Value', classifier, 'belongs to class', gnb.predict([classifier])[0])