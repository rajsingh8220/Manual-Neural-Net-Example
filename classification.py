from main import *
import numpy as np
#Classification example..............................................
import matplotlib.pyplot as plt
#%matplotlib inline

def sigmoid(z):
    return 1/(1+np.exp(-z))

sample_z = np.linspace(-10,10,100)
sample_a = sigmoid(sample_z)

plt.plot(sample_z,sample_a)



from sklearn.datasets import make_blobs
data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
print(data)

features = data[0]
labels = data[1]
plt.scatter(features[:,0],features[:,1])

x = np.linspace(0,11,10)
y = -x+5

plt.scatter(features[:,0],features[:,1], c=labels, cmap='coolwarm')
plt.plot(x,y)



    
class Sigmoid(Operation):
    def __init__(self,z):
        super().__init__([z])
    def compute(self,z_val):
        return 1/ (1+np.exp(-z_val))
                   
                   
# (1,1)* f -5 = 0
g = Graph()
g.set_as_default()
x = Placeholder()
w = Variable([1,1])
b = Variable(-5)
z= add(matmul(w,x),b)

a = Sigmoid(z)

sess = Session()
result = sess.run(operation=a, feed_dict={x:[8,10]})

print(result) # gives 0.99 means the point 8,10 falls under first class above the line

