import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import matplotlib.pyplot as plt

l1 = np.array(["Ciao bello come stai", "bello come giornata", "Napoleone ha conquistato una giornata"])

cv = CountVectorizer()
hv = HashingVectorizer(n_features=2**4)
l1_t = (cv.fit_transform(l1)).toarray()
l1_h = (hv.fit_transform(l1)).toarray()

print(l1_t)
print(l1_t.shape)

print(l1_h)
print(l1_h.shape)


l2 = np.array(["Ciao bello stai", "bello bello bello chiaro", "Napoleone ha conquistato una giornata"])
l2_t = (cv.transform(l2)).toarray()
l2_h = (hv.fit_transform(l2)).toarray()
print(l2_t)
print(l2_t.shape)

print(l2_h)
print(l2_h.shape)

x = np.linspace(-10, 10, 200)    # get a sample of the x axis
y = np.exp(-(x**2)/(2*1))        # compute the function for all points in the sample
plt.plot(x, y)                   # add the curve to the plot
plt.ylim(-0.05,1.05)             # set bottom and top limits for y axis
plt.show()                       # show the plot