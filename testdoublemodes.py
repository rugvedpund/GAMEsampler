import flowlikelihood
import numpy as np

print('testing sample arrays [[amplitude,width,numin]]')

arr1 = np.array([[1.0, 14.0, 16.4]])
test1 = flowlikelihood.likelihood(arr1)
print(arr1, "\nlikelihood:", test1)
assert np.allclose(test1,-793.66693)

arr2 = np.array([[0.5,10.0,15.0],
                [1.5,15.0,20.0]])
test2 = flowlikelihood.likelihood(arr2)
print(arr2, "\nlikelihood:", test2)
assert np.allclose(test2,[-1542.9315, -944.0876])

print("all ok")
