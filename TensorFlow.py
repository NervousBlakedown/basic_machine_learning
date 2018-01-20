#Import TensorFlow into Jupyter Notebook program; opens new browser tab.
import tensorflow as tf
hello = tf.constant("hello world")
sess = tf.session()
print(sess.run(hello))

#Numpy Reference.
import numpy as np
my_list = [1, 2, 3]
type(np.array(my_list))
arr = np.array(my_list)
np.arange(0, 11, 2) #goes up to 11, does not include 11.  Shows interval between range.
np.zeros((3, 5))
np.ones(3)
np.ones((3,5))
np.linspace(0, 11, 100) #shows number of points you want between range.
np.random.randint(0, 100, (3, 3))
np.random.normal
np.random.seed(101)
arr = np.random.randint(0, 100, 10)
arr.max()
arr.min()
arr.mean()
arr.argmax() #shows index location of max value.
arr.argmin() #shows index location of min value.
arr.reshape(2, 5)
mat = np.arange(0, 100).reshape(10, 10)
mat[0, 1] #zero-index based.
mat[:, 0]
mat[5, :] #"row, comma, column".  colons show everything.
mat[0:3, 0:3] #start counting at zero.  Up to, not including to last number.
my_filter = mat > 50
mat[my_filter]
mat[mat > 50]

#Pandas Reference.
import pandas as pd
pwd #Shows directory you are currently in.
df = pd.read_csv('salaries.csv') #tab will auto-complete extension and show all file extension types.
df['salary', 'name'] #shows column names
df['salary'].max() #shows maximum value in column.
df.describe() #shows numerical columns data within dataframe (count, mean, std, min, 25%, 50%, 75%, max).
my_salary = df['salary'] > 60000
df[df['salary'] > 60000] or df[my_salary]
df.as_matrix() #returns numpy array.

#Data Visualization Reference.
import numpy as np
import pandas as pd
import matplotlib.pylot as plt
%matplotlib inline #jupyter notebook only.  below line for everything else.
plt.show()
x = np.arange(0, 10)
y = x ** 2
plt.plot(x, y, 'red') #shows red line.
plt.plot(x, y, '*') #shows stars on graph.
plt.plot(x, y, 'r--') #shows red line with dashes.
plt.xlim(0, 4) #shows x-axis limits at 0 and 4.
plt.ylim(0, 10) #shows y-axis limits at 0 and 10.
plt.title("title goes here")
plt.xlabel('x label goes here')
plt.ylabel('y label goes here')
mat = np.arange(0, 100).reshape(10, 10) #makes array.
plt.imshow(mat, cmap = 'RdYlGn')
mat = np.random.randint(0, 1000, (10, 10))
plt.imshow(mat)
plt.colorbar()
df = pd.read_csv('salaries.csv')
df.plot(x = 'salary', y = 'age', kind = 'scatter') #kind could be 'line' or whatever else you need.

#SciKit-Learn Reference/Pre-Processing.
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data = np.random.randint(0, 100, (10, 2))
scaler_model = MinMaxScaler()
type(scaler_model)
scaler_model.fit(data)
scaler_model.transform(data)
scaler_model.fit_transform(data) #do fit then transform, don't use this line.
import pandas as pd
mydata = np.random.randint(0, 101, (50, 4))
df = pd.dataframe(data = mydata, columns = ['f1', 'f2', 'f3', 'label'])
X = df[['f1', 'f2', 'f3']]
y = df['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train.shape
X_test.shape
