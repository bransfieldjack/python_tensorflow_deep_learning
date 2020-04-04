![tf](https://github-jackalack117.s3-ap-southeast-2.amazonaws.com/1_FxMUvjm1mlfKJhIC_cOJSw.png)

### Setup: 

I'm using a conda environment (conda 4.8.3).
To get going: 

```
conda env create -f environment.yml
```
**You might have some issues with setup as I am using python 3.8 with TF v.2.0.
If this is the case, I recommend you install a blank conda env from scratch with python 3.6. **

Test your tensorflow version from python shell: 

```
python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3
```

### Notes: 

The **syntax.py** file is a script with basic tensorflow computations and operations. 
The conda environment for this repo was built with tensorflow v2, but this script uses v1. 
You can switch between the two, check the headers at the top of the script for information on how. 

![graphs](https://github-jackalack117.s3-ap-southeast-2.amazonaws.com/nodegraph.PNG)

**graphs.py** produces the above default graph. 
n1 + n2 are two constant nodes, 1 & 2.
n3 is an operation node taking n1 & n2 as inputs and outputting an addition operation.

![models1graph](https://github-jackalack117.s3-ap-southeast-2.amazonaws.com/graphmodel1.PNG)

**graph wx + b = z**

Weight variable (w) will be multiplied with a placeholder (x) and fed into matmul operation.
The matmul operation will feed into another operation addition node in the graph, which has a bias (b) variable. 
*
The term bias is used to adjust the final output matrix as the y-intercept does. For instance, in the classic equation, y = mx + c, if c = 0, then the line will always pass through 0. Adding the bias term provides more flexibility and better generalisation to our Neural Network model
*
Final output can be passed into an activation function. (sigmoid function)