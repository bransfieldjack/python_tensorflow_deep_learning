![tf](https://github-jackalack117.s3-ap-southeast-2.amazonaws.com/1_FxMUvjm1mlfKJhIC_cOJSw.png)

### Setup: 

I'm using a conda environment (conda 4.8.3).
To get going: 

```
conda env create -f environment.yml
```

Test your tensorflow version from python shell: 

```
python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3
```

### Notes: 

The syntax.py file is a script with basic tensorflow computations and operations. 
Ths conda environment for this repo was built with tensorflow v2, but this script ustilise command syntax from v1. 
You can switch between the two, check the headers at the top of the script for information on how. 