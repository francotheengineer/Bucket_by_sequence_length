# Tensorflow's bucket_by_sequence_length Example

### Example on how to use a Tensorflow Queue, bucket_by_sequence_length and TFRecords to feed data to your model.

Compatible with Tensorflow 1.x and Python 3.4 . Please upgrade if you have Tensorflow 0.x.
Note: Please bear in mind that this was designed for a word level rnn and hence coupled input and output tensors were needed hence the tupple being passed to bucket_by_sequence_length..

<div align="center">  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" width="200"><br><br></div>
This is an dirty implementation of this function using TFRecords. I will be commenting this soon, in the meantime please make an issue for help.
Tensorflow 1.2 and Python 3.4

## How to use it?

```
git clone git@github.com:/francotheengineer/Bucket_by_sequence_length.git 
python3 main.py
```

## Output: 

```python
 length [5 5 5]
[[4 4 4 4 0] [4 4 4 4 0] [5 5 5 5 5]] [[9 9 9 9 0] [9 9 9 9 0] [9 9 9 9 9]]

length [5 5 5]
[[5 5 5 5 5] [5 5 5 5 5] [4 4 4 4 0]] [[9 9 9 9 9] [9 9 9 9 9] [9 9 9 9 0]]

length [5 5 5]
[[5 5 5 5 5] [3 3 3 0 0] [3 3 3 0 0]] [[9 9 9 9 9] [9 9 9 0 0] [9 9 9 0 0]]

length [1 1 1]
[[1] [1] [1]] [[9] [9] [9]]

length [5 5 5]
[[5 5 5 5 5] [3 3 3 0 0] [3 3 3 0 0]] [[9 9 9 9 9] [9 9 9 0 0] [9 9 9 0 0]]

length [2 2 2]
[[2 2] [2 2] [2 2]] [[9 9] [9 9] [9 9]]

length [4 4 4]
[[3 3 3 0] [4 4 4 4] [4 4 4 4]] [[9 9 9 0] [9 9 9 9] [9 9 9 9]]

Done - out of range
 ```
 
## For Search Purposes:

from tensorflow.contrib.training.python.training.bucket_ops /import bucket_by_sequence_length
