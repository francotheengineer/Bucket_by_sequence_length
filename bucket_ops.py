import tensorflow as tf
from tensorflow.contrib.training.python.training.bucket_ops import bucket_by_sequence_length

def bucket_by_seq_len(seq_len, input_tensors):
    #detuple the tensors
    dequeue1, dequeue2 = input_tensors
    #seq_len was passed in in tf.int64 and is required to be tf.int32
    seq_len = tf.cast(seq_len, tf.int32)
    sequence_length, bucketed_tensors = bucket_by_sequence_length(input_length=seq_len,
                                                    tensors=[dequeue1, dequeue2],
                                                    batch_size=3,
                                                    bucket_boundaries=[2, 3, 6],
                                                    dynamic_pad=True,
                                                    allow_smaller_final_batch=True,
                                                    capacity=100,
                                                    keep_input=True
                                                    )
    #where sequence_length[0] is the pre-padding length of the sequences
    #sequence_length[1] is the post bucketing length
    return(sequence_length, bucketed_tensors)