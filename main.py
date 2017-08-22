#TODO: Add comments

import tensorflow as tf
import os
from random_shuffle_queue import random_shuffle_queue
from TFRecords_write_read import tf_record_reader, tf_record_writer
from bucket_ops import bucket_by_seq_len
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.reset_default_graph()

#define nested lists each containing a sequence. This example has been modelled for
#a word level rnn
sequences = [[1], [2,2], [3,3,3], [4,4,4,4], [5,5,5,5,5]]
sequences2 =[[9], [9,9], [9,9,9], [9,9,9,9], [9,9,9,9,9]]
output_filename = 'recordout.tfr'


max_length = tf_record_writer(input_tensors=(sequences, sequences2),
                              output_filename=output_filename)

parsed_length, parsed_input_1, parsed_input_2 = tf_record_reader(output_filename)

dequeue_seq_len, dequeue1, dequeue2 = random_shuffle_queue(parsed_length,
                                                           parsed_input_1,
                                                           parsed_input_2)

sequence_length, bucketed_tensors = bucket_by_seq_len(seq_len=dequeue_seq_len,
                                                      input_tensors=(dequeue1, dequeue2))
sequence_length = tf.reduce_max(sequence_length, axis=0)

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while tf.errors.OutOfRangeError:
            seq_len, buk = sess.run([sequence_length, bucketed_tensors])
            print("length", seq_len)
            print(buk)

    except tf.errors.OutOfRangeError:
        print('Done - out of range')
    finally:
        coord.request_stop()
