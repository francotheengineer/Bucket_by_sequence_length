import tensorflow as tf


def random_shuffle_queue(seq_len, q_input_tensor_1, q_input_tensor_2):

    #get shapes to then reshape after the random shuffle queue
    _tensor_1_shape = q_input_tensor_1 .get_shape()
    _tensor_2_shape = q_input_tensor_2.get_shape()
    _seq_len_shape = seq_len.get_shape()

    randomshufflequeue = tf.RandomShuffleQueue(
        capacity=100,
        dtypes=[tf.int64, tf.int64, tf.int64],
        min_after_dequeue=0,
        name="random_shuffle_queue")

    enqueue = randomshufflequeue.enqueue((q_input_tensor_1, q_input_tensor_2, seq_len))
    qr = tf.train.QueueRunner(randomshufflequeue, [enqueue])
    tf.train.add_queue_runner(qr)

    try:
        dequeue_tensor_1, dequeue_tensor_2, dequeue_seq_len = randomshufflequeue.dequeue()
        dequeue_tensor_1.set_shape(_tensor_1_shape)
        dequeue_tensor_2.set_shape(_tensor_2_shape)
        dequeue_seq_len.set_shape(_seq_len_shape)

    except tf.errors.OutOfRangeError:
        randomshufflequeue.close()
        print('Done - Qut of range - Queue closed')


    return (dequeue_seq_len, dequeue_tensor_1, dequeue_tensor_2)

