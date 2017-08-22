import tensorflow as tf


def make_sequence_example(seq_length, inputs1, inputs2):

    context_features = {
        'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_length]))
    }
    context = tf.train.Features(feature=context_features)
    # Feature lists for the two sequential features of our example

    input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[input_])) for input_ in inputs1]
    input_features2 = [tf.train.Feature(int64_list=tf.train.Int64List(value=[input_])) for input_ in inputs2]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'inputs2': tf.train.FeatureList(feature=input_features2),
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)

    return tf.train.SequenceExample(context=context ,feature_lists=feature_lists)
