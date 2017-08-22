import tensorflow as tf
import os
from make_sequence_example import make_sequence_example



def tf_record_writer(input_tensors, output_filename):
        sequence_1, sequence_2 = input_tensors
        output_file = os.path.join(os.getcwd(), output_filename )
        writer = tf.python_io.TFRecordWriter(output_file)
        #max_length = 0
        for _sequence_1, _sequence_2 in zip(sequence_1, sequence_2):
            #if max_length < len(_sequence_1):
            #    max_length = len(_sequence_1)

            ex = make_sequence_example(len(_sequence_1), _sequence_1, _sequence_2)
            writer.write(ex.SerializeToString())

        writer.close()
        #return max_length


def tf_record_reader(input_filename):

        ## 2: deserialize/read part
        tf.reset_default_graph()
        file_list = [os.path.join(os.getcwd(), input_filename)]
        file_queue = tf.train.string_input_producer(file_list, num_epochs=5)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)

        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_features = {
            "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "inputs2": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example
        parsed_context, parsed_sequence = tf.parse_single_sequence_example(
            context_features=context_features,
            serialized=serialized_example,
            sequence_features=sequence_features)

        # Batch the variable length tensor with dynamic padding
        parsed_input_1 = parsed_sequence['inputs']
        parsed_input_2 = parsed_sequence['inputs2']
        parsed_length = parsed_context['length']
        return (parsed_length, parsed_input_1, parsed_input_2)