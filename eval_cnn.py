import tensorflow as tf
import data_helpers 
import numpy as np 


checkpoint_dir = '/home/zhangjw/anzhentmp/runs/1522247512/checkpoints'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

x_test_text, y_test = data_helpers.load_text_and_labels('../data/testset.txt')


graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        
        input_x = graph.get_operation_by_name('input_x').outputs[0]

        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        
        sess.run(tf.tables_initializer())

        predictions = graph.get_operation_by_name('output/predictions').outputs[0]
        
        max_sentence_length = input_x.shape[1]
        batches = data_helpers.batch_iter(x_test_text, y_test, 64, 1, max_sentence_length, shuffle=False)

        all_predictions = []
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch, y_batch = list(x_batch), list(y_batch)
            batch_predictions = sess.run(predictions, feed_dict={input_x: x_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

y_test = np.array(list(map(lambda a: a.index(1), y_test)))
correct_predictions = sum(all_predictions == y_test)
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/len(y_test)))

