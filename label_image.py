import tensorflow as tf, sys

image_path = sys.argv[1]

# The path where to find the image
# The file is provided by python image_classifier.py ./images/image2.jpg
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Load the generated label files for labeling the image
# the rstrip prevent that return characters are also read
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Load the im age and predict the label
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor,
             {'DecodeJpeg/contents:0': image_data})

    # Use the outcome of the prediction to show the first label that could
    # be it, followed by the next option, etc
    # The labels are ordered by confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    # Take the list with scores, and print them out
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
