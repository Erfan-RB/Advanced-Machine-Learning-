#Human Activity Recognition
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

%matplotlib inline 

sns.set(style="whitegrid", palette="muted", font_scale=1.5)
RANDOM_SEED = 42

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()

reshaped_segments = np.asarray(
	segments, dtype = np.float32).reshape(
	-1 , N_time_steps, N_features)

reshaped_segments.shape
X_train, X_test, Y_train, Y_test = train_test_split(
	reshaped_segments, labels, test_size = 0.2, 
	random_state = RANDOM_SEED)

def create_LSTM_model(inputs):
	W = {
		'hidden': tf.Variable(tf.random_normal([N_features, N_hidden_units])),
		'output': tf.Variable(tf.random_normal([N_hidden_units, N_classes]))
	}
	biases = {
		'hidden': tf.Variable(tf.random_normal([N_hidden_units], mean = 0.1)),
		'output': tf.Variable(tf.random_normal([N_classes]))
	}
	X = tf.transpose(inputs, [1, 0, 2])
	X = tf.reshape(X, [-1, N_features])
	hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
	hidden = tf.split(hidden, N_time_steps, 0)

	lstm_layers = [tf.contrib.rnn.BasicLSTMCell(
		N_hidden_units, forget_bias = 1.0) for _ in range(2)]
	lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

	outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, 
										hidden, dtype = tf.float32)

	lstm_last_output = outputs[-1]
	return tf.matmul(lstm_last_output, W['output']) + biases['output']
L2_LOSS = 0.0015
l2 = L2_LOSS * \
sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits = pred_y, labels = Y)) + l2

Learning_rate = 0.0025
optimizer = tf.train.AdamOptimizer(learning_rate = Learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred_softmax , 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype = tf.float32))
N_epochs = 50
batch_size = 1024

saver = tf.train.Saver()
history = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[])
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train_count = len(X_train)

for i in range(1, N_epochs + 1):
	for start, end in zip(range(0, train_count, batch_size), 
						range(batch_size, train_count + 1, batch_size)):
		sess.run(optimizer, feed_dict={X: X_train[start:end],
									Y: Y_train[start:end]})
	_, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
		X: X_train, Y: Y_train})
	_, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
		X: X_test, Y: Y_test})
	history['train_loss'].append(loss_train)
	history['train_acc'].append(acc_train)
	history['test_loss'].append(loss_test)
	history['test_acc'].append(acc_test)

	if (i != 1 and i % 10 != 0):
		print(f'epoch: {i} test_accuracy:{acc_test} loss:{loss_test}')
predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], 
											feed_dict={X: X_test, Y: Y_test})
print()
print(f'final results : accuracy : {acc_final} loss : {loss_final}')
plt.figure(figsize=(12,8))

plt.plot(np.array(history['train_loss']), "r--", label="Train loss")
plt.plot(np.array(history['train_acc']), "g--", label="Train accuracy")

plt.plot(np.array(history['test_loss']), "r--", label="Test loss")
plt.plot(np.array(history['test_acc']), "g--", label="Test accuracy")

plt.title("Training session's progress over iteration")
plt.legend(loc = 'upper right', shadow = True)
plt.ylabel('Training Progress(Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()
max_test = np.argmax(Y_test, axis=1)
max_predictions = np.argmax(predictions, axis = 1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(16,14))
sns.heatmap(confusion_matrix, xticklabels = LABELS, yticklabels = LABELS, annot =True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel('Predicted_label')
plt.ylabel('True Label')
plt.show()

