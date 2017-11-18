import numpy as np
import tensorflow as tf

'''
Task

count = 0
for i in input_string:
    if i == '1':
        count+=1
'''

NUM_CLASSES = 21
NUM_EXAMPLES = 10000
NUM_LSTM_HIDDEN = 24
BATCH_SIZE = 1000
EPOCH = 2000

# Generate Data
'''
[
 array([[0],[0],[1],[0],[0],[1],[0],[1],[1],[0],[0],[0],[1],[1],[1],[1],[1],[1],[0],[0]]), 
 array([[1],[1],[0],[0],[0],[0],[1],[1],[1],[1],[1],[0],[0],[1],[0],[0],[0],[1],[0],[1]]), 
 .....
]

[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
This is a sample output for a sequence which belongs to 4th class i.e has 4 ones

'''
train_input = ['{0:020b}'.format(i) for i in range(2**20)]
np.random.shuffle(train_input)
train_output = np.array([sum(map(lambda x: int(x),i)) for i in train_input])
train_output = np.eye(NUM_CLASSES)[train_output]
train_input = np.array([list(map(lambda x: [int(x)],i)) for i in train_input])


# Split 
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] #everything beyond 10,000
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES] #till 10,000


#Model
data = tf.placeholder(tf.float32, [None, 20, 1])
target = tf.placeholder(tf.float32, [None, NUM_CLASSES])


cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_LSTM_HIDDEN,state_is_tuple=True)
lstm_outputs, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32,
			time_major=False)

lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
last = lstm_outputs[-1,:,:]

weight = tf.Variable(tf.truncated_normal([NUM_LSTM_HIDDEN, NUM_CLASSES]))
bias = tf.Variable(tf.random_normal([1,NUM_CLASSES]))
	
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	no_of_batches = int(len(train_input)/BATCH_SIZE)
	for i in range(EPOCH):
		ptr = 0
		for j in range(no_of_batches):
			sess.run(optimizer,{data: train_input[ptr:ptr+BATCH_SIZE], target: train_output[ptr:ptr+BATCH_SIZE]})
			ptr+=BATCH_SIZE
		if i%100 == 0:
			incorrect = sess.run(error,{data: train_input, target: train_output})
			print('Epoch {:2d} error {:3.3f}%'.format(i, 100 * incorrect))
	incorrect = sess.run(error,{data: test_input, target: test_output})
	print('TEST error {:3.3f}%'.format(100 * incorrect))