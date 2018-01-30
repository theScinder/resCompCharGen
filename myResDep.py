
# Dynamic reservoir RNN
#
# Inspired by vinkhux and karpathy
# https://gist.github.com/vinhkhuc/7ec5bf797308279dc587
# https://gist.github.com/karpathy/d4dee566867f8291f086


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time



#myFile = "./data/dullJack.txt"
#myFile = "./data/wikiRC.txt"
#myFile = "./schmitt1Res1.py"
myFile = "./data/thesisTex.txt"


data = open(myFile, "r").read()  # Use this source file as input for RNN
chars = sorted(list(set(data)))
dataSize, vocabSize = len(data), len(chars)

print("Data has %d characters, %d unique." % (dataSize, vocabSize))
charToIx = {ch: i for i, ch in enumerate(chars)}
ixToChar = {i: ch for i, ch in enumerate(chars)}

#Network Parameters
hiddenSize = 1024
#number of steps to unroll
seqLength = 256
lR = 1e-3
lRDecay = 1- 1e-5
# train for epochs and display every dispIt
epochs = 1e7
dispIt = 5e3
sampleLength = 256#1024

def oneHot(x):
	hotOne = np.eye(vocabSize)[x]
	return hotOne

inputs = tf.placeholder(shape=[None,vocabSize],dtype=tf.float32,name="inputs")
targets = tf.placeholder(shape=[None,vocabSize],dtype=tf.float32,name="targets")
a0PrevState = tf.placeholder(shape=[1,hiddenSize],dtype=tf.float32,name="a0Prev")
a1PrevState = tf.placeholder(shape=[1,hiddenSize],dtype=tf.float32,name="a1Prev")




# Weights
# Reservoir (not trainable)
spCh = 0.75
np.random.seed(1337)
sparse1 = (np.random.random((hiddenSize,hiddenSize)) < spCh)
sparse2 = (np.random.random((hiddenSize,hiddenSize)) < spCh)
Wxa0 = tf.Variable(tf.random_normal([vocabSize,hiddenSize],stddev=0.025),name="Wxa0",trainable=False)
Wa0a1 = tf.Variable(sparse1*tf.random_normal([hiddenSize,hiddenSize],stddev=0.0025),name="Wa0a1",trainable=False)
Wa1a0 = tf.Variable(sparse2*tf.random_normal([hiddenSize,hiddenSize],stddev=0.0025),name="Wa1a0",trainable=False)

#Linear classifier weights (trainable)
Wa1y = tf.Variable(tf.truncated_normal([hiddenSize,vocabSize],stddev=0.05),name="Wa0a1")

# biases 
ba0= tf.Variable(tf.truncated_normal([hiddenSize],stddev=0.1),name="ba0")
ba1= tf.Variable(tf.truncated_normal([hiddenSize],stddev=0.1),name="ba1")
by = tf.Variable(tf.truncated_normal([vocabSize],stddev=0.1),name="by")

def myRes(inputs, a0PrevState, a1PrevState):
	myData = tf.reshape(inputs,[-1,vocabSize])
	a0 = tf.nn.tanh(tf.matmul(myData,Wxa0)+tf.matmul(a1PrevState,Wa1a0)+ba0) 
	a1 = tf.nn.tanh(tf.matmul(a0,Wa0a1)+ba1)
	#a0 = tf.nn.tanh(a0PrevState+tf.matmul(myData,Wxa0)+tf.matmul(a1PrevState,Wa1a0)+ba0) 
	#a1 = tf.nn.tanh(a1PrevState+tf.matmul(a0,Wa0a1)+ba1)
	ys = tf.matmul(a1,Wa1y)+by

	return a0, a1, ys

with tf.variable_scope("Reservoir") as scope:
	ys = []
	for t, xst in enumerate(tf.split(inputs, seqLength, axis=0)):
		a0Prev, a1Prev, yst = myRes(xst,a0PrevState,a1PrevState) 
		ys.append(yst)	


outputSoftmax = tf.nn.softmax(ys[-1])
outputs = tf.concat(ys,axis=0)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets,logits=outputs))

minimizer = tf.train.AdamOptimizer(learning_rate = lR)
gradsAndVars = minimizer.compute_gradients(loss)

mySaver = tf.train.Saver()

# 
# gradient clipping
#gradClipping = tf.constant(5.0, name="grad_clipping")
#clippedGradsAndVars = []
#for grad, var in gradsAndVars:
#    clippedGrad = tf.clip_by_value(grad, -gradClipping, gradClipping)
#    clippedGradsAndVars.append((clippedGrad, var))

# gradient updates
#updates = minimizer.apply_gradients(clippedGradsAndVars)
updates = minimizer.apply_gradients(gradsAndVars)


# Initialize the session
mySess = tf.Session()
init = tf.global_variables_initializer()
mySess.run(init)

n = 0
p = 0
a0PrevVal = np.zeros([1,hiddenSize]) 
a1PrevVal = np.zeros([1,hiddenSize]) 

t0 = time.time()
# set up smooth loss and myAlpha coefficient for exponential averaging 
smoothLoss = -np.log(1.0/vocabSize)*seqLength
myAlpha = 0.999
myFile = open('./tfWikiRCOut.txt','a')
while n < epochs:
	if (p + seqLength + 1 >= len(data)) or (n == 0):
		# Reset 
		p = 0
		a0PrevVal = np.zeros([1,hiddenSize]) 
		a1PrevVal = np.zeros([1,hiddenSize]) 


	# Targets are one character in the future
	inputVals = [charToIx[ch] for ch in data[p:p+seqLength]]
	targetVals = [charToIx[ch] for ch in data[p+1:p+seqLength+1]]

	inputVals = oneHot(inputVals)
	targetVals = oneHot(targetVals)
	
	a0PrevVal, a1PrevVal, lossVal, _ = mySess.run([a0Prev, a1Prev, loss, updates],
						feed_dict={inputs: inputVals, 
						targets: targetVals,
						a0PrevState: a0PrevVal,
						a1PrevState: a1PrevVal})
	smoothLoss = myAlpha * smoothLoss + (1-myAlpha) * lossVal
		
	if (n % dispIt == 0):
		# Perform Sampling
		elapsed = time.time() - t0
		print('Iteration %d, loss: %.3e, smooth loss: %.3e, time elapsed: %.2f, learning rate: %.2e'%(n,lossVal,smoothLoss,elapsed,lR))

		startIx = np.random.randint(0, len(data) - seqLength)
		sampleSeqIx = [charToIx[ch] for ch in data[startIx:startIx + seqLength]]
		ixes = []
		samplea0Prev = np.copy(a0PrevVal)
		samplea1Prev = np.copy(a1PrevVal)	
		
		for t in range(sampleLength):
			sampleInputVals = oneHot(sampleSeqIx)
			sampleOutputSoftmaxVal, samplea0Prev, samplea1Prev = \
                		mySess.run([outputSoftmax, a0Prev, a1Prev],
                         	feed_dict={inputs: sampleInputVals, 
					a0PrevState: samplea0Prev,
					a1PrevState: samplea1Prev})

			ix = np.random.choice(range(vocabSize), p=sampleOutputSoftmaxVal.ravel())
			ixes.append(ix)
			sampleSeqIx = sampleSeqIx[1:] + [ix]

		txt = ''.join(ixToChar[ix] for ix in ixes)
		print('----\n %s \n----\n' % (txt,))
		myFile.write(txt)
	lR = lR * lRDecay
	p += seqLength
	n += 1
myFile.close()
mySaver.save(mySess,'./models/Res1/',global_step=n)









