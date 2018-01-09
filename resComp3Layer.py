#Reservoir computing 
#              ________  
#              |       |     
#              V       |  
#input--->o--->o--->o  |
#         |    |    |  | 
#         V    V    V  |
##         o--->o--->o--
#         |    |    |
#         V    V    V
#         o--->o--->o--->output
   
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
parser = argparse.ArgumentParser(description='define learning rate')

parser.add_argument('lR',type=float,default=1e-5,help='learning rate goes here')

args = parser.parse_args()
def cap(z0,z1):
    # Activation function based on a capacitor charge/discharge
    t = 3
    tau = 1
    fCap = (z1-z0) * (1-np.exp(-t/tau)) + z1
    return fCap


def one_hot(vocabSize,v):
    return np.eye(vocabSize)[v]

# Load data 

myFile = '../../geneDesigner/data/sequence.fasta'
#myFile = '/home/main/Desktop/Desktopception/projects/andrejsCharRNN/char-rnn/data/tinyshakespeare/input.txt'
data = open(myFile, 'r').read()  # Use this source file as input for RNN
chars = sorted(list(set(data)))
dataSize, vocabSize = len(data), len(chars)
print('Data has %d characters, %d unique.' % (dataSize, vocabSize))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Training settings
myIt = 100001
dispIt = 1000
lR = args.lR #1e-3
lRDecay = 1e-4
sampleSize = 512

# Parameters
## random seed
mySeed = 1337
myDim = 1024
mySqrt = int(np.sqrt(myDim))
## Sparsity coefficient
spCh = 0.05
## meta - Weight
#wtWt = 0.5000025/myDim**2

wtWt = 1.01/myDim**2
seq_length = 1
np.random.seed(mySeed)

seqLength = 1
# Define the Reservoir
# Layers
a0 = np.zeros((myDim,1))
a1 = np.zeros((myDim,1))

# Weights
    # input to a0
theta0 = np.random.random((vocabSize,myDim))
    # a0 to a0
thetaJ0_0 = wtWt * (np.random.random((myDim,myDim))-0.5) * (np.random.random((myDim,myDim)) < spCh)
    # a0 to a1
thetaJ0_1 = wtWt * (np.random.random((myDim,myDim))-0.5) * (np.random.random((myDim,myDim)) < spCh)
    # a1 to a0
thetaJ1_0 = wtWt * (np.random.random((myDim,myDim))-0.5) * (np.random.random((myDim,myDim)) < spCh)
    # a1 to a1
thetaJ1_1 = wtWt * (np.random.random((myDim,myDim))-0.5) * (np.random.random((myDim,myDim)) < spCh)
    # a1 to output
myDim1 = int(myDim/2)
myDim2 = int(myDim1/2)
myDim3 = int(myDim2/2)

theta1 =  1e-2*np.random.random((myDim,myDim1))
theta2 =  1e-2*np.random.random((myDim1,myDim2))
theta3 =  1e-2*np.random.random((myDim2,myDim3))

# weights from Reservoir output to predictions
Woo = 1e-2*np.random.random((myDim3,vocabSize))
# biases 
by = np.zeros((vocabSize,1))
br1 = np.zeros((myDim1,1))
br2 = np.zeros((myDim2,1))
br3 = np.zeros((myDim3,1))

   
def sample(a0,a1,seed_ix, n):
	""" 
	sample a sequence of integers from the model 
	a0, a1 is previous state of layers, seed_ix is seed letter for first time step
	"""
	x = np.zeros((vocabSize, 1))
	x[seed_ix] = 1
	#print(x[seed_ix])
	#print(ix_to_char[seed_ix])
	ixes = [seed_ix]
	#print(np.shape(x))
	for t in range(n):
		
		a0 = cap(a0,np.dot(theta0.T,x) + np.dot(thetaJ0_0.T,a0) +  np.dot(thetaJ1_0.T,a1))
		a1 = cap(a1,np.dot(thetaJ0_1.T,a0)+ np.dot(thetaJ1_1.T,a1))
		resOut1 = np.tanh(np.dot(theta1.T,a1)) + br1
		resOut2 = np.tanh(np.dot(theta2.T,resOut1)) + br2
		resOut3 = np.tanh(np.dot(theta3.T,resOut2)) + br3
		#print(np.shape(out))
		y = np.dot(Woo.T,resOut3) + by 
		p = np.exp(y) / np.sum(np.exp(y))
		#print(np.shape(p))
		ix = np.random.choice(range(vocabSize), p=p.ravel())
		x = np.zeros((vocabSize, 1))
		x[ix] = 1
		ixes.append(ix)
	return ixes



def funLoss(inputs,targets,a0,a1):
	""""""
	xs, a0s, a1s, resOut1,resOut2,resOut3, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}
	a0s[-1] = np.copy(a0)
	a1s[-1] = np.copy(a1)
	myLoss = 0
	
	# Forward Pass
	for t in range(len(inputs)):
		xs[t] = np.zeros((vocabSize,1))
		xs[t][inputs[t]] = 1
		a0s[t] = cap(a0s[t-1],np.dot(theta0.T,xs[t]) + np.dot(thetaJ0_0.T,a0s[t-1]) +  np.dot(thetaJ1_0.T,a1))
		a1s[t] = cap(a1s[t-1],np.dot(thetaJ0_1.T,a0s[t-1])+ np.dot(thetaJ1_1.T,a1s[t-1]))
		#resOut[t] = np.tanh(np.dot(theta1.T,a1s[t-1])) + br
		resOut1[t] = np.tanh(np.dot(theta1.T,a1s[t])) + br1
		resOut2[t] = np.tanh(np.dot(theta2.T,resOut1[t])) + br2
		resOut3[t] = np.tanh(np.dot(theta3.T,resOut2[t])) + br3
		#ys[t] = np.dot(Woo.T,a1s[t]) + by
		ys[t] = np.dot(Woo.T,resOut3[t]) + by
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) 
		myLoss += -np.log(ps[t][targets[t],0]) #CE loss
	# Backward pass
	dWoo, dtheta1, dtheta2, dtheta3 = np.zeros_like(Woo), np.zeros_like(theta1), np.zeros_like(theta2), np.zeros_like(theta3)
	dby, dbr1, dbr2, dbr3 = np.zeros_like(by), np.zeros_like(br1), np.zeros_like(br2), np.zeros_like(br3)
	for t in reversed(range(len(inputs))):
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
		
		#print(np.shape(dy),np.shape(resOut3[t]),np.shape(dWoo))
		#print(np.shape(np.dot(dy,resOut3[t].T)))
		
		dWoo += np.dot(dy,resOut3[t].T).T
		#dWoo +=	np.dot(a1s[t],dy.T)
		dby += dy
		dr3 = np.dot(Woo, dy)
		dr3Temp =  (1 - resOut3[t] * resOut3[t]) * dr3
		dbr3 += dr3Temp
		dtheta3 += np.dot(dr3Temp,resOut2[t].T).T

		dr2 = np.dot(theta3, dr3)
		dr2Temp =  (1 - resOut2[t] * resOut2[t]) * dr2
		dbr2 += dr2Temp
		dtheta2 += np.dot(dr2Temp,resOut1[t].T).T

		dr1 = np.dot(theta2, dr2)
		dr1Temp =  (1 - resOut1[t] * resOut1[t]) * dr1
		dbr1 += dr1Temp
		dtheta1 += np.dot(dr1Temp,a1s[t].T).T

	for dparam in [dWoo, dtheta3, dtheta2, dtheta1, dby, dbr3, dbr2, dbr1]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	return myLoss, dWoo, dtheta3, dtheta2, dtheta1, dby, dbr3, dbr2, dbr1, a0s[len(inputs)-1], a1s[len(inputs)-1]

n, p = 0, 0
mWoo, mtheta3, mtheta2, mtheta1 = np.zeros_like(Woo), np.zeros_like(theta3), np.zeros_like(theta2), np.zeros_like(theta1)
mbr1, mbr2, mbr3, mby = np.zeros_like(br1), np.zeros_like(br2), np.zeros_like(br3), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocabSize)*seq_length # loss at iteration 0

for k in range(myIt):

	if (p+seq_length+1 >= len(data) or k == 0):
		a0prev = np.zeros((myDim,1))
		a1prev = np.zeros((myDim,1))
		p = 0 
	
	inputs = [char_to_ix[ch] for ch in data[p:p+seqLength]]
	targets = [char_to_ix[ch] for ch in data[p+1:p+seqLength+1]]
	# sample occasionally	
	if(k % dispIt == 0):
		#mySampleInput = inputs[int((vocabSize-1)*np.random.random(1))]
		#sample_ix = sample(a0,a1,inputs[0], sampleSize)
		sample_ix = sample(a0,a1,char_to_ix['>'], sampleSize)
		txt = ''.join(ix_to_char[ix] for ix in sample_ix)
		print('----\n %s \n----' % (txt, ))

		
	myLoss, dWoo, dtheta3, dtheta2, dtheta1, dby, dbr3, dbr2, dbr1, a0prev, a1prev = funLoss(inputs,targets,a0prev,a1prev)

	smooth_loss = smooth_loss * 0.999 + myLoss * 0.001
	
	if(k % dispIt == 0): #give us a training update
		print('iter %d, lR: %.3e loss: %f, smooth loss: %f' % (k, lR, myLoss, smooth_loss)) # print progress

	for param, dparam, mem in zip([Woo, theta3, theta2, theta1, br3, br2, br1, by],
					[dWoo, dtheta3, dtheta2, dtheta1, dbr3, dbr2, dbr1, dby],
					[mWoo, mtheta3, mtheta2, mtheta1, mbr3, mbr2, mbr1, mby]):
		mem += dparam * dparam
		param += -lR * dparam / np.sqrt(mem + 1e-8) #Adagrad update. need to look this up: 
	p += seq_length
	lR *= (1-lRDecay)
	







