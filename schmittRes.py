# Reservoir computing character prediction text generator
# Perturb a dynamic reservoir with character inputs, predict next character with linear classifier 
#              ________  
#              |       |     
#              V       |  random connections with some sparsity
#input--->o--->o--->o  |
#         |    |    |  | 
#         V    V    V  |
#         o--->o--->o--
#         |    |    |
#         V    V    V
#         o--->o--->o--->output --> linear classifier
#
# This implementation is inspired in no small part by Andrej Karpathy's char-rnn gist and blog post: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
#
#https://gist.github.com/karpathy/d4dee566867f8291f086 
# BSD Licence
   
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
parser = argparse.ArgumentParser(description='define learning rate')

parser.add_argument('lR',type=float,default=1e-5,help='learning rate goes here')

args = parser.parse_args()

def schmitt(z0,z1):
    # Based on a Schmitt trigger 
    # use tanh with hysteresis
    myTemp = np.tanh(z0)
    mySchmitt = np.tanh(myTemp+z1)
    return mySchmitt



# Load data 
# Replace with training text of your choice
myFile = './wikiRC.txt'

data = open(myFile, 'r').read()  # Use this source file as input for RNN
chars = sorted(list(set(data)))
dataSize, vocabSize = len(data), len(chars)
print('Data has %d characters, %d unique.' % (dataSize, vocabSize))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Training settings
myIt = 101    
myStart = 0
dispIt = 10
lR = args.lR #1e-3
lRDecay = 1e-6

# Parameters
## random seed
mySeed = 1337
myDim = 1024
mySqrt = int(np.sqrt(myDim))

## Sparsity coefficient
spCh = 0.75
wtWt =  (2**11)/myDim**2

np.random.seed(mySeed)

# Reservoir trains based on dynamics over a sequence of seqLength
seqLength = 1024
# Length to sample
sampleLength= 256

# Define the Reservoir
# Layers
a0 = np.zeros((myDim,1))

# Weights
# input to a0
theta0 = np.random.random((myDim,vocabSize))
# a0 to a0
thetaJ0_0 = wtWt * (np.random.random((myDim,myDim))-0.5) * (np.random.random((myDim,myDim)) < spCh)
    
# weights from Reservoir output to predictions
Wry = 1e-2*np.random.random((vocabSize,myDim))
# Uncomment to load pretrained weights for "All work and no play makes Jack a dull boy
#Wry = np.load('./weights/Woo1024.npy') 


# biases 
by = np.zeros((vocabSize,1))
ba0 = np.zeros((myDim,1))
   
def sample(a0,seed_ix, n,myIter):
	""" 
	sample a sequence of integers from the model 
	a0 is previous state of the reservoir, seed_ix is seed letter for first time step
	"""
	x = np.zeros((vocabSize, 1))
	x[seed_ix] = 1
	ixes = []
	#print(np.shape(x))
	for t in range(n):
		
		a0 = schmitt(a0,np.dot(theta0,x) + np.dot(thetaJ0_0,a0)) + ba0
		
		y = np.dot(Wry,a0) + by 
		p = np.exp(y) / np.sum(np.exp(y))
		
		ix = np.random.choice(range(vocabSize), p=p.ravel())
		x = np.zeros((vocabSize, 1))
		x[ix] = 1
		ixes.append(ix)
				
	return ixes



def funLoss(inputs,targets,a0):
	""""""
	xs, a0s, ys, ps = {}, {}, {}, {}
	a0s[-1] = np.copy(a0)

	myLoss = 0
	
	# Forward Pass
	for t in range(len(inputs)):
		xs[t] = np.zeros((vocabSize,1))
		xs[t][inputs[t]] = 1
		a0s[t] = schmitt(a0s[t-1],np.dot(theta0,xs[t]) + np.dot(thetaJ0_0,a0s[t-1])) + ba0

		ys[t] = np.dot(Wry,a0s[t]) + by

		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) 
		myLoss += -np.log(ps[t][targets[t],0]) #CE loss
	# Backward pass
	da0y = np.zeros_like(Wry)
	dby, = np.zeros_like(by), 

#	da0next = np.zeros_like(a0[0])	
	for t in reversed(range(len(inputs))):
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1 # backprop into y.
		da0y +=	np.dot(dy,a0s[t].T)
		dby += dy

		
	for dparam in [dby, da0y]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	return myLoss, dy, dby, da0y, a0s[len(inputs)-1]#, a1s[len(inputs)-1]

n, p = 0, 0
# memory variables for Adagrad
mWry  = np.zeros_like(Wry)
mby = np.zeros_like(by)
# smooth loss at iteration 0
smooth_loss = -np.log(1.0/vocabSize)*seqLength 
a0prev = np.zeros((myDim,1))
for k in range(myStart,myIt):

	if (p+seqLength+1 >= len(data) or k == 0):
		a0prev = np.zeros((myDim,1))
		
		p = 0 
	
	inputs = [char_to_ix[ch] for ch in data[p:p+seqLength]]
	targets = [char_to_ix[ch] for ch in data[p+1:p+seqLength+1]]
	# sample occasionally	
	if(k % dispIt == 0):
		mySampleInput = char_to_ix['A']
		sample_ix = sample(a0prev,mySampleInput, sampleLength,k)
		#sample_ix = sample(a0prev,inputs[int((vocabSize-1)*np.random.random(1))], sampleLength,k)
		txt = ''.join(ix_to_char[ix] for ix in sample_ix)		
		print('----\n %s \n----' % (txt, ))
		

		
	myLoss, dy, dby, da0y, a0prev = funLoss(inputs,targets,a0prev)

	smooth_loss = smooth_loss * 0.999 + myLoss * 0.001
	
	if(k % dispIt == 0): #give us a training update
		print('iter %d, lR: %.3e loss: %f, smooth loss: %f' % (k, lR, myLoss, smooth_loss)) # print progress

	for param, dparam, mem in zip([Wry, by],#, thetaJ0_1, thetaJ1_1, ba1, theta0, thetaJ0_0, thetaJ1_0, ba0],
					[da0y, dby],#, da01, da11, dba1, dxa0, da00,  da10, dba0],
					[mWry, mby]):#, mJ01, mJ11, mba1, mxa0, mJ00, mJ10, mba0]):
		mem += dparam * dparam
		param += -lR * dparam / np.sqrt(mem + 1e-8) #Adagrad update. need to look this up: 
	p += seqLength
	lR *= (1-lRDecay)
	n += 1

if(0):
	#Uncomment to save weights as .npy
	np.save('./weights/Wry1024.npy',Wry)






