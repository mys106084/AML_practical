import numpy
import theano
import theano.tensor as T
import scipy.optimize as op
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix()
y = T.vector()
s_X = theano.shared(numpy.asarray(D[0], dtype="float64"), name='s_X')
s_Y = theano.shared(numpy.asarray(D[1], dtype="float64"), name='s_Y')

wb = theano.shared(rng.randn(feats+1), name="wb")
w = wb[:feats]
b = wb[feats:]
b = 0.0

print "Initial model:"

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
#
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
grad = T.grad(cost, wb)             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following s:ection of this tutorial)


batch_cost = theano.function([], outputs=cost, givens={x:s_X,y:s_Y})

batch_grad = theano.function([], outputs=grad, givens={x:s_X,y:s_Y})


def function(weights):
	wb.set_value(weights, borrow=True)
	return batch_cost()

def function_grad(weights):
	wb.set_value(weights, borrow=True)
	return batch_grad()

best_wb = op.fmin_l_bfgs_b(func=function,
	x0 = rng.randn(feats+1),
	fprime=function_grad,
	maxiter=1000)
predict = theano.function(inputs=[x], outputs=prediction)

print "Final model:"
print wb.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])




