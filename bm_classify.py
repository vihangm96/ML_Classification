import numpy as np

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    Y = np.where(y==0,-1,y)
    #Y = np.array([Y])
    multiplier = step_size/N

    w = np.zeros(D)
    b = 0

    ##########new########33
    '''
        locY = np.where(y==0, -1, y)
        locX = np.concatenate( (np.ones((N, 1)), X) , 1)
        locW = np.insert(w, 0, b)

        if loss == "perceptron":
            ############################################
            # TODO 1 : Edit this if part               #
            #          Compute w and b here            #
            w = np.zeros(D)
            b = 0
            yx = locY[:, None] * locX
            for j in range(max_iterations):
                predY = np.matmul( locX, locW)
                sum =  np.multiply(locY, predY)

                for i in range(N):
                    if sum[i] <= 0:
                        locW = locW + ( (step_size/N)*yx[i] )

            b = locW[0]
            w = locW[1:]
            ############################################
    '''
    #######################

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #

        X = np.concatenate( (np.ones((N, 1)), X) , 1)
        y_x = np.array([Y]).transpose() * X
        W = np.insert(w, 0, b)
        for i in range(max_iterations):
            validator =  np.multiply(Y, np.matmul( X, W))
            for n in range(N):
                if validator[n] <= 0:
                    W += ( multiplier*y_x[n] )
        b = W[0]
        w = W[1:]

        '''
        Y = np.array([Y]).T
        w = w[np.newaxis].T

        for i in range(max_iterations):
            validator_XY = np.dot((X * Y).T,np.where(Y * (np.dot(X,w) + b) >=0,1,0))
            w = np.add(w,validator_XY * multiplier)
            b += np.sum(Y)*multiplier
        w = w.reshape((D))
        '''
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        '''
        w = np.array([w])

        for i in range(max_iterations):
            sigmoidArr = sigmoid((np.dot(w,X.transpose())+b) * -Y)
            w = np.add(w,np.dot(sigmoidArr,np.multiply(X,Y.transpose()))*multiplier)
            b += multiplier* np.dot(Y,sigmoidArr.transpose())
        '''
        for i in range(max_iterations):
            updateArr = sigmoid(-Y * (np.dot(X,w)+b) ) * Y
            w += multiplier * np.dot(updateArr.transpose(),X)
            b += multiplier * np.sum(updateArr)
        ############################################

    else:
        raise "Loss Function is undefined."

    w = w.reshape((D))
    assert w.shape == (D,)
    return w, b

def sigmoid(z):

    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1/(1+np.exp(-z))
    ############################################

    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.where(np.dot(X,w.transpose()) + b < 0, 0, 1)
        '''
        for n in range(N):
            pred = np.dot(w.transpose(),X[n]) + b
            if(pred<0):
                preds.append(0)
            else:
                preds.append(1)
        '''
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.where(sigmoid(np.dot(X,w.transpose()) + b) < 0.5, 0, 1)
        '''
        for n in range(N):
            pred = sigmoid(np.dot(w.transpose(),X[n]) + b)
            if(pred<0.5):
                preds.append(0)
            else:
                preds.append(1)
        '''
        ############################################


    else:
        raise "Loss Function is undefined."

    #preds = np.array(preds)
    assert preds.shape == (N,)
    return preds



def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."


    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    C = len(b)
    w = np.array(w)
    preds = []
    for n in range(N):
        maxProbYet = -1
        maxClsIdxYet = -1

        for c in range(C):
            currentProb = np.dot(w[c].transpose(),X[n]) + b[c]

            if(currentProb > maxProbYet):
                maxProbYet = currentProb
                maxClsIdxYet = c
        preds.append(maxClsIdxYet)

    ############################################
    preds = np.array(preds)
    assert preds.shape == (N,)
    return preds