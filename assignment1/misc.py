def iter_over_pairs(pairs):
    """
    Return an iterator over pairs present in the 'pairs' input.

    :type pairs: dictionary or iterable
    :param pairs: The pairs to iterate upon. These may be stored either as
    (key, value) items in a dictionary, or directly as pairs in any kind of
    iterable structure

    :rtype: iterable
    :returns: an iterable yielding pairs

    """
    if isinstance(pairs, dict):
        return pairs.iteritems()
    else:
        return pairs


import matplotlib.pyplot as plt
import matplotlib

def bgd_visualization(X, y, theta_hist, loss_function, X_validation=None, y_validation=None):
    """
    visulaize the loss in batch gradient descent
        X-axis: iteration
        y-axis: the loss function value
    """
    #TODO
    num_iter = theta_hist.shape[0]
    loss_hist = np.log([loss_function(X, y, theta_hist[i]) for i in range(num_iter)])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence plot")
    plt.plot(range(num_iter), loss_hist)
    plt.legend(["Training set"])
    print "Training: %r" %loss_function(X, y, theta_hist[num_iter-1])
    
    if (X_validation != None) and (y_validation != None):
        loss_hist_val = np.log([loss_function(X_validation, y_validation, theta_hist[i]) for i in range(num_iter)])
        print "Validation: %r" %loss_function(X_validation, y_validation, theta_hist[num_iter-1])
        plt.plot(range(num_iter), loss_hist_val)
        plt.legend(["Training set", "Validation set"])
    plt.show()
    #plt.savefig()
