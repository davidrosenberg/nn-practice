import numpy
import theano
import theano.tensor as T
import cPickle
import gzip
import os


def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    print '... loading data'
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    ret_datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                    (test_set_x, test_set_y)]
    info = {'xdim': train_set[0].shape[1],
            'y_num_categories': 1 + max(train_set[1])}
    rval = {'data': ret_datasets, 'info': info}
    return rval


if __name__ == '__main__':
    data_path = "/home/drosen/repos/DeepLearningTutorials/data"
    datasets = load_data(dataset=os.path.join(data_path, 'mnist.pkl.gz'))
    print datasets["info"]["xdim"]
    print datasets["info"]["y_num_categories"]
