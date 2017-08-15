import scipy.io
import sys
import pickle
import numpy as np

if __name__ == '__main__':
    mat = scipy.io.loadmat(sys.argv[1])
    make_names = mat['make_names']
    make_names_list = []
    for make in make_names:
        make_names_list.append(make[0][0])
    print make_names_list
    print len(make_names_list)
    with open('make_names.p', 'wb') as handle:
        pickle.dump(make_names_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model_names = mat['model_names']
    model_names_list = []
    for model in model_names:
        temp = str(model[0])
        temp = temp.replace("[", "")
        temp = temp.replace("]", "")
        temp = temp.replace("'", "")
        if len(temp) > 1:
            temp = temp[1:]
        model_names_list.append(temp)
    print model_names_list
    print len(model_names_list)
    print type(model_names_list[0])
    with open('model_names.p', 'wb') as handle:
        pickle.dump(model_names_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
