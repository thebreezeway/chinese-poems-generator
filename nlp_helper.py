import numpy as np

def load_words_and_embeddings(path):
    
    try:
        return  np.load(path+'/words.npy')[()],\
                np.load(path+'/embedding_vec.npy')[()]

    except FileNotFoundError:
        print("未在以下路径找到相应embedding数据:\n" + \
                path + '/words.npy' + \
                path + '/embedding_vec.npy'
        )
    
def load_char_embeddings(path):
    '''
    Returns:
    char_to_index -- dict1
    ndex_to_char -- dict2
    index_to_vec -- dict3
    '''
    try:
        return np.load(path+'/char_to_index.npy')[()], \
            np.load(path+'/index_to_char.npy')[()], \
            np.load(path+'/index_to_vec_char.npy')[()]
    except FileNotFoundError:
        print("未在以下路径找到相应embedding数据:\n" + \
                path + '/char_to_index.npy\n' +\
                path + '/index_to_char.npy\n' +\
                path + '/index_to_vec_char.npy'
        )
def load_corpus(path):

    with open(path, 'r') as f:
        data = f.readlines()

    return data
