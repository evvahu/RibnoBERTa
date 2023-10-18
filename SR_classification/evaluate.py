import torch
import numpy as np 
#from utils import list_roles
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.cluster import KMeans




def get_accuracy(predictions, labels):
    """
    Given an list of class predictions and a list of labels, this method returns the accuracy
    :param predictions: a list of class predictions [int]
    :param labels: a list of labels [int]
    :return: [float] accuracy for given predictions and true class labels
    """
    correct = 0
    for p, l in zip(predictions, labels):
        if p == l:
            correct += 1
    accuracy = correct / len(predictions)
    return correct, accuracy


def get_balanced_accuracy(results):
    #Accuracy = (TP + TN) / (TP+FN+FP+TN)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for r in results:
        if r == 'hit':
            tp += 1
        elif r == 'correct':
            tn += 1
        elif r == 'miss':
            fp += 1
        else:
            fn += 1
    try:
        accuracy = (tp+tn)/(tp + tn + fp + fn)
        specificity = tn/(tn+fp)   # Sensitivity= TP / (TP + FN), Specificity =TN / (TN + FP)
        sensitivity = tp/(tp+fn)
        balanced_accuracy = (specificity + sensitivity)/2
    except:
        print('dividing by zero')
        accuracy = 0
        balanced_accuracy = 0
    return accuracy, balanced_accuracy


def calculate_accuracy(embeddings, labels, threshold):
    #nr_a, nr_p = count_nr_roles(info)
    #role_list = list_roles(info)
    sim_matrix = similarity_matrix(embeddings)
    results = []
    for i in range(0, sim_matrix.shape[0]):
        for j in range(0, sim_matrix.shape[1]):
            first = labels[i]
            second = labels[j]
            sim = sim_matrix[i,j]
            res = get_miss_or_hit(first, second, sim, threshold)
            results.append(res)
    acc, b_acc = get_balanced_accuracy(results)
    return np.round(acc, decimals=2), np.round(b_acc, decimals=2)
   # return results

def cluster_embeddings(embeddings):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(embeddings)
    return kmeans.labels_

    """
    As = torch.zeros([nr_a,256])
    Ps = torch.zeros([nr_p,256])
    a_idx = 0
    p_idx = 0
    for i, inf in enumerate(info):
        emb = embeddings[i]
        if inf[0] == 'A':
            As[a_idx] = emb
            a_idx += 1
        elif inf[0] == 'P':
            Ps[p_idx] = emb
            p_idx +=1 
        else:
            print('SR not available')
    """
def similarity_matrix(first_array, second_array = False, measure='cosine'):
    if second_array:
        return pairwise_distances(first_array, second_array, metric = measure)
    return pairwise_distances(first_array, metric = measure)


def get_miss_or_hit(cat1, cat2, sim, threshold):
    if cat1 == cat2:
        if sim > threshold:
            return 'hit'
        else:
            return 'miss'
    else:
        if sim > threshold:
            return 'false'
        else:
            return 'correct'


