import numpy as np
import os
import pandas as pd
from numpy.random import randint
from random_FC import random_FC


def selective_randint(low, high, exclude):
    """
    Returns a random integer in the given range.
    Will not return integers in the exclusion list.
    """

    r_int = randint(low, high)
    return r_int if r_int not in exclude else selective_randint(low, high, exclude)


def subject_sample(unsampled, exclude, num_subjects, data_ids=None):
    """
    Evenly samples subjects. An unsampled pool of subjects is sampled from
    and reset when all subjects have been sampled once from it.
    """

    # Current subject is the only one left
    if len(unsampled) == 1 and unsampled[0] == exclude:
        sub, _ = subject_sample([x for x in range(num_subjects)], exclude, num_subjects)
        return sub, unsampled

    # Pick and remove a random subject that has not been sampled
    sub = exclude
    sub_ix = None
    while sub == exclude:
        sub_ix = randint(0, len(unsampled))
        sub = unsampled[sub_ix]
    del unsampled[sub_ix]

    # Reset once all subjects sampled
    if len(unsampled) == 0:
        unsampled = [x for x in range(num_subjects)]

    return sub, unsampled


def generate_FCs(dataset_type, num_subjects, num_days, num_nodes, negative_edges, greatest_k_edges, min_edge_weight):
    """
    Get FCs, with num_days FCs for each subject.
    Either the actual HCP data (dataset_type = 'FC')
    or a random FC for each scan (dataset_type == 'scan_random')
    """

    try:
        import google.colab
        directory = 'drive/MyDrive/3T_HCP1200_MSMAll_d100_ts2'
    except:
        directory = 'data/3T_HCP1200_MSMAll_d' + str(num_nodes) + '_ts2'

    IC_data = np.zeros((num_subjects, num_days, num_nodes, 1200))
    FC_data = np.zeros((num_subjects, num_days, num_nodes, num_nodes), dtype=np.float32)

    if dataset_type == 'scan_random':
        for i in range(num_subjects):
            for j in range(num_days):
                FC_data[i,j] = random_FC(num_nodes)
    elif dataset_type == 'FC':
        # Load fMRI time series and create FCs
        for i, filename in enumerate(os.listdir(directory)):
            if i < num_subjects:
                with open(directory + '/' + filename) as f:
                    # Read time series and compute FC of each of the subject's 4 scans
                    df = pd.read_csv(f, sep=' ', header=None)
                    data = np.transpose(df.values)

                    # Split data into multiple time series and get FCs as correlation matrices
                    IC_data[i] = np.split(data, num_days, axis=1)
                    FCs = np.array([np.corrcoef(x) for x in IC_data[i]], dtype=np.float32)
                
                    # Preprocess the subject's FCs
                    for j, FCj in enumerate(FCs):
                        # Zero diagonal, remove negative/clip edges
                        np.fill_diagonal(FCj, 0)

                        if not negative_edges: 
                            FCj = FCj.clip(min=0)
                        if greatest_k_edges:
                            values_ascending = np.sort(np.abs(np.triu(FCj).flatten()))
                            kth_value = values_ascending[-greatest_k_edges]
                            FCj[np.where(np.abs(FCj) < kth_value)] = 0                        
                        if min_edge_weight:
                            FCj[np.where(np.abs(FCj) < min_edge_weight)] = 0

                        FCs[j] = FCj

                    FC_data[i] = FCs

    return FC_data


def make_pairs(FC_data):
    """
    Make the same-subject and different-subject pairs.
    Creates all possible same-subject pairs from FC_data and a
    different-subject pair for every same-subject pair.

    Returns:

        diffsub_ids: used later when constructing additional different-subject pairs
                     to augment training data so that no two different-subject pair is
                     used twice.
    """

    # Same-subject and different-subject FC pairs
    FC_samesub = []
    FC_diffsub = []

    # Index of subjects/FCs for graphs in dataset
    samesub_ids = []
    diffsub_ids = []

    unsampled = [x for x in range(len(FC_data))]
    diffsub_distribution = np.zeros((len(FC_data)))

    for i in range(len(FC_data)):
        for j in range(4):
            for k in range(4):
                if k > j:
                    samepair = (FC_data[i,j], FC_data[i,k]) if np.random.choice([True, False]) else (FC_data[i,k], FC_data[i,j])
                    FC_samesub.append(samepair)

                    rand_sub, unsampled = subject_sample(unsampled, i, len(FC_data))

                    if rand_sub == i:
                        # Confident this won't happen but vital that it doesn't
                        raise ValueError("Selected Different Subject was actually Same Subject!")

                    diffsub_distribution[rand_sub] += 1

                    rand_FC = randint(0,4)

                    diffpair = (FC_data[i,j], FC_data[rand_sub, rand_FC]) if np.random.choice([True, False]) else (FC_data[rand_sub, rand_FC], FC_data[i,j])
                    FC_diffsub.append(diffpair)

                    samesub_ids.append([(i, j), (i, k)])
                    diffsub_ids.append([(i, j), (rand_sub, rand_FC)])

    samesub_ids = np.array(samesub_ids)
    diffsub_ids = np.array(diffsub_ids)

    samesub_labels = np.ones(len(FC_samesub))
    diffsub_labels = np.full(len(FC_diffsub), -1)

    pairs = np.concatenate((FC_samesub, FC_diffsub))
    labels = np.concatenate((samesub_labels, diffsub_labels))

    indices = np.random.permutation(len(pairs))
    shuffled_pairs = pairs[indices]
    shuffled_labels = labels[indices]

    ids = np.concatenate((samesub_ids, diffsub_ids))
    shuffled_ids = ids[indices]

    return shuffled_pairs, shuffled_labels, shuffled_ids


def construct_dataset(config):
    """
    Builds the dataset of same-subject/different-subject pairs.

    Config values used:
        dataset_type, num_subjects, num_daysnum_nodes, min_edge_weight,
        negative_edges, greatest_k_edges, num_augpairs

    Returns:
        pairs: pairs of FC graphs
        labels: class labels for each pair; -1 for different subs and 1 for same subs
        ids: index of (subject,FC) for FCs in the dataset
    """

    dataset_type = config['dataset_type']
    num_subjects = config['data']['num_subjects']
    num_days = config['data']['num_dats']
    num_nodes = config['data']['num_nodes']
    min_edge_weight = config['data']['min_edge_weight']
    negative_edges = config['data']['negative_edges']
    greatest_k_edges = config['data']['greatest_k_edges']

    # Same-sub, diff-sub FC pairs
    FC_samesub = []
    FC_diffsub = []

    if dataset_type == 'all_random':
        for i in range(6000):
            FC_samesub.append((random_FC(num_nodes), random_FC(num_nodes)))
            FC_diffsub.append((random_FC(num_nodes), random_FC(num_nodes)))
            ids = np.zeros(num_subjects*12, 2)
    else:
        FC_data = generate_FCs(dataset_type, num_subjects, num_days, num_nodes, negative_edges, greatest_k_edges, min_edge_weight)
        pairs, labels, ids = make_pairs(FC_data)

    return pairs, labels, ids