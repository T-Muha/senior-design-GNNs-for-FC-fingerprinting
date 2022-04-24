import numpy as np

def split(full_dataset, full_labels, ids, train_size, valid_size, test_size):
    """Splits data when not using cross-validation"""

    train_pairs = full_dataset[:train_size]
    train_labels = full_labels[:train_size]
    train_ids = ids[:train_size]
    valid_pairs = full_dataset[train_size:train_size+valid_size]
    valid_labels = full_labels[train_size:train_size+valid_size]
    valid_ids = ids[train_size:train_size+valid_size]
    test_pairs = full_dataset[train_size+valid_size:train_size+valid_size+test_size]
    test_labels = full_labels[train_size+valid_size:train_size+valid_size+test_size]
    test_ids = ids[train_size+valid_size:train_size+valid_size+test_size]

    return train_pairs, train_labels, train_ids, valid_pairs, valid_labels, valid_ids, test_pairs, test_labels, test_ids
    

def cross_validation_split(full_dataset, full_labels, ids, valid_size, test_size, fold):
    """Splits data when using cross validation."""

    test_start = test_size * fold
    test_end = test_size * (fold + 1)

    test_pairs = full_dataset[test_start:test_end]
    test_labels = full_labels[test_start:test_end]
    test_ids = ids[test_start:test_end]

    excl_test_pairs = np.concatenate((full_dataset[:test_start], full_dataset[test_end:]))
    excl_test_labels = np.concatenate((full_labels[:test_start], full_labels[test_end:]))
    excl_test_ids = np.concatenate((ids[:test_start], ids[test_end:]))

    valid_indices = np.random.choice(range(len(excl_test_pairs)), valid_size, replace=0)

    train_pairs = np.delete(excl_test_pairs, valid_indices, axis=0)
    train_labels = np.delete(excl_test_labels, valid_indices, axis=0)
    train_ids = np.delete(excl_test_ids, valid_indices, axis=0)

    valid_pairs = excl_test_pairs[valid_indices]
    valid_labels = excl_test_labels[valid_indices]
    valid_ids = excl_test_ids[valid_indices]

    return train_pairs, train_labels, train_ids, valid_pairs, valid_labels, valid_ids, test_pairs, test_labels, test_ids