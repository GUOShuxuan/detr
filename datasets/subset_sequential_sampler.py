"""Subset sequential sampler."""

from copy import deepcopy
import random

from torch.utils.data.sampler import Sampler


class SubsetSequentialSampler(Sampler):
    """SubsetSequentialSampler.

    Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        """Constructor."""
        self.indices = indices

    def __iter__(self):
        """"Iter."""
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        """Length."""
        return len(self.indices)


class BalancedSubsetSequentialSampler(Sampler):
    """BalancedSubsetSequentialSampler.

    Samples in a balanced way images ensuring that exactly half of the images have labels.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices, supervised, unsupervised):
        """Constructor."""
        self.indices = indices
        random.shuffle(supervised)
        random.shuffle(unsupervised)
        self.len_supervised = len(supervised)
        self.len_unsupervised = len(unsupervised)
        if self.len_supervised > self.len_unsupervised:
            ratio = self.len_supervised // self.len_unsupervised
            module = self.len_supervised % self.len_unsupervised
            self.supervised = supervised
            self.unsupervised = ratio * unsupervised + unsupervised[:module]
        else:
            ratio = self.len_unsupervised // self.len_supervised
            module = self.len_unsupervised % self.len_supervised
            self.unsupervised = unsupervised
            self.supervised = ratio * supervised + supervised[:module]
        print(len(self.supervised), len(self.unsupervised))

    def _add_lists_alternatively(self, lst1, lst2):
        random.shuffle(lst1)
        random.shuffle(lst2)
        return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]

    def __iter__(self):
        """"Iter."""
        all_indices = self._add_lists_alternatively(self.supervised, self.unsupervised)
        return (all_indices[i] for i in range(len(all_indices)))

    def __len__(self):
        """Length."""
        return len(self.len_supervised) + len(self.unsupervised)


class BalancedSampler(Sampler):
    """BalancedSampler.

    Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices, supervised, unsupervised, ratio=1):
        """Constructor."""
        self.indices = indices
        self.supervised = supervised
        self.unsupervised = unsupervised
        self.new_list = []
        self.ratio = ratio

    def _create_balanced_batch(self):
        copy_supervised = deepcopy(self.supervised)
        copy_unsupervised = deepcopy(self.unsupervised)
        random.shuffle(copy_supervised)
        random.shuffle(copy_unsupervised)
        self.new_list = []
        while copy_supervised:
            self.new_list.append(copy_supervised.pop())
            if len(copy_unsupervised) < self.ratio:
                copy_unsupervised = deepcopy(self.unsupervised)
                random.shuffle(copy_unsupervised)
            for _ in range(self.ratio):
                self.new_list.append(copy_unsupervised.pop())
        return self.new_list

    def __iter__(self):
        """Iter."""
        all_indices = self._create_balanced_batch()
        return (all_indices[i] for i in range(len(all_indices)))

    def __len__(self):
        """Length."""
        return len(self.new_list)
