#!/usr/bin/env python3

from sklearn.datasets import load_iris

def iris(): return load_iris()

from sklearn.model_selection import train_test_split

def train_test_data(data, first_label, sec_label): return train_test_split(data[first_label], data[sec_label], random_state=0)
