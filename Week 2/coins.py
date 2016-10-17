#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def create_coins():

    coins = np.random.randint(2, size=(1000, 10))

    return coins

def return_fractions(coins):

    fractions = np.sum(coins, axis=1)

    return fractions

def find_coins(coins, fractions, v, index):

    v[0, index] = fractions[0]

    v[1, index] = fractions[np.random.randint(1000)]

    v[2, index] = np.amin(fractions)

    return v

def main():

    v = np.zeros(shape=(3, 100000))

    for index in range(100000):

        print('Attempt no. ', index+1)

        coins = create_coins()

        fractions = return_fractions(coins)

        v = find_coins(coins, fractions, v, index)

    means = np.mean(v, axis=1)

    print(means/10)

if __name__ == "__main__":
    main()
