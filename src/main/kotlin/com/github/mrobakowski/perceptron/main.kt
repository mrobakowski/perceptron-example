package com.github.mrobakowski.perceptron

import golem.mat

fun main(args: Array<String>) {
    val p = Perceptron(3)
    print(p.response(mat[1, 2, 3]))
}