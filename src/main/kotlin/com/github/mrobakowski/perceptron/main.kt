package com.github.mrobakowski.perceptron

import golem.mat
import java.util.*

fun main(args: Array<String>) {
    val p = Perceptron(2)
    p.learn(generateDataPoints(), maxEpochs = 100)
    print(p.response(mat[1, 2]))
}

fun generateDataPoints(num: Int = 100, dim: Int = 2): List<DataPoint> {
    val rand = Random()
    return (1..num).map {
        val x = rand.nextDouble()
        val y = rand.nextDouble()
        val res = if (x + y < 1) 1.0 else 0.0
        DataPoint(mat[rand.nextDouble(), rand.nextDouble(), res])
    }
}