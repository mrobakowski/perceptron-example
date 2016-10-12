package com.github.mrobakowski.perceptron
import golem.mapMat
import golem.mat
import golem.matrix.Matrix
import golem.randn

class Perceptron(
        val numInputs: Int,
        val activationFunction: (Matrix<Double>) -> Matrix<Double> = { it.mapMat { (it >= 0).toDouble() } }
) {
    var weights = randn(numInputs + 1 /*bias*/) / 2
    fun response(x: Matrix<Double>) = activationFunction(weights * x.extendWithColumns(mat[1]).T)
}