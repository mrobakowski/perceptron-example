package com.github.mrobakowski.perceptron

import golem.*
import golem.matrix.Matrix

class Perceptron(
        numInputs: Int,
        val activationFunction: (Matrix<Double>) -> Matrix<Double> = { it.mapMat { (it >= 0.0).toDouble() } }
) {
    var weights = randn(numInputs + 1)

    fun response(x: Matrix<Double>) = activationFunction(net(x))
    fun net(x: Matrix<Double>) = weights * x.extendWithColumns(mat[1]).T
    fun learn(examples: List<DataPoint>, trainRate: Double = 0.1, maxEpochs: Int = 5) {
        for (epoch in 0..maxEpochs) {
            val shuffled = examples.shuffle()
            var totalError = 0.0

            for ((inputs, desiredOutput) in shuffled) {
                val output = response(inputs)
                val error = desiredOutput - output
                totalError += abs(error).sum()
                weights += trainRate * error * inputs.extendWithColumns(mat[1])
            }

            println("Epoch #$epoch err=$totalError")

            if (totalError == 0.0) break
        }
    }

    fun adalineLearn(examples: List<DataPoint>, trainRate: Double = 0.1, maxEpochs: Int = 5, eps: Double = 0.001) {
        for (epoch in 0..maxEpochs) {
            val shuffled = examples.shuffle()
            var totalError = 0.0

            for ((inputs, desiredOutput, desiredNet) in shuffled) {
                desiredNet!!
                val output = net(inputs)
                val error = desiredNet - output
                totalError += (error emul error).sum()
                weights += trainRate * error * inputs.extendWithColumns(mat[1])
            }

            println("Epoch #$epoch err=$totalError (adaline)")

            if (totalError in -eps..eps) break
        }
    }
}

data class DataPoint(val input: Matrix<Double>, val output: Matrix<Double>, var desiredNet: Matrix<Double>? = null) {
    constructor(combined: Matrix<Double>) : this(
            combined[
                    0..(combined.numRows() - 1),
                    0..(combined.numCols() - 3)
                    ],
            combined[
                    0..(combined.numRows() - 1),
                    combined.numCols() - 2
                    ],
            combined[
                    0..(combined.numRows() - 1),
                    combined.numCols() - 1
                    ]
    )
}