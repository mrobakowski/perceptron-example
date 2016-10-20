package com.github.mrobakowski.perceptron

import golem.*
import golem.matrix.Matrix

class Neuron(
        numInputs: Int
) {
    fun activationFunctionUnipolar(it: Matrix<Double>) = it.mapMat { (it >= 0).toDouble() }
    fun activationFunctionBipolar(it: Matrix<Double>) = it.mapMat { if (it >= 0) 1.0 else -1.0 }

    var weights = randn(numInputs + 1)

    fun response(x: Matrix<Double>, bipolar: Boolean = false) = if (!bipolar)
        activationFunctionUnipolar(net(x))
    else
        activationFunctionBipolar(x)

    fun net(x: Matrix<Double>) = weights * x.extendWithColumns(mat[1]).T
    fun perceptronLearn(examples: List<DataPoint>, trainRate: Double = 0.1, maxEpochs: Int = 5): Pair<Double, Int> {
        var lastError = 0.0
        var numIter = 0
        for (epoch in 1..maxEpochs) {
            numIter += 1
            val shuffled = examples.shuffle()
            var totalError = 0.0

            for ((inputs, desiredOutput) in shuffled) {
                val output = response(inputs)
                val error = desiredOutput - output
                totalError += abs(error).sum()
                weights += trainRate * error * inputs.extendWithColumns(mat[1])
            }

//            println("Epoch #$epoch err=$totalError")

            lastError = totalError

            if (totalError == 0.0) break
        }
        return lastError to numIter
    }

    fun adalineLearn(examples: List<DataPoint>, trainRate: Double = 0.1, maxEpochs: Int = 5, eps: Double = 0.001): Pair<Double, Int> {
        var lastError = 0.0
        var numIter = 0
        for (epoch in 1..maxEpochs) {
            numIter += 1
            val shuffled = examples.shuffle()
            var totalError = 0.0

            for ((inputs, unipolarOut, bipolarOut) in shuffled) {
                val output = net(inputs)
                val error = bipolarOut - output
                totalError += (error emul error).sum()
                weights += trainRate * error * inputs.extendWithColumns(mat[1])
            }

            totalError /= shuffled.size

//            println("Epoch #$epoch err=$totalError (adaline)")

            lastError = totalError
            if (totalError in -eps..eps) break
        }
        return lastError to numIter
    }
}

data class DataPoint(val input: Matrix<Double>, val unipolarOut: Matrix<Double>, var bipolarOut: Matrix<Double>) {
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