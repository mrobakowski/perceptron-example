package com.github.mrobakowski.perceptron

import golem.mat
import golem.matrix.Matrix
import golem.plot
import golem.title
import java.util.*

fun generateDataPoints(num: Int = 1000, lParam1: Double, lParam2: Double, lParam3: Double, perceptron: Perceptron,
                       displayPerceptron: Boolean): List<DataPoint> {
    val rand = Random()
    val xs = DoubleArray(num)
    val ys = DoubleArray(num)

    val res = (1..num).map {
        val x = rand.nextDouble()
        xs[it - 1] = x
        val y = rand.nextDouble()
        ys[it - 1] = y

        val res = if (lParam1 * x + lParam2 * y + lParam3 > 0) 1.0 else 0.0

        DataPoint(mat[x, y, res, lParam1 * x + lParam2 * y + lParam3])
    }

    draw(res, perceptron, displayPerceptron)

    return res
}

fun draw(dps: List<DataPoint>, perceptron: Perceptron, displayPerceptron: Boolean) {
    val pts = dps.groupBy { it.output[0] }

    val (xsAbove, ysAbove) = pts[0.0]?.map { it.input[0] to it.input[1] }?.unzip() ?: listOf<Double>() to listOf()
    val (xsBelow, ysBelow) = pts[1.0]?.map { it.input[0] to it.input[1] }?.unzip() ?: listOf<Double>() to listOf()

    val a = perceptron.weights
    val xStart = 0.0
    val yStart = line(a, xStart)
    val xEnd = 1.0
    val yEnd = line(a, xEnd)

    draw(xsAbove, xsBelow, ysAbove, ysBelow, xStart, yStart, xEnd, yEnd, displayPerceptron)
}

private fun line(a: Matrix<Double>, xStart: Double) = (a[0] * xStart + a[2]) / -a[1]

private fun draw(xsAbove: List<Double>, xsBelow: List<Double>, ysAbove: List<Double>, ysBelow: List<Double>,
                 xStart: Double, yStart: Double, xEnd: Double, yEnd: Double, displayPerceptron: Boolean) {
    figureEx(0)
    remSeries("0s", "1s", "perceptron")
    if (xsAbove.isNotEmpty() && ysAbove.isNotEmpty())
        plot(xsAbove.toDoubleArray(), ysAbove.toDoubleArray(), "o", lineLabel = "0s")
    if (xsBelow.isNotEmpty() && ysBelow.isNotEmpty())
        plot(xsBelow.toDoubleArray(), ysBelow.toDoubleArray(), "g", lineLabel = "1s")
    title("generated data")
    scatter("0s", "1s")
    if (displayPerceptron)
        plot(doubleArrayOf(xStart, xEnd), doubleArrayOf(yStart, yEnd), "r", lineLabel = "perceptron")
}
