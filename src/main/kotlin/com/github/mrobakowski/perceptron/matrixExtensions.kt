package com.github.mrobakowski.perceptron

import golem.mapMatIndexed
import golem.mapRows
import golem.matrix.Matrix
import java.util.*


fun Boolean.toInt() = if (this) 1 else 0
fun Boolean.toDouble() = if (this) 1.0 else 0.0
val Matrix<Double>.doubleArray: Array<Double> get() = getDoubleData().toTypedArray()
fun Matrix<Double>.extendWithColumns(cols: Matrix<Double>): Matrix<Double> {
    cols.numRows() == this.numRows() || throw InvalidDimensionsException(this.numRows(), cols.numRows())

    var i = 0
    return mapRows {
        golem.mat.get(*it.doubleArray, *cols.getRow(i++).doubleArray)
    }
}

infix fun Matrix<Double>.eq(other: Matrix<Double>) =
        this.mapMatIndexed { i, j, v -> if (v == other[i, j]) 0.0 else 1.0 }.sum() == 0.0

class InvalidDimensionsException(expected: Int, got: Int) : Exception("Invalid dimensions - expected: $expected, got: $got")


fun <T> List<T>.shuffle(): List<T> {
    val items = this.toMutableList()
    val rg: Random = Random()
    for (i in 0..items.size - 1) {
        val randomPosition = rg.nextInt(items.size)
        val tmp: T = items[i]
        items[i] = items[randomPosition]
        items[randomPosition] = tmp
    }
    return items
}