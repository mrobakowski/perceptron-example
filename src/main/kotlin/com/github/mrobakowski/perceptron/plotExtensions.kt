package com.github.mrobakowski.perceptron

import golem.figures
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle.Scatter
import org.knowm.xchart.style.markers.SeriesMarkers.CIRCLE

private var figNum = 0

fun figureEx(num: Int) {
    golem.figure(num)
    figNum = num
}

fun scatter(vararg series: String) {
    figures[figNum]?.first?.apply {
        series.forEach {
            seriesMap[it]?.apply {
                xySeriesRenderStyle = Scatter
                marker = CIRCLE
            }
        }
    }
    figures[figNum]?.second?.repaint()
}

fun remSeries(vararg name: String) {
    figures[figNum]?.first?.apply {
        name.forEach {
            if (seriesMap.containsKey(it))
                removeSeries(it)
        }
    }
}


