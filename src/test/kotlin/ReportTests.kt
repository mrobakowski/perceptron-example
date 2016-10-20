import com.github.mrobakowski.perceptron.Neuron
import com.github.mrobakowski.perceptron.generateDataPoints
import golem.tan
import org.junit.Ignore
import org.junit.Test

const val ITERS = 1
const val EPOCHS = 100
const val INCREMENTS = 100

class ReportTests {
    @Ignore @Test fun testIterationError() {
        val perceptronErrors = DoubleArray(EPOCHS)
        val adalineErrors = DoubleArray(EPOCHS)

        for (i in 0..ITERS - 1) {
            val (l1, l2, l3) = getLineParams(i)
            val data = generateDataPoints(100, l1, l2, l3, blind = true, displayPerceptron = false)

            val perc = Neuron(2)
            val adaline = Neuron(2)

            for (epoch in 0..EPOCHS - 1) {
                val percErr = perc.perceptronLearn(data, maxEpochs = 1, trainRate = 0.1).first
                val adaErr = adaline.adalineLearn(data, maxEpochs = 1, trainRate = 0.1).first

                perceptronErrors[epoch] = (perceptronErrors[epoch] * i + percErr) / (i + 1)
                adalineErrors[epoch] = (adalineErrors[epoch] * i + adaErr) / (i + 1)
            }
        }

        println("Perceptron Errors")
        perceptronErrors.forEachIndexed { i, d ->
            println("${i + 1}\t$d")
        }

        println("\nADALINE Errors")
        adalineErrors.forEachIndexed { i, d ->
            println("${i + 1}\t$d")
        }
    }

    @Ignore @Test fun testLearningRate() {
        val perceptronData = DoubleArray(INCREMENTS)
        val adalineData = DoubleArray(INCREMENTS)

        for (i in 0..ITERS - 1) {
            val (l1, l2, l3) = getLineParams(i)
            val data = generateDataPoints(100, l1, l2, l3, blind = true, displayPerceptron = false)

            val percStartWeights = Neuron(2).weights
            val adaStartWeights = Neuron(2).weights

            for (incNum in 1..INCREMENTS) {
                val trainRate = (1.0 / INCREMENTS) * incNum

                val percIters = Neuron(2).apply { weights = percStartWeights.copy() }.perceptronLearn(data, maxEpochs = 1000, trainRate = trainRate).second
                val adaIters = Neuron(2).apply { weights = adaStartWeights.copy() }.adalineLearn(data, maxEpochs = 1000, trainRate = trainRate, eps = 0.35).second

                perceptronData[incNum - 1] = (perceptronData[incNum - 1] * i + percIters) / (i + 1)
                adalineData[incNum - 1] = (adalineData[incNum - 1] * i + adaIters) / (i + 1)
            }
        }

        println("Perceptron iters")
        perceptronData.forEachIndexed { i, d ->
            println("${(1.0 / INCREMENTS) * (i+1)}\t$d")
        }

        println("\nADALINE iters")
        adalineData.forEachIndexed { i, d ->
            println("${(1.0 / INCREMENTS) * (i+1)}\t$d")
        }
    }
}

const val PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062
const val ANGLE_STEP = PI / ITERS

fun getLineParams(i: Int): Triple<Double, Double, Double> {
    val l1 = tan(ANGLE_STEP * i)
    val l2 = -1.0
    val l3 = 0.5 - (l1 * 0.5)
    return Triple(l1, l2, l3)
}