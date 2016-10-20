package com.github.mrobakowski.perceptron.ui

import com.github.mrobakowski.perceptron.DataPoint
import com.github.mrobakowski.perceptron.Neuron
import com.github.mrobakowski.perceptron.draw
import com.github.mrobakowski.perceptron.generateDataPoints
import golem.mat
import javafx.beans.property.SimpleBooleanProperty
import javafx.beans.property.SimpleDoubleProperty
import javafx.beans.property.SimpleIntegerProperty
import javafx.beans.property.SimpleObjectProperty
import javafx.collections.FXCollections
import javafx.scene.control.Label
import javafx.scene.control.cell.TextFieldTableCell
import javafx.util.StringConverter
import tornadofx.*
import kotlin.reflect.KMutableProperty1

class PerceptronView : View() {
    init {
        title = "Perceptron Example"
    }

    val data = FXCollections.observableArrayList<ObservableDataPoint2D>()!!

    val numDataProp = SimpleIntegerProperty(100)
    var numData by numDataProp

    val lParam1Prop = SimpleDoubleProperty(-1.0)
    var lParam1 by lParam1Prop

    val lParam2Prop = SimpleDoubleProperty(1.0)
    var lParam2 by lParam2Prop

    val lParam3Prop = SimpleDoubleProperty(0.0)
    var lParam3 by lParam3Prop

    val perceptronProp = SimpleObjectProperty<Neuron>(Neuron(2))
    var perceptron by perceptronProp

    val trainRateProp = SimpleDoubleProperty(0.1)
    var trainRate by trainRateProp

    val maxEpochsProp = SimpleIntegerProperty(50)
    var maxEpochs by maxEpochsProp

    val epsProp = SimpleDoubleProperty(0.01)
    var eps by epsProp

    val dispProp = SimpleBooleanProperty(false)
    val disp by dispProp

    init {
        dispProp.onChange { draw(data.map { it.dp }, perceptron, it!!) }
    }

    override val root = vbox {

        form {
            fieldset("Example data") {
                field("Number") {
                    textfield().bind(numDataProp)
                }
                field("Line param 1") {
                    textfield().bind(lParam1Prop)
                }

                field("Line param 2") {
                    textfield().bind(lParam2Prop)
                }

                field("Line param 3") {
                    textfield().bind(lParam3Prop)
                }

                button("Generate") {
                    setOnMouseClicked {
                        data.clear()
                        data += generateDataPoints(numData, lParam1, lParam2, lParam3, perceptron, disp).map(::ObservableDataPoint2D)
                    }
                }

                field {
                    tableview(data) {
                        columnResizePolicy = SmartResize.POLICY
                        isEditable = true

                        fun dataPointColumn(name: String, prop: KMutableProperty1<ObservableDataPoint2D, Double>) {
                            val col = column(name, prop)
                            col.cellFactory = TextFieldTableCell.forTableColumn(object : StringConverter<Double>() {
                                override fun toString(p: Double?) = p.toString()
                                override fun fromString(string: String?) = string?.toDouble()
                            })
                            col.setOnEditCommit {
                                prop.set(it.tableView.items[it.tablePosition.row], it.newValue)

                                draw(data.map { it.dp }, perceptron, disp)
                            }
                        }

                        dataPointColumn("x", ObservableDataPoint2D::x)
                        dataPointColumn("y", ObservableDataPoint2D::y)
                        dataPointColumn("unipolar", ObservableDataPoint2D::unipolar)
                        dataPointColumn("bipolar", ObservableDataPoint2D::bipolar)
                    }
                }
            }

            fieldset("Perceptron") {
                field("Display") {
                    checkbox().bind(dispProp)
                }

                field("Max # of epochs") {
                    textfield().bind(maxEpochsProp)
                }
                field("Train rate") {
                    textfield().bind(trainRateProp)
                }
                field("Epsilon (adaline)") {
                    textfield().bind(epsProp)
                }

                var l: Label? = null

                fun rebindWeightsLabel() {
                    l?.bind(perceptronProp, readonly = true, converter = object : StringConverter<Neuron>() {
                        override fun toString(p: Neuron) = "${p.weights[0]}, ${p.weights[1]}, ${p.weights[2]}"
                        override fun fromString(string: String?) = throw UnsupportedOperationException("not implemented")
                    })
                }

                field("Weights") {
                    l = label()
                    rebindWeightsLabel()

                    button("Randomize") {
                        setOnMouseClicked {
                            perceptron = Neuron(2)
                            draw(data.map { it.dp }, perceptron, disp)
                        }
                    }
                }
                field {
                    button("Train") {
                        setOnMouseClicked {
                            val dps = data.map { it.dp }
                            perceptron.perceptronLearn(dps, trainRate, maxEpochs)
                            rebindWeightsLabel()
                            draw(dps, perceptron, disp)
                        }
                    }
                    button("Train (ADALINE)") {
                        setOnMouseClicked {
                            val dps = data.map { it.dp }
                            perceptron.adalineLearn(dps, trainRate, maxEpochs, eps)
                            rebindWeightsLabel()
                            draw(dps, perceptron, disp)
                        }
                    }
                }
            }
        }
    }
}

class ObservableDataPoint2D(val dp: DataPoint) {
    val xProp = SimpleDoubleProperty()
    var x by xProp

    init {
        x = dp.input[0]
        xProp.onChange { if (it != null) dp.input[0] = it.toDouble() }
    }

    val yProp = SimpleDoubleProperty()
    var y by yProp

    init {
        y = dp.input[1]
        yProp.onChange { if (it != null) dp.input[1] = it.toDouble() }
    }

    val unipolarProp = SimpleDoubleProperty()
    var unipolar by unipolarProp

    init {
        unipolar = dp.unipolarOut[0]
        unipolarProp.onChange { if (it != null) dp.unipolarOut[0] = it.toDouble() }
    }

    val bipolarProp = SimpleDoubleProperty()
    var bipolar by bipolarProp

    init {
        bipolar = dp.bipolarOut[0]
        unipolarProp.onChange { if (it != null) dp.bipolarOut = mat[it.toDouble()] }
    }
}

