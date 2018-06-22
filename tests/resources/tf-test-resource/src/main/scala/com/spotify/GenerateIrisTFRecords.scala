package com.spotify

import com.spotify.featran._
import com.spotify.featran.scio._
import com.spotify.featran.tensorflow._
import com.spotify.featran.transformers._
import com.spotify.scio._
import com.spotify.scio.bigquery._
import com.spotify.scio.tensorflow._
import org.tensorflow.example.{Example => TFExample}

object IrisFeaturesSpec {

  @BigQueryType.fromTable("data-integration-test:zoltar.iris")
  class Record

  case class Iris(sepalLength: Option[Double],
                  sepalWidth: Option[Double],
                  petalLength: Option[Double],
                  petalWidth: Option[Double],
                  className: Option[String])

  object Iris {
    def apply(record: Record): Iris =
      Iris(record.petal_length,
        record.petal_width,
        record.sepal_length,
        record.sepal_width,
        record.class_name)
  }

  val irisFeaturesSpec: FeatureSpec[Iris] = FeatureSpec
    .of[Iris]
    .optional(_.petalLength)(StandardScaler("petal_length", withMean = true))
    .optional(_.petalWidth)(StandardScaler("petal_width", withMean = true))
    .optional(_.sepalLength)(StandardScaler("sepal_length", withMean = true))
    .optional(_.sepalWidth)(StandardScaler("sepal_width", withMean = true))

  val irisLabelSpec: FeatureSpec[Iris] = FeatureSpec
    .of[Iris]
    .optional(_.className)(OneHotEncoder("class_name"))

  val irisSpec = MultiFeatureSpec(irisFeaturesSpec, irisLabelSpec)
}

object IrisFeaturesJob {
  def main(cmdLineArgs: Array[String]): Unit = {
    import IrisFeaturesSpec._

    val (sc, args) = ContextAndArgs(cmdLineArgs)

    val data = sc.typedBigQuery[Record]().map(Iris(_))
    val extractedFeatures = irisSpec.extract(data)

    val (train, test) = extractedFeatures
      .featureValues[TFExample]
      .randomSplit(.9)

    extractedFeatures.featureSettings.saveAsTextFile(
      args("output") + "/settings")
    train.saveAsTfExampleFile(args("output") + "/train", extractedFeatures)
    test.saveAsTfExampleFile(args("output") + "/eval", extractedFeatures)

    sc.close()
  }
}