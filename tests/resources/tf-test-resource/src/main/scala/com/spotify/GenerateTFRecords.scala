package com.spotify

import com.spotify.featran.scio._
import com.spotify.featran.tensorflow._
import com.spotify.featran.transformers.{Identity, MinMaxScaler, OneHotEncoder}
import com.spotify.featran.{FeatureSpec, MultiFeatureSpec}
import com.spotify.scio._
import com.spotify.scio.tensorflow._
import org.tensorflow.example.Example

import scala.util.Random

/*
sbt "runMain com.spotify.GenerateTFRecords --runner=DirectRunner --output=./tf-records"
*/

object GenerateTFRecords {
  def main(cmdlineArgs: Array[String]): Unit = {
    val (sc, args) = ContextAndArgs(cmdlineArgs)

    case class Point(f3: Double,
                     f1: Int,
                     f2: String,
                     label: Int)

    val cnt = 1000
    val rng = new Random(42)
    val points = 1.to(cnt).map { i =>
      val n = rng.nextGaussian()
      Point(n, i, if (i % 2 == 0) "ODD" else "EVEN", if (n >= 0) 1 else 0)
    }

    val features = FeatureSpec.of[Point]
      .required(_.f3)(MinMaxScaler("f3"))
      .required(_.f1.toDouble)(Identity("f1"))
      .required(_.f2)(OneHotEncoder("f2"))

    val label = FeatureSpec.of[Point]
      .required(_.label.toDouble)(Identity("label"))


    val scP = sc.parallelize(points)

    val fs = MultiFeatureSpec(features, label)
      .extract(scP)
    
    val (train, eval) = fs
      .featureValues[Example]
      .randomSplit(.9)

    train.saveAsTfExampleFile(args("output") + "/train", fs)
    eval.saveAsTfExampleFile(args("output") + "/eval", fs)

    fs.featureSettings
      .saveAsTextFile(args("output") + "/settings", numShards=1)

    sc.close().waitUntilFinish()
  }
}
