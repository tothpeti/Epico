package simulations

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import simulations.BaseSimulation

case class WithoutColumnExcluding(
                                   pathToDatasets: String,
                                   pathToMetrics: String,
                                   pathToMetricsColExcluding: String,
                                   pathToPredictions: String,
                                   pathToPredictionsColExcluding: String
                                 ) extends BaseSimulation
{

  override def loadData[T](sparkSession: SparkSession): DataFrame = {
    val df = sparkSession.read.csv(pathToDatasets).toDF()
    df
  }

  override protected def runModel[A](): DataFrame = ???

  override protected def testModel[A](): DataFrame = ???

  override def createMetrics[A](): DataFrame = ???

  override def saveMetrics(): Unit = ???

  override def run[T](): Unit = ???

  private def getFileNames(df: DataFrame): RDD[String] =
    df.select(col("file_names")).rdd.distinct().map{
    }
}
