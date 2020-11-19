package simulations

import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

trait BaseSimulation {
  def loadData[T](sparkSession: SparkSession): DataFrame
  def run[T](): Unit
  def runModel[T](): DataFrame
  def testModel[A](): DataFrame
  def createMetrics[T](): DataFrame
  def saveMetrics(): Unit
}
