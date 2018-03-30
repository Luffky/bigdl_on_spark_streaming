package org.kaiyu

import scala.collection.immutable.HashSet

object Constant {
  case class Model_property (input_dimension: Int, time_step: Int, train_percent: Double, output_dimension: Int, batch_size: Int, epoch:Int, lstm_hidden_dimension: Int)

  object Model_property{
    val INPUT_DIMENSION: Int = 4
    val TIME_STEP: Int = 50
    val TRAIN_PERCENT: Float = 0.8f
    val OUTPUT_DIMENSION: Int = 4
    val BATCH_SIZE: Int = 288
    val EPOCH: Int = 50
    val LSTM_HIDDEN_DIMENSION: Int = 50
    def apply(train_percent: Double, batch_size: Int, epoch: Int): Model_property = {
      new Model_property(INPUT_DIMENSION, TIME_STEP, train_percent, OUTPUT_DIMENSION, batch_size, epoch, LSTM_HIDDEN_DIMENSION)
    }
    def apply(): Model_property = {
      new Model_property(INPUT_DIMENSION, TIME_STEP, TRAIN_PERCENT, OUTPUT_DIMENSION, BATCH_SIZE, EPOCH, LSTM_HIDDEN_DIMENSION)
    }
  }

  val WINDOW_SLIDING_SECONDS: Int = 1
  val WINDOW_DURATION_SECONDS: Int = 1500

  object Kafka_property_test{
    val ZOOKEEPER_CONNECT: String = "localhost:2181"
    val TOPICS: String = "feature-realtimetraining"
    val BOOTSTRAP_SERVERS: String = "localhost:9092"
  }

  val TRAINED_KEY: HashSet[String] = HashSet[String]("HOME.20")

}
