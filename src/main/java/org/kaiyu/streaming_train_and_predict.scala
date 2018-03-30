package org.kaiyu

import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.streaming.{Seconds, StreamingContext}
import Constant._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import kafka.serializer.StringDecoder
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.streaming.kafka.KafkaUtils
import org.codehaus.jettison.json.JSONObject

import scala.collection.immutable.{HashMap, HashSet}
import scala.collection.mutable.ArrayBuffer

object streaming_train_and_predict {
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName("lstm").set("spark.task.maxFailures", "1")
    val ssc = new StreamingContext(conf, Seconds(WINDOW_SLIDING_SECONDS))
    Engine.init

    // test
    val topics = Kafka_property_test.TOPICS
    // product
//    val topics = Kafka_property.TOPICS

    var topicSet = new HashSet[String]
    ssc.checkpoint("checkpoint")
    topicSet += topics

    var kafkaParam = new HashMap[String, String]
    // test
    kafkaParam += ("zookeeper.connect" -> Kafka_property_test.ZOOKEEPER_CONNECT
      , "bootstrap.servers" -> Kafka_property_test.BOOTSTRAP_SERVERS)
    // product
//    kafkaParam += ("zookeeper.connect" -> Kafka_property.ZOOKEEPER_CONNECT
//      , "bootstrap.servers" -> Kafka_property.BOOTSTRAP_SERVERS)

    val kafkaStream = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](ssc, kafkaParam, topicSet)
    val inputStream = kafkaStream.flatMap(flatMap_kernel).map(map_kernel)
//    inputStream.reduceByKeyAndWindow(reduceFunc = reduceFunc
//      , invReduceFunc = (a,b) => invReduceFunc(a, b), slideDuration = Seconds(WINDOW_SLIDING_SECONDS), windowDuration = Seconds(WINDOW_DURATION_SECONDS)
//      , filterFunc = (key_value) => filterFunc(key_value._1, key_value._2)).foreachRDD(
//      rdd => {
//        rdd.foreach(
//          record => {
//            println(record._1, record._2.length)
//            record._2.foreach(
//              data => {
//                data.foreach(a => print(a.toString + " "))
//                println()
//              }
//            )
//          }
//        )
//      }
//    )
    val predictStream = inputStream.reduceByKeyAndWindow(reduceFunc = reduceFunc
      , invReduceFunc = (a,b) => invReduceFunc(a, b), slideDuration = Seconds(WINDOW_SLIDING_SECONDS), windowDuration = Seconds(WINDOW_DURATION_SECONDS)
      , filterFunc = (key_value) => filterFunc(key_value._1, key_value._2)).map {
      case (key: String, value: ArrayBuffer[Array[Float]]) => {
        val temp = value.flatten.toArray
        if (temp.size == Model_property.INPUT_DIMENSION * (Model_property.TIME_STEP + 1)) {
          val sample = Sample(featureTensor = Tensor(temp.dropRight(Model_property.INPUT_DIMENSION), Array(Model_property.TIME_STEP, Model_property.INPUT_DIMENSION)),
            labelTensor = Tensor(temp.take(Model_property.INPUT_DIMENSION), Array(Model_property.OUTPUT_DIMENSION)))
          Some((key, sample))
        }
        else {
          None
        }
      }
    }.filter(_.isDefined).map(_.get).transform(
      rdd=>{
        bigdl_predict.optimize_online(rdd)
      }
    ).transform(
      rdd=>{
        bigdl_predict.predict(rdd)
      }
    )

    inputStream.join(predictStream).foreachRDD(
      rdd => {
        rdd.foreachPartition(
          records => {
            if (records.hasNext) {
              records.foreach(
                record => {
                  val temp = Tensor(record._2._1.flatten.toArray, Array(Model_property.OUTPUT_DIMENSION))
                  print("key: " + record._1 + " ")
                  println(record._2._2.toTensor[Float] - temp)
                }
              )
            }
          }
        )
      }
    )

    ssc.start()
    ssc.awaitTermination()







  }

  def flatMap_kernel(input: (String, String)): ArrayBuffer[String] = {
    val emits = new ArrayBuffer[String]()
    val json = new JSONObject(input._2)
    val iter = json.keys()

    while (iter.hasNext){
      emits += json.getString(iter.next().toString)
    }
    emits
  }

  def map_kernel(input: String): (String, ArrayBuffer[Array[Float]]) = {
    val json = new JSONObject(input)
    val value = transform(json)
    (json.getString("name"), value)
  }

  def transform(json: JSONObject): ArrayBuffer[Array[Float]] = {
    val quantile95 = json.getDouble("quantile95").toFloat
    val quantile05 = json.getDouble("quantile05").toFloat
    val median = json.getDouble("median").toFloat
    val percent = json.getDouble("percent").toFloat
    return ArrayBuffer(Array(quantile95, quantile05, median, percent))
  }

  def reduceFunc(old_array: ArrayBuffer[Array[Float]] , new_one: ArrayBuffer[Array[Float]]): ArrayBuffer[Array[Float]] = {
    old_array ++= new_one
    if (old_array.size > Model_property.TIME_STEP + 1) { // 留存tiem_step + 1个数据，以便训练和预测，使用0-time_step个数据作为feature，第time_step + 1数据作为label，使用1-time_step + 1个数据作为feature，用来预测
      val temp = old_array.drop(old_array.size - Model_property.TIME_STEP - 1)
      temp
    }
    else {
      old_array
    }
  }

  def invReduceFunc(old_array: ArrayBuffer[Array[Float]], depreicated_one: ArrayBuffer[Array[Float]]): ArrayBuffer[Array[Float]] = {
    if (old_array.size > Model_property.TIME_STEP + 1) { // 留存tiem_step + 1个数据，以便训练和预测，使用0-time_step个数据作为feature，第time_step + 1数据作为label，使用1-time_step + 1个数据作为feature，用来预测
      old_array.drop(old_array.size - Model_property.TIME_STEP - 1)
    }
    else {
      old_array
    }
  }

  def filterFunc(key: String, value: ArrayBuffer[Array[Float]]): Boolean = {
    if(TRAINED_KEY.contains(key)) {
      return true
    }
    else {
      return false
    }
  }






}
