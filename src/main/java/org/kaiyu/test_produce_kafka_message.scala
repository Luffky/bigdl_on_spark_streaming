package org.kaiyu

import java.util

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerConfig, ProducerRecord}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.kaiyu.Constant.{Kafka_property_test}
import org.codehaus.jettison.json.JSONObject
import org.apache.spark.sql.Row

object test_produce_kafka_message {
  def main(args: Array[String]): Unit = {
    val topic = Kafka_property_test.TOPICS
    val brokers = Kafka_property_test.BOOTSTRAP_SERVERS
    val props = new util.HashMap[String, Object]
    props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, brokers)
    props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer")
    props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer")

    val producer = new KafkaProducer[String, String](props)
    val dataSet = getDataFromCsv("./HOME20(4).csv").map(row => row.toSeq.map(a=>a.asInstanceOf[String].toFloat).toArray[Float])
    dataSet.foreach(
      row => {
        val events = new JSONObject()
        val event = createKafkaMessage("HOME", "20", row(0), row(1), row(2), row(3))
        events.put(event.getString("name"), event)
        val producer = new KafkaProducer[String, String](props)
        val message = new ProducerRecord[String, String](topic, null, events.toString())
        producer.send(message)
        println(message)
        Thread.sleep(1000)
      }
    )


  }


  def createKafkaMessage(module: String, key: String, quantile95: Double, quantile05: Double, median: Double, percent_of_absence: Double): JSONObject = {
    val event = new JSONObject()
    val common_key = module + "." + key
    event.put("name", common_key)
    event.put("quantile95", quantile95)
    event.put("quantile05", quantile05)
    event.put("median", median)
    event.put("percent", percent_of_absence)
    event
  }

  def getDataFromCsv(dataset_dir: String): Array[Row] = {
    val conf = new SparkConf().setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()


    val schemaString = "quantile95,quantile05,median,count,percent"
    val fields = schemaString.split(",").map(fieldName=>StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)
    val dataset_rdd = spark.read.format("csv").option("mode", "FAILFAST").option("inferSchema", "false")
      .option("header", "true").load(dataset_dir).rdd
    val dataset = spark.createDataFrame(dataset_rdd, schema)

    dataset.createOrReplaceTempView("dataset")
    spark.sql("select quantile95, quantile05, median, percent from dataset").collect()

  }
}
