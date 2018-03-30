package org.kaiyu

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import Constant.Model_property
import com.intel.analytics.bigdl.dataset.{DataSet, TensorSample, _}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.StructField
import org.apache.spark.SparkFiles
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag


case class Config(input_dimension: Int, time_step:  Int, train_percent: Double,
                  output_dimension: Int, batch_size: Int,
                  epoch: Int, lstm_hidden_dimension: Int, load_model: Boolean, save_model: Boolean, save_dir: String, load_dir: String,
                  dataset_dir: String, dataset_mode: Int, partition_num:Int)

object bigDL {
  LoggerFilter.redirectSparkInfoLogs()

  def buildModel(): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(Recurrent[Float]().add(LSTM(Model_property.INPUT_DIMENSION, Model_property.LSTM_HIDDEN_DIMENSION)))
      .add(Select(2,-1)).add(Linear(Model_property.LSTM_HIDDEN_DIMENSION, Model_property.OUTPUT_DIMENSION))

    model
  }

  //  def buildModelWithGraph(): Graph[Float] = {
  //
  //  }
  def main(args: Array[String]): Unit = {
    val parser = new scopt.OptionParser[Config]("lstm_feature_predict") {
      head("scopt", "3.7")

      opt[Int]("input_dimension").action((x, c) => c.copy(input_dimension = x)).text("dimension of each step")
      opt[Int]("time_step").action((x, c) => c.copy(time_step = x)).text("time_step")
      opt[Double]('t', "train_percent").action((x, c) => c.copy(train_percent = x)).text("training/total")
      opt[Int]("output_dimension").action((x, c) => c.copy(output_dimension = x)).text("dimension of result")
      opt[Int]('e', "epoch").action((x, c) => c.copy(epoch = x)).text("total trainning epoch")
      opt[Int]('b', "batch_size").action((x, c) => c.copy(batch_size = x)).text("mini_batch_size of each iteration")
      opt[Int]("lstm_hidden_dimension").action((x, c) => c.copy(lstm_hidden_dimension = x)).text("dimension of lstm layers")

      opt[Unit]("load_model").action((_, c) => c.copy(load_model = true)).text("whether to load existed model from file")
      opt[Unit]("save_model").action((_, c) => c.copy(save_model = true)).text("whether to save model to file")
      opt[String]("load_dir").action((x, c) => c.copy(load_dir = x)).text("the dir from which load model")
      opt[String]("save_dir").action((x, c) => c.copy(save_dir = x)).text("the dir to which save model")
      opt[String]("dataset_dir").action((x, c) => c.copy(dataset_dir = x)).text("dataset_dir")
      opt[Int]("dataset_mode").action((x, c) => c.copy(dataset_mode = x)).text("data_set_mode")
      opt[Int]("partition_num").action((x, c) => c.copy(partition_num = x)).text("partition_num")


      help("help").text("prints this usage text")

    }

    parser.parse(args, Config(input_dimension = Model_property.INPUT_DIMENSION, time_step = Model_property.TIME_STEP,
      train_percent = Model_property.TRAIN_PERCENT, output_dimension = Model_property.OUTPUT_DIMENSION,
      batch_size = Model_property.BATCH_SIZE, epoch = Model_property.EPOCH, lstm_hidden_dimension = Model_property.LSTM_HIDDEN_DIMENSION,
      load_model = false, save_model = false, load_dir = "/tmp", save_dir = "/tmp", dataset_dir = "/tmp/HOME20(4).csv", dataset_mode = 0, partition_num = 1)) match{
      case Some(config) => {        var model_property: Model_property = null

        model_property = Model_property(config.input_dimension, config.time_step, config.train_percent, config.output_dimension,
          config.batch_size, config.epoch, config.lstm_hidden_dimension)


        val conf = Engine.createSparkConf().setAppName("lstm").set("spark.task.maxFailures", "1")
        val spark = SparkSession.builder().config(conf).getOrCreate()
        //    val ssc = new StreamingContext(spark.sparkContext,  Seconds(30))
        Engine.init




        val schemaString = "quantile95,quantile05,median,count,percent"
        val fields = schemaString.split(",").map(fieldName=>StructField(fieldName, StringType, nullable = true))
        val schema = StructType(fields)
        val dataset_rdd = spark.read.format("csv").option("mode", "FAILFAST").option("inferSchema", "false")
          .option("header", "true").load(config.dataset_dir).rdd
        val dataset = spark.createDataFrame(dataset_rdd, schema)

        dataset.createOrReplaceTempView("dataset")
        val temp = spark.sql("select quantile95, quantile05, median, percent from dataset").collect()
        val X = temp.sliding(model_property.time_step).toArray.map(record => record.map(row => row.toSeq.map(a=>a.asInstanceOf[String].toFloat).toArray[Float]))
        val Y = temp.drop(50).sliding(1).toArray.map(record=>record(0).toSeq.map(a=>a.asInstanceOf[String].toFloat).toArray[Float])
        val X_Y = X.zip(Y)

        val numberOfSamples = X.length
        val AllDataSet = X_Y
        //    val trainX = X.slice(0, (Model_property.TRAIN_PERCENT * numberOfSamples).toInt)
        //    val trainY = Y.slice(0, (Model_property.TRAIN_PERCENT * numberOfSamples).toInt)

        //    val testX = X.drop((Model_property.TRAIN_PERCENT * numberOfSamples).toInt)
        //    val testY = Y.drop((Model_property.TRAIN_PERCENT * numberOfSamples).toInt)

        val bigDL_trainX_RDD_with_map = if(config.dataset_mode == 0) DataSet.array(AllDataSet, spark.sparkContext).data(false)
          .map{
            case (input: Array[Array[Float]], label: Array[Float]) => {
              Sample(featureTensor = Tensor(input.flatten, Array(model_property.time_step, model_property.input_dimension)),
                labelTensor = Tensor(label, Array(model_property.output_dimension))
              )
            }
          }
        else if(config.dataset_mode == 1) DataSet.array(AllDataSet, spark.sparkContext).transform(TimeseriesToSample[Float](model_property)).toDistributed().data(false)
        else if(config.dataset_mode == 2) DataSet.array(AllDataSet, spark.sparkContext)

        println(config.toString)

        val trainedModel = if(config.dataset_mode == 1 || config.dataset_mode == 0){
          val Array(trainningRDD, valRDD) = bigDL_trainX_RDD_with_map.asInstanceOf[RDD[Sample[Float]]].randomSplit(Array(model_property.train_percent, 1-model_property.train_percent))
          valRDD.persist()

          var exist_model: Module[Float] = null
          if(config.load_model == true){
            exist_model = Module.loadModule(config.load_dir + "/model.bigdl", config.load_dir + "model.bin")
          }
          else {
            exist_model = buildModel()
          }

          val optimizer = Optimizer(
            model = exist_model,
            sampleRDD = trainningRDD,
            criterion = new MSECriterion[Float](),
            batchSize = model_property.batch_size
          )

          optimizer.setOptimMethod(new SGD[Float](learningRate = 1e-3, learningRateDecay = 0.0))
            .setValidation(Trigger.everyEpoch, valRDD, Array(new Loss(new MSECriterion[Float]())), model_property.batch_size)
            .setEndWhen(Trigger.maxEpoch(model_property.epoch))
            .optimize()
        }

        else {
          val Array(trainningRDD, valRDD) = bigDL_trainX_RDD_with_map.asInstanceOf[DataSet[(Array[Array[Float]], Array[Float])]].toDistributed()
            .data(false).randomSplit(Array(model_property.train_percent, 1-model_property.train_percent))
          val trainningDataSet = DataSet.rdd(trainningRDD).transform(TimeseriesToMiniBatch[Float](batch_size = model_property.batch_size,
            partitionNum = config.partition_num, mp = model_property))
          val valDataSet = DataSet.rdd(valRDD).transform(TimeseriesToMiniBatch[Float](batch_size = model_property.batch_size,
            partitionNum = config.partition_num, mp = model_property))

          var exist_model: Module[Float] = null
          if(config.load_model == true){
            exist_model = Module.loadModule(config.load_dir + "/model.bigdl", config.load_dir + "model.bin")
          }
          else {
            exist_model = buildModel()
          }

          val optimizer = Optimizer(
            model = exist_model,
            dataset = trainningDataSet,
            criterion = new MSECriterion[Float]()
          )
          optimizer.setOptimMethod(new SGD[Float](learningRate = 1e-3, learningRateDecay = 0.0))
            .setValidation(trigger = Trigger.everyEpoch, dataset = valDataSet, vMethods = Array(new Loss(new MSECriterion[Float]())))
            .setEndWhen(Trigger.maxEpoch(model_property.epoch))
            .optimize()
        }




        if(config.save_model == true) {
          trainedModel.saveModule(config.save_dir + "/model.bigdl", config.save_dir + "/model.bin", true)
        }





        spark.stop()


      }

      case None => {
        return
      }
    }


  }
}
class TimeseriesToMiniBatch[T: ClassTag](batch_size: Int, featurePaddingParam: Option[PaddingParam[T]] = None,
                                         labelPaddingParam: Option[PaddingParam[T]] = None, partitionNum: Int, numFeature: Int = 1
                                         , numLabel: Int = 1, md: Model_property)(implicit ev: TensorNumeric[T]) extends Transformer[(Array[Array[T]], Array[T]), MiniBatch[T]]{
  private val batchPerPartition = batch_size / partitionNum
  var miniBatchBuffer: MiniBatch[T] = null
  private val batchSize = batchPerPartition
  private val sampleData = new Array[Sample[T]](batchSize)

  override def apply(prev: Iterator[(Array[Array[T]], Array[T])]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      override def hasNext: Boolean = prev.hasNext
      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext){
            val (input, label) = prev.next()
            sampleData(i) = Sample[T](featureTensor = Tensor[T](input.flatten, Array(md.time_step, md.input_dimension)),
              labelTensor = Tensor[T](label, Array(md.output_dimension)))
            i += 1
          }
          val firstTimeSeriesData = sampleData(0)
          if (miniBatchBuffer == null){
            val firstSample = sampleData(0)
            miniBatchBuffer = MiniBatch(firstSample.numFeature(), firstSample.numLabel(), featurePaddingParam, labelPaddingParam)
          }
          if(i < batchSize){
            miniBatchBuffer.set(sampleData.slice(0, i))
          }
          else {
            miniBatchBuffer.set(sampleData)
          }
        }
        else {
          null
        }
      }
    }
  }
}

object TimeseriesToMiniBatch {
  def apply[T: ClassTag](batch_size: Int, featurePaddingParam: Option[PaddingParam[T]] = None,
                         labelPaddingParam: Option[PaddingParam[T]] = None, partitionNum: Int, numFeature: Int = 1
                         , numLabel: Int = 1, mp: Model_property)(implicit ev: TensorNumeric[T]): TimeseriesToMiniBatch[T] = new TimeseriesToMiniBatch[T](batch_size, featurePaddingParam, labelPaddingParam, partitionNum, numFeature, numLabel, mp)
}



class TimeseriesToSample[T: ClassTag](mp: Model_property)(implicit ev: TensorNumeric[T]) extends Transformer[(Array[Array[T]], Array[T]), Sample[T]]{
  override def apply(prev: Iterator[(Array[Array[T]], Array[T])]): Iterator[Sample[T]] = {
    new Iterator[Sample[T]] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): Sample[T] = {
        if (prev.hasNext){
          val temp = prev.next()
          Sample(featureTensor = Tensor(temp._1.flatten[T], Array(mp.time_step, mp.input_dimension)),
            labelTensor = Tensor(temp._2, Array(mp.output_dimension)))
        }
        else {
          null
        }
      }
    }
  }
}

object TimeseriesToSample {
  def apply[T: ClassTag](mp: Model_property)(implicit ev: TensorNumeric[T]) = {
    new TimeseriesToSample[T](mp)
  }
}
