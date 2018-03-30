package org.kaiyu

import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.{MSECriterion, Module}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{LocalOptimizer, Optimizer, SGD, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class bigdl_predict private (){

  var exist_optimizer: Optimizer[Float, MiniBatch[Float]] = null
  var exist_model: AbstractModule[Activity, Activity, Float] = null
  def load_optimizer(model_dir: String, weight_dir: String, batchSize: Int, sc: SparkContext): Optimizer[Float, MiniBatch[Float]] = {
    exist_model = Module.loadModule[Float](model_dir, weight_dir)
    val dataSet: RDD[Sample[Float]] = sc.emptyRDD[Sample[Float]]
    exist_optimizer = Optimizer(
      model = exist_model,
      sampleRDD = dataSet,
      batchSize = batchSize,
      criterion = new MSECriterion[Float]()
    ).setOptimMethod(new SGD[Float](learningRate = 1e-3, learningRateDecay = 0.0))
      .setEndWhen(Trigger.maxEpoch(1))

    exist_optimizer
  }

  def load_model(model_dir: String, weight_dir: String, batchSize: Int): AbstractModule[Activity, Activity, Float] = {
    exist_model = Module.loadModule[Float](model_dir, weight_dir)
    exist_model
  }
}

object bigdl_predict {
  val bigdl_predict = new bigdl_predict()
  val model_dir: String = "model.bigdl"
  val weight_dir: String = "model.bin"
  val batch_size: Int = 1
  var input_dimension: Int = 4
  var output_dimension: Int = 4
  var time_step: Int = 50

  def apply(input_dimension: Int, output_dimension: Int, time_step: Int): bigdl_predict = {
    this.input_dimension = input_dimension
    this.output_dimension = output_dimension
    this.time_step = time_step
    bigdl_predict
  }
  def GetModelInstance(sc: SparkContext): bigdl_predict = {
    if (bigdl_predict.exist_optimizer == null){
      println("loading model successfully")
      bigdl_predict.load_optimizer(model_dir, weight_dir, batch_size, sc)
    }
    else {
      println("return existed model")
    }
    bigdl_predict
  }

  def predict(dataset: RDD[(String, Sample[Float])], batchSize: Int = batch_size): RDD[(String, Activity)] = {
    if (bigdl_predict.exist_model == null) {
      println("loading model successfully")
      bigdl_predict.load_optimizer(model_dir, weight_dir, batchSize, dataset.sparkContext)
    }
    dataset.map(x => (x._1, bigdl_predict.exist_model.forward(Tensor[Float](x._2.getData().drop(4), Array(batchSize, time_step, input_dimension)))))
  }

  def predict_and_optimize_rdd(dataset: RDD[Sample[Float]], batchSize: Int = batch_size): RDD[Activity]  = {
    if (bigdl_predict.exist_optimizer == null) {
      println("loading model successfully")
      bigdl_predict.load_optimizer(model_dir, weight_dir, batchSize, dataset.sparkContext)
    }
    bigdl_predict.exist_model = bigdl_predict.exist_optimizer.setTrainData(dataset, batchSize).optimize()
    val record = DataSet.rdd(dataset).toLocal().data(false)
    if (record.hasNext) {
      val feature = Sample(Tensor(record.next().getData().drop(4), Array(time_step, input_dimension)))
      bigdl_predict.exist_model.predict(dataset.sparkContext.parallelize(Seq(feature)))

    }

    else {
      dataset.sparkContext.emptyRDD[Activity]
    }
  }

  def optimize_online(dataset: RDD[(String, Sample[Float])], batchSize: Int = batch_size): RDD[(String, Sample[Float])]  = {
    if (bigdl_predict.exist_optimizer == null) {
      println("loading model successfully")
      bigdl_predict.load_optimizer(model_dir, weight_dir, batchSize, dataset.sparkContext)
    }
    if(dataset.isEmpty){
      dataset.sparkContext.emptyRDD[(String, Sample[Float])]
    }
    else {
      dataset.collect()
      bigdl_predict.exist_model = bigdl_predict.exist_optimizer.setTrainData(dataset.map(_._2), batchSize).optimize()
      dataset
    }

  }


}
