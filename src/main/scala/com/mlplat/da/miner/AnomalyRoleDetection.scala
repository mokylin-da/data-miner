package com.mlplat.da.miner

import org.apache.spark.ml.feature._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql._

/**
  * 异常角色检测
  */
object AnomalyRoleDetection {

  val spark = SparkSession
    .builder()
    .master("yarn")
    .appName("CF")
    .enableHiveSupport()
    .getOrCreate()


  def main(args: Array[String]) = {
    import spark.implicits._
    assert(args.length == 2, "Need 2 argument: game date")
    val game = args(0)
    val dt = args(1)
    val input = spark.sql(s"select * from ${game}_result.active_role where dt='$dt' and to_date(lastLoginTime)='$dt' ")
    input.cache()
    val assembler = new VectorAssembler().setInputCols(Array("rolelevel", "totalpayamount", "totalpaytimes", "totalshopamount", "totalshoptimes", "totalonlinetime")).setOutputCol("features")
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    val kmeans = new KMeans().setK(10).setFeaturesCol("scaledFeatures").setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))

    val kMeansPredictionModel = pipeline.fit(input)

    val predictionResult = kMeansPredictionModel.transform(input)

    val pre = predictionResult.groupBy("prediction").count().orderBy("count")
    pre.show()

    val preLabels = pre.collect().map {
      case Row(p: Int, _) => p
    }


    //    for (i <- preLabels) {
    //      predictionResult.filter(s"prediction=$i").show()
    //    }

    val firstPreLabel = preLabels(0)

    spark.sqlContext.setConf("hive.exec.dynamic.partition.mode", "nonstrict")

    predictionResult.filter(s"prediction=$firstPreLabel")
      .drop("features", "scaledFeatures", "prediction")
      .write.insertInto(s"${game}_result.abnormal_role")
  }

}
