package com.mlplat.da.miner

import org.ansj.splitWord.analysis.NlpAnalysis
import org.ansj.domain.{Result, Term}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._

import collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

/**
  * 中文分词器
  */
object WordSpliter {


  def main(args: Array[String]) = {

    val spark = SparkSession
      .builder()
      .master("yarn")
      .appName("WordSpliter")
      .enableHiveSupport()
      .getOrCreate()

//    val text = "加Q群492447  来就送首冲  萌妹子陪玩，常在送VIP3。 长城ol玩家交流群：467209045准备******公测，同步大电影推出。"
//
//    val t = NlpAnalysis.parse(text)
//    println(t)
//
//    println(parseContent(text))


    val chatData = spark.sql(
      """
        | select content
        | from gw_chat.chat_data_22_su
        | where other!='1'
      """.stripMargin)

    import spark.implicits._

    val wordData = chatData.flatMap {
      case Row(c: String) => parseContent(c)
    }.filter(w => {
      !w.name.trim.isEmpty && w.nature != "null"
    }).map(x => (x.name, x.nature))
      .toDF("name", "nature")

    val countData = wordData.groupBy("name", "nature").count()
    countData.write.mode(SaveMode.Overwrite).saveAsTable("gw_chat.word_count")
  }

  case class Word(name: String, nature: String)

  def parseContent(text: String) = {
    val t = NlpAnalysis.parse(text)
    val ts = t.getTerms.toList
    for (t <- ts) yield Word(t.getName, t.getNatureStr)
  }

}
