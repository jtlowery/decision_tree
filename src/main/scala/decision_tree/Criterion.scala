package decision_tree

object Criterion {

  def log2(x: Double): Double = scala.math.log(x) / scala.math.log(2)

  def giniImpurity(labels: Vector[Int]): Double = {
    1 - labels.groupBy(x => x).
      mapValues(x => x.length.toDouble / labels.length.toDouble).
      mapValues(p => p * p).
      foldLeft(0.0)(_ + _._2)
  }

  def misclassificationError(labels: Vector[Int]): Double = {
    1 - labels.groupBy(x => x).
      map(x => x._2.length.toDouble / labels.length.toDouble).
      maxBy(x => x)
  }

  def meanSquaredError(labels: Vector[Double], predictions: Vector[Double]): Double = {
    labels.zip(predictions).
      foldLeft(0.0)((acc, t) =>
        acc + scala.math.pow(t._1 - t._2, 2)) / predictions.length.toDouble
  }

  def entropy(labels: Vector[Int]): Double = {
    labels.groupBy(x => x).
      mapValues(x => x.length.toDouble / labels.length).
      mapValues(p => -p * log2(p)).
      foldLeft(0.0)(_ + _._2)
  }

}
