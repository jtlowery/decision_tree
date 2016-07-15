package decision_tree



sealed trait Dtree
case class Leaf(label: Any) extends Dtree
case class Branch(left: Dtree, right: Dtree, splitValue: (Int, Any),
                  data: Vector[Vector[Any]]) extends Dtree


object Dtree {

  type Labels = Vector[Any]
  type Point = Vector[Any]
  type Points = Vector[Point]


  def predict(dt: Dtree, point: Point): Any =
    dt match {
      case Leaf(x) => x
      case Branch(left, right, splitValue, data) => {
        if (splitValue._2 == point(splitValue._1)) predict(left, point)
        else predict(right, point)
      }
  }

  def fit(data: Points): Dtree = {
    val split = decideSplit(data)
    lazy val l = if (entropy(split._1.map(x => x.last)) == 0) Leaf(split._1.last.last) else fit(split._1)
    lazy val r = if (entropy(split._2.map(x => x.last)) == 0) Leaf(split._2.last.last) else fit(split._2)
    Branch(l, r, split._3, data)
  }

  def decideSplit(data: Points): (Points, Points, (Int, Any)) = {
    val bestSplitsOnCols = (0 until data(0).length - 2).map(col => splitVariable(data, col))
    val bestSplitWithInfoGain = bestSplitsOnCols.map(x => (x,
      infoGain(data.map(z => z.last), x._1.map(z => z.last), x._2.map(z => z.last)))).
      maxBy(m => m._2)
    bestSplitWithInfoGain._1
  }

  def splitVariable(data: Points, idx: Int): (Points, Points, (Int, Any)) = {
    val possibleVals = data.map(x => x(idx)).distinct
    val partitions = possibleVals.map(p => data.partition(x => x(idx) == p))
    val partsWithInfoGain = partitions.map(x => (x,
      infoGain(data.map(z => z.last), x._1.map(z => z.last), x._2.map(z => z.last))))
    val maxPartition = partsWithInfoGain.maxBy(m => m._2)._1
    (maxPartition._1, maxPartition._2, (idx, maxPartition._1.last.last))
  }

  def log2(x: Double): Double = scala.math.log(x) / scala.math.log(2)

  def findType(o: Any): String = o match {
    case d: Double => "Double"
    case i: Int => "Int"
    case s: String => "String"
  }

  def entropy(labels: Labels): Double = {
    labels.groupBy(x => x).
      mapValues(x => x.length.toDouble / labels.length).
      mapValues(p => -p * log2(p)).
      foldLeft(0.0)(_ + _._2)
  }

  def infoGain(parent: Labels, split1: Labels, split2: Labels): Double = {
    val totalLength = (split1.length + split2.length).toDouble
    val splitsEntropy = (split1.length / totalLength) * entropy(split1) +
      (split2.length / totalLength) * entropy(split2)
    entropy(parent) - splitsEntropy
  }


}


