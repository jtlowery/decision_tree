package decision_tree



sealed trait Dtree
case class Leaf(label: AnyVal) extends Dtree
case class Branch(left: Dtree,
                  right: Dtree,
                  depth: Int,
                  informationGain: Double,
                  splitCol: Int,
                  splitPredicate: AnyVal => Boolean,
                  data: Vector[Vector[AnyVal]]) extends Dtree

object Dtree {

  type Labels = Vector[AnyVal]
  type Point = Vector[AnyVal]
  type Points = Vector[Point]


  def predict(dt: Dtree, point: Point): AnyVal =
    dt match {
      case Leaf(x) => x
      case Branch(left, right, depth, informationGain, splitCol, splitPredicate, data) => {
        if (splitPredicate(point(splitCol))) predict(left, point)
        else predict(right, point)
      }
  }

  def fit(data: Points, depth: Int = 0, maxDepth: Int = 10): Dtree = {
    val split = decideSplit(data)
    lazy val l =
      if (terminateSplitting(split._1, depth, maxDepth)) {
        Leaf(findLabel(split._1.map(x => x.last)))
      }
      else {
        fit(split._1, depth + 1, maxDepth)
      }
    lazy val r =
      if (terminateSplitting(split._2, depth, maxDepth)) {
        Leaf(findLabel(split._2.map(x => x.last)))
      }
      else {
        fit(split._2, depth + 1, maxDepth)
      }
    Branch(l, r, depth, split._3, split._4, split._5, data)
  }

  def terminateSplitting(data: Points, depth: Int, maxDepth: Int): Boolean = {
    (entropy(data.map(x => x.last)) == 0) || (depth >= maxDepth - 1)
  }

  def findLabel(labels: Labels): AnyVal = {
    labels.groupBy(x => x).maxBy(_._2.size)._1
  }

  def decideSplit(data: Points): (Points, Points, Double, Int, AnyVal => Boolean) = {
    val bestSplitsOnCols = (0 until data(0).length - 2).map(colIdx => splitVariable(data, colIdx))
    val bestSplit = bestSplitsOnCols.maxBy(m => m._3)
    (bestSplit._1, bestSplit._2, bestSplit._3, bestSplit._4, bestSplit._5)
  }

  def splitVariable(data: Points, idx: Int): (Points, Points, Double, Int, AnyVal => Boolean) = {
    val samplePoint = data(0)(idx)
    if (findType(samplePoint) == "Int") splitCategorical(data, idx)
    else splitContinuous(data, idx)
  }

  def splitCategorical(data: Points, idx: Int): (Points, Points, Double, Int, AnyVal => Boolean) = {
    val possibleVals = data.map(x => x(idx)).distinct
    val partitions = possibleVals.map(p =>
      (data.partition(x => x(idx) == p), (x: AnyVal) => x == p))
    val partsWithInfoGain = partitions.map(x => (x,
      infoGain(data.map(z => z.last), x._1._1.map(z => z.last), x._1._2.map(z => z.last))))
    val maxPartition = partsWithInfoGain.maxBy(m => m._2)
    (maxPartition._1._1._1, maxPartition._1._1._2, maxPartition._2, idx, maxPartition._1._2)
  }

  def splitContinuous(data: Points, idx: Int): (Points, Points, Double, Int, AnyVal => Boolean) = {
    ???
  }


  def log2(x: Double): Double = scala.math.log(x) / scala.math.log(2)

  def findType(o: AnyVal): String = o match {
    case d: Double => "Double"
    case i: Int => "Int"
    case _ => "Other"
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


