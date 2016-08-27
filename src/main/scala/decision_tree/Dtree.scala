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
case class Split(leftData: Vector[Vector[AnyVal]],
                 rightData: Vector[Vector[AnyVal]],
                 iGain: Double,
                 index: Int,
                 predicate: AnyVal => Boolean)
case class BasicSplit(splitData: (Vector[Vector[AnyVal]], Vector[Vector[AnyVal]]),
                      predicate: AnyVal => Boolean)

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

  def fit(data: Points, depth: Int = 0, maxDepth: Int = 10, minSamplesSplit: Int = 1): Dtree = {
    if (terminateSplitting(data, depth, maxDepth)) {
      Leaf(findLabel(data.map(row => row.last)))
    }
    else {
      val split = decideSplit(data, minSamplesSplit)
      if (split.isEmpty) {
        Leaf(findLabel(data.map(row => row.last)))
      }
      else {
        lazy val l = fit(split.get.leftData, depth + 1, maxDepth)
        lazy val r = fit(split.get.rightData, depth + 1, maxDepth)
        Branch(l, r, depth, split.get.iGain, split.get.index, split.get.predicate, data)
      }
    }
  }

  def terminateSplitting(data: Points, depth: Int, maxDepth: Int): Boolean = {
    (entropy(data.map(x => x.last)) == 0) || (depth >= maxDepth - 1)
  }

  def findLabel(labels: Labels): AnyVal = {
    labels.groupBy(x => x).maxBy(_._2.size)._1
  }

  def decideSplit(data: Points, minSamplesSplit: Int): Option[Split] = {
    val bestSplitsOnCols = (0 until data(0).length - 2).
      map(colIdx => splitVariable(data, colIdx, minSamplesSplit)).
      filter(split => split.isDefined)
    if (bestSplitsOnCols.isEmpty) {
      None
    }
    else {
      val bestSplit = bestSplitsOnCols.maxBy(m => m.get.iGain).get
      Some(Split(bestSplit.leftData, bestSplit.rightData,
        bestSplit.iGain, bestSplit.index, bestSplit.predicate))
    }
  }

  def splitVariable(data: Points, idx: Int, minSamplesSplit: Int): Option[Split] = {
    val samplePoint = data(0)(idx)
    if (findType(samplePoint) == "Int") splitCategorical(data, idx, minSamplesSplit)
    else splitContinuous(data, idx, minSamplesSplit)
  }

  def splitCategorical(data: Points, idx: Int, minSamplesSplit: Int): Option[Split] = {
    val possibleVals = data.map(x => x(idx)).distinct
    val partitions = possibleVals.map(p =>
      BasicSplit(data.partition(x => x(idx) == p), (x: AnyVal) => x == p))
    val partsWithMinSamples = partitions.filter(x => x.splitData._1.size > minSamplesSplit)
    if (partsWithMinSamples.isEmpty) {
      None
    }
    else {
      val partsWithInfoGain = partsWithMinSamples.map(x => (x,
        infoGain(data.map(row => row.last),
                  x.splitData._1.map(row => row.last),
                  x.splitData._2.map(row => row.last))))
      val maxPartition = partsWithInfoGain.maxBy(m => m._2)
      Some(Split(maxPartition._1.splitData._1, maxPartition._1.splitData._2,
        maxPartition._2, idx, maxPartition._1.predicate))
    }
  }

  def splitContinuous(data: Points, idx: Int, minSamplesSplit: Int): Option[Split] = {
    // extract continuous vals and sort them
    val sortedVals = data.map(x => x(idx)).sortBy(x => x)
    // find midpoints of sorted vals
    val middleVals = sortedVals.
      zip(sortedVals.tail).
      map({ case (left: Double, right: Double) => left + (right - left) / 2 })
    val partitions = middleVals.map(p =>
      BasicSplit(data.partition(x => x(idx) <= p), (x: AnyVal) => x <= p))
    val partsWithMinSamples = partitions.filter(x => x.splitData._1.size > minSamplesSplit)
    if (partsWithMinSamples.isEmpty) {
      None
    }
    else {
      val partsWithInfoGain = partsWithMinSamples.map(x => (x,
        infoGain(data.map(row => row.last),
          x.splitData._1.map(row => row.last),
          x.splitData._2.map(row => row.last))))
      val maxPartition = partsWithInfoGain.maxBy(m => m._2)
      Some(Split(maxPartition._1.splitData._1, maxPartition._1.splitData._2,
        maxPartition._2, idx, maxPartition._1.predicate))
    }
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

  def giniImpurity(labels: Labels): Double = {
    1 - labels.groupBy(x => x).
      mapValues(x => x.length.toDouble / labels.length.toDouble).
      mapValues(p => p * p).
      foldLeft(0.0)(_ + _._2)
  }

  def misclassificationError(labels: Labels): Double = {
    1 - labels.groupBy(x => x).
      map(x => x._2.length.toDouble / labels.length.toDouble).
      maxBy(x => x)
  }

  def meanSquaredError(labels: Vector[Double], predictions: Vector[Double]): Double = {
    labels.zip(predictions).
      foldLeft(0.0)((acc, t) =>
        acc + scala.math.pow(t._1 - t._2, 2)) / predictions.length.toDouble
  }
}


