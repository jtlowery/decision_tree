package decision_tree

sealed trait Dtree

case class Leaf(label: AnyVal) extends Dtree
case class Branch(left: Dtree,
                  right: Dtree,
                  depth: Int,
                  informationGain: Double,
                  splitCol: Int,
                  splitPredicate: AnyVal => Boolean,
                  dataRowIndexes: Vector[Int]) extends Dtree
case class Split(leftDataRowIndexes: Vector[Int],
                 rightDataRowIndexes: Vector[Int],
                 iGain: Double,
                 colIndex: Int,
                 predicate: AnyVal => Boolean)
case class TreeData(features: Vector[AnyVal], label: AnyVal, rowIndex: Int)
case class FeatureDataPoint(feature: AnyVal, label: AnyVal, rowIndex: Int, colIndex: Int)
case class BasicSplit(splitData: (Vector[FeatureDataPoint], Vector[FeatureDataPoint]),
                      predicate: AnyVal => Boolean)
object Dtree {

  type Labels = Vector[AnyVal]
  type Point = Vector[AnyVal]
  type Points = Vector[Point]

  def prepareData(rawData: Points): Vector[TreeData] = {
    rawData.zipWithIndex.map({case (pt: Point, rowIdx: Int) => TreeData(pt.dropRight(1), pt.last, rowIdx)})
  }

  def predict(dt: Dtree, point: Point): AnyVal =
    dt match {
      case Leaf(x) => x
      case Branch(left, right, depth, informationGain, splitCol, splitPredicate, dataRowIndexes) => {
        if (splitPredicate(point(splitCol))) predict(left, point)
        else predict(right, point)
      }
  }

  def fit(fullData: Vector[TreeData],
          dataRowIndexes: Vector[Int],
          depth: Int = 0,
          maxDepth: Int = 10,
          minSamplesSplit: Int = 1): Dtree = {

    val data = dataRowLookup(dataRowIndexes, fullData)
    // if splitting should be stopped - terminate in a leaf
    if (terminateSplitting(data, depth, maxDepth)) {
      Leaf(findLabel(data.map(row => row.label)))
    }
    else {
      val split = decideSplit(data, minSamplesSplit)
      // there may be no valid splits in which case we need to make a leaf
      if (split.isEmpty) {
        Leaf(findLabel(data.map(row => row.label)))
      }
      // if there is a valid split - make a branch
      // TODO I'm not confident that lazy val here is the right thing to do
      else {
        lazy val l = fit(fullData, split.get.leftDataRowIndexes, depth + 1, maxDepth)
        lazy val r = fit(fullData, split.get.rightDataRowIndexes, depth + 1, maxDepth)
        Branch(l, r, depth, split.get.iGain, split.get.colIndex, split.get.predicate, dataRowIndexes)
      }
    }
  }

  def dataRowLookup(dataRowIndexes: Vector[Int], data: Vector[TreeData]): Vector[TreeData] = {
    dataRowIndexes.map(rowIdx => data(rowIdx))
  }

  def terminateSplitting(data: Vector[TreeData], depth: Int, maxDepth: Int): Boolean = {
    (entropy(data.map(row => row.label)) == 0) || (depth >= maxDepth - 1)
  }

  def findLabel(labels: Labels): AnyVal = {
    labels.groupBy(x => x).maxBy(_._2.size)._1
  }

  def decideSplit(data: Vector[TreeData], minSamplesSplit: Int): Option[Split] = {
    // given Vector[TreeData] decide the best split
    // among possible features and values of the features
    val nFeatures = data(0).features.length
    val bestSplitsOnCols = (0 until nFeatures - 1).
      map(colIdx => splitVariable(data, colIdx, minSamplesSplit)).
      filter(split => split.isDefined)
    if (bestSplitsOnCols.isEmpty) {
      None
    }
    else {
      val bestSplit = bestSplitsOnCols.maxBy(m => m.get.iGain).get
      Some(Split(bestSplit.leftDataRowIndexes, bestSplit.rightDataRowIndexes,
        bestSplit.iGain, bestSplit.colIndex, bestSplit.predicate))
    }
  }

  def splitVariable(data: Vector[TreeData], colIdx: Int, minSamplesSplit: Int): Option[Split] = {
    val featureData = data.map(x =>
      FeatureDataPoint(feature = x.features(colIdx), label = x.label, rowIndex = x.rowIndex, colIndex = colIdx))
    if (findType(featureData(0).feature) == "Int") splitCategorical(featureData, minSamplesSplit)
    else splitContinuous(featureData, minSamplesSplit)
  }

  def splitCategorical(data: Vector[FeatureDataPoint], minSamplesSplit: Int): Option[Split] = {
    // all possible categories
    val possibleVals = data.map(x => x.feature).distinct
    // find all partitions and corresponding predicates
    val partitions = possibleVals.map(p =>
      BasicSplit(data.partition(x => x.feature == p), (a: AnyVal) => a == p))
    val partsWithMinSamples = partitions.filter(x => x.splitData._1.size > minSamplesSplit)
    if (partsWithMinSamples.isEmpty) {
      None
    }
    else {
      val partsWithInfoGain = partsWithMinSamples.map(x => (x,
        infoGain(data.map(row => row.label),
                  x.splitData._1.map(l => l.label),
                  x.splitData._2.map(r => r.label))))
      val maxPartition = partsWithInfoGain.maxBy(m => m._2)
      Some(Split(maxPartition._1.splitData._1.map(l => l.rowIndex),
                  maxPartition._1.splitData._2.map(r => r.rowIndex),
                  maxPartition._2, data(0).colIndex, maxPartition._1.predicate))
    }
  }

  def splitContinuous(data: Vector[FeatureDataPoint], minSamplesSplit: Int): Option[Split] = {
    ???
    /*
    // extract continuous vals and sort them
    val sortedVals = data.map(x => x(idx)).sortBy(x => x)
    // find midpoints of sorted vals
    // this could be improved to only consider midpoints between
    // examples from different classes
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
    */
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


