package decision_tree

import decision_tree.Criterion._
sealed trait Dtree

case class Leaf(label: Int) extends Dtree
case class Branch(left: Dtree,
                  right: Dtree,
                  depth: Int,
                  informationGain: Double,
                  splitCol: Int,
                  splitPredicate: Either[Int, Double] => Boolean,
                  dataRowIndexes: Vector[Int]) extends Dtree
case class Split(leftDataRowIndexes: Vector[Int],
                 rightDataRowIndexes: Vector[Int],
                 iGain: Double,
                 colIndex: Int,
                 predicate: Either[Int, Double] => Boolean)
case class TreeData(features: Vector[Either[Int, Double]], label: Int, rowIndex: Int)
case class FeatureDataPoint(feature: Either[Int, Double], label: Int, rowIndex: Int, colIndex: Int)
case class BasicSplit(splitData: (Vector[FeatureDataPoint], Vector[FeatureDataPoint]),
                      predicate: Either[Int, Double] => Boolean)

object Dtree {

  type Labels = Vector[Int]
  type Point = Vector[Either[Int, Double]]
  type Points = Vector[Point]

  def prepareData(rawData: Points, labels: Labels): Vector[TreeData] = {
    rawData.
      zip(labels).
      zipWithIndex.
      map({case ((pt: Point, label: Int), rowIdx: Int) => TreeData(pt, label, rowIdx)})
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

  def findLabel(labels: Labels): Int = {
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
    featureData(0).feature match {
      case Left(i) => splitCategorical(featureData, minSamplesSplit)
      case Right(d) => splitContinuous(featureData, minSamplesSplit)
    }
  }

  def splitCategorical(data: Vector[FeatureDataPoint], minSamplesSplit: Int): Option[Split] = {
    // all possible categories
    val possibleVals = data.map(x => x.feature).distinct
    // find all partitions and corresponding predicates
    val basicSplits = possibleVals.map(p =>
      BasicSplit(data.partition(x => x.feature == p), (a: Either[Int, Double]) => a == p))
    val basicSplitsWithMinSamples = basicSplits.filter(x => x.splitData._1.size > minSamplesSplit)
    if (basicSplitsWithMinSamples.isEmpty) {
      None
    }
    else {
      val basicSplitsWithInfoGain = basicSplitsWithMinSamples.map(basicSplit => (basicSplit,
        infoGain(data.map(row => row.label),
                  basicSplit.splitData._1.map(l => l.label),
                  basicSplit.splitData._2.map(r => r.label))
        )
      )
      val maxPartition = basicSplitsWithInfoGain.maxBy({ case (basicSplit, infoGain) => infoGain })
      Some(Split(maxPartition._1.splitData._1.map(l => l.rowIndex),
                  maxPartition._1.splitData._2.map(r => r.rowIndex),
                  maxPartition._2, data(0).colIndex, maxPartition._1.predicate))
    }
  }

  def splitContinuous(data: Vector[FeatureDataPoint], minSamplesSplit: Int): Option[Split] = {
    // extract continuous vals and sort them
    val sortedVals = data.map(x => x.feature).sortBy(featureVal => featureVal.right.get)
    // find midpoints of sorted vals
    // TODO improve to only consider midpoints between examples from different classes
    val middleVals = sortedVals.
      zip(sortedVals.tail).
      map({ case (left, right) => left.right.get + (right.right.get - left.right.get) / 2 })
    val basicSplits = middleVals.map(p =>
      BasicSplit(data.partition(x => x.feature.right.get <= p), (a: Either[Int, Double]) => a.right.get <= p))
    val basicSplitsWithMinSamples = basicSplits.filter(x => x.splitData._1.size > minSamplesSplit)
    if (basicSplitsWithMinSamples.isEmpty) {
      None
    }
    else {
      val basicSplitsWithInfoGain = basicSplitsWithMinSamples.map(basicSplit => (basicSplit,
        infoGain(data.map(row => row.label),
                  basicSplit.splitData._1.map(l => l.label),
                  basicSplit.splitData._2.map(r => r.label))
        )
      )
      val maxPartition = basicSplitsWithInfoGain.maxBy({ case (basicSplit, infoGain) => infoGain })
      Some(Split(maxPartition._1.splitData._1.map(l => l.rowIndex),
                  maxPartition._1.splitData._2.map(r => r.rowIndex),
                  maxPartition._2, data(0).colIndex, maxPartition._1.predicate))
    }
  }

  def findType(o: AnyVal): String = o match {
    case d: Double => "Double"
    case i: Int => "Int"
    case _ => "Other"
  }

  def infoGain(parent: Labels, split1: Labels, split2: Labels): Double = {
    // this should be altered to allow entropy to be swapped out at some point
    val totalLength = (split1.length + split2.length).toDouble
    val splitsEntropy = (split1.length / totalLength) * entropy(split1) +
      (split2.length / totalLength) * entropy(split2)
    entropy(parent) - splitsEntropy
  }

}


