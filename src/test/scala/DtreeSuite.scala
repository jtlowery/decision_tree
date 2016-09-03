package decision_tree

import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.Matchers._

import Dtree._
import Criterion._

@RunWith(classOf[JUnitRunner])
class DtreeSuite extends FunSuite {

  test("dataprep - simple cases") {
    assert(
      prepareData(rawData = Vector(Vector(Left(1))), labels = Vector(1)) ===
        Vector(TreeData(features = Vector(Left(1)), label = 1, rowIndex = 0))
    )
    assert(
      prepareData(rawData = Vector(Vector(Left(1), Left(1)), Vector(Left(2), Left(1))), labels = Vector(1, 0)) ===
        Vector(TreeData(features = Vector(Left(1), Left(1)), label = 1, rowIndex = 0),
          TreeData(features = Vector(Left(2), Left(1)), label = 0, rowIndex = 1))
    )
  }

  test("dataRowIndexLookup - basic cases") {
    val td1 = TreeData(features = Vector(Left(1), Left(1)), label = 1, rowIndex = 0)
    val td2 = TreeData(features = Vector(Left(2), Left(1)), label = 0, rowIndex = 1)
    val td3 = TreeData(features = Vector(Left(2), Left(2)), label = 0, rowIndex = 2)

    assert(
      dataRowLookup(Vector(0), Vector(td1)) === Vector(td1)
    )
    assert(
      dataRowLookup(Vector(1), Vector(td1, td2)) === Vector(td2)
    )
    assert(
      dataRowLookup(Vector(0, 2), Vector(td1, td2, td3)) === Vector(td1, td3)
    )
  }

  test("information gain simple cases") {
    assert(infoGain(Vector(1, 1, 2, 2), Vector(1,1), Vector(2,2)) === 1.0)
    assert(infoGain(Vector(1, 1, 2, 2), Vector(1,2), Vector(1,2)) === 0.0)
    assert(infoGain(Vector(1, 1, 2, 2), Vector(1,1,2), Vector(2)) === 0.3112 +- .002)
  }

  test("information gain and entropy return same result") {
    assert(entropy(Vector(1,1,2,2)) - .75 * entropy(Vector(1,1,2)) - .25*entropy(Vector(2)) ===
      (1.0 - 0.75 * .9182958340544896 - 0.25*0.0) +- .002)
  }

  test("find best split on a col -- splitVariable") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val dt = prepareData(rawData = data, labels = labels)
    val splitResult = splitVariable(dt, 1, 1)
    assert(splitResult.get.leftDataRowIndexes === Vector(0, 1))
    assert(splitResult.get.rightDataRowIndexes === Vector(2, 3))
    assert(splitResult.get.iGain === 1.0)
    assert(splitResult.get.colIndex === 1)
    assert(splitResult.get.predicate(Left(1)) === true)
    assert(splitResult.get.predicate(Left(0)) === false)
  }

  test("find best split on another col -- splitVariable") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val dt = prepareData(rawData = data, labels = labels)
    val splitResult = splitVariable(dt, colIdx = 0, minSamplesSplit = 1)
    assert(splitResult.get.leftDataRowIndexes === Vector(0, 1, 3))
    assert(splitResult.get.rightDataRowIndexes === Vector(2))
    assert(splitResult.get.iGain === infoGain(Vector(1, 1, 2, 2), Vector(1, 1, 2), Vector(2)))
    assert(splitResult.get.colIndex === 0)
    assert(splitResult.get.predicate(Left(1)) === true)
    assert(splitResult.get.predicate(Left(0)) === false)
  }

  test("splitCategorical - basic case") {
    val fd1 = FeatureDataPoint(feature = Left(1), label = 1, rowIndex = 0, colIndex = 0)
    val fd2 = FeatureDataPoint(feature = Left(1), label = 1, rowIndex = 1, colIndex = 0)
    val fd3 = FeatureDataPoint(feature = Left(2), label = 2, rowIndex = 2, colIndex = 0)
    val fd4 = FeatureDataPoint(feature = Left(2), label = 2, rowIndex = 3, colIndex = 0)
    val split = splitCategorical(Vector(fd1, fd2, fd3, fd4), minSamplesSplit = 1)
    assert(split.get.leftDataRowIndexes === Vector(0, 1))
    assert(split.get.rightDataRowIndexes === Vector(2, 3))
    assert(split.get.iGain === infoGain(Vector(1, 1, 2, 2), Vector(1, 1), Vector(2, 2)))
    assert(split.get.colIndex === 0)
    assert(split.get.predicate(Left(1)) === true)
    assert(split.get.predicate(Left(0)) === false)
  }

  test("splitCategorical - uneven split") {
    val fd1 = FeatureDataPoint(feature = Left(1), label = 1, rowIndex = 0, colIndex = 0)
    val fd2 = FeatureDataPoint(feature = Left(1), label = 1, rowIndex = 1, colIndex = 0)
    val fd3 = FeatureDataPoint(feature = Left(1), label = 1, rowIndex = 2, colIndex = 0)
    val fd4 = FeatureDataPoint(feature = Left(2), label = 2, rowIndex = 3, colIndex = 0)
    val split = splitCategorical(Vector(fd1, fd2, fd3, fd4), minSamplesSplit = 1)
    assert(split.get.leftDataRowIndexes === Vector(0, 1, 2))
    assert(split.get.rightDataRowIndexes === Vector(3))
    assert(split.get.iGain === infoGain(Vector(1, 1, 1, 2), Vector(1, 1, 1), Vector(2)))
    assert(split.get.colIndex === 0)
    assert(split.get.predicate(Left(1)) === true)
    assert(split.get.predicate(Left(0)) === false)
  }

  test("find best split overall -- decideSplit") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val dt = prepareData(rawData = data, labels = labels)
    val splitResult = splitVariable(dt, 1, 1)
    assert(splitResult.get.leftDataRowIndexes === Vector(0, 1))
    assert(splitResult.get.rightDataRowIndexes === Vector(2, 3))
    assert(splitResult.get.iGain === 1.0)
    assert(splitResult.get.colIndex === 1)
    assert(splitResult.get.predicate(Left(1)) === true)
    assert(splitResult.get.predicate(Left(0)) === false)
  }

  test("fit basic") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val dt = prepareData(rawData = data, labels = labels)

    val fitResult = fit(dt, Vector(0,1,2,3))
    fitResult match {
      case Branch(l, r, depth, iGain, idx, splitPred, data) =>
        assert(l === Leaf(1))
        assert(r === Leaf(2))
        assert(depth === 0)
        assert(iGain === 1.0)
        assert(idx === 1)
        assert(splitPred(Left(1)) === true)
        assert(splitPred(Left(0)) === false)
        assert(data === Vector(0,1,2,3))
      case Leaf(_) => assert(false)
    }
  }

  test("test for maxDepth stopping of fit") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 1)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val dt = prepareData(rawData = data, labels = labels)
    val fitResult = fit(dt, dataRowIndexes = Vector(0,1,2,3), depth = 0, maxDepth = 1)
    fitResult match {
      case Leaf(x) => assert(x === 1)
      case Branch(l, r, depth, iGain, idx, splitPred, data) =>
        assert(false)
    }
  }

  test("test terminate splitting on entropy") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d = Vector(d1, d2)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val zeroEntropyData = prepareData(rawData = data, labels = labels)
    assert(terminateSplitting(zeroEntropyData, depth = 1, maxDepth = 10) === true)

    val dd = Vector(d1, d2, d3)
    val labels2 = dd.map(x => x.last)
    val data2 = dd.map(x => x.dropRight(1).map(z => Left(z)))
    val nonZeroEntropyData = prepareData(rawData = data2, labels = labels2)
    assert(terminateSplitting(nonZeroEntropyData, depth = 1, maxDepth = 10) === false)
  }

  test("test terminate splitting on depth") {
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d = Vector(d2, d3)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val dt = prepareData(rawData = data, labels = labels)
    val nonZeroEntropyData = prepareData(rawData = data, labels = labels)
    assert(terminateSplitting(nonZeroEntropyData, depth = 0, maxDepth = 1) === true)
    assert(terminateSplitting(nonZeroEntropyData, depth = 0, maxDepth = 10) === false)
  }

  test("minSampleSplit stops a split would otherwise happen") {
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d = Vector(d2, d3)
    val labels = d.map(x => x.last)
    val data = d.map(x => x.dropRight(1).map(z => Left(z)))
    val dt = prepareData(rawData = data, labels = labels)
    val nonZeroEntropyData = prepareData(rawData = data, labels = labels)
    assert(fit(nonZeroEntropyData, dataRowIndexes = Vector(0, 1), depth = 0, maxDepth = 10, minSamplesSplit = 2)
      === Leaf(2))
  }

  test("predict with a tree of a single leaf") {
    val d1 = Vector(Left(1), Left(1), Left(1), Left(1))
    assert(predict(Leaf(1), d1) === 1)
    assert(predict(Leaf(2), d1) === 2)
  }

  test("predict with a tree of two leaves and a single branch") {
    val d1 = Vector(1, 1, 1, 1).dropRight(1).map(x => Left(x))
    val d2 = Vector(1, 1, 0, 1).dropRight(1).map(x => Left(x))
    val d3 = Vector(0, 0, 1, 2).dropRight(1).map(x => Left(x))
    val d4 = Vector(1, 0, 0, 2).dropRight(1).map(x => Left(x))

    val testTree = Branch(left = Leaf(1), right = Leaf(2), depth = 0, informationGain = 1.0, splitCol = 1,
      splitPredicate = (x: Either[Int, Double]) => x == Left(1), dataRowIndexes = Vector(0,1,2,3))
    assert(predict(testTree, d1) === 1)
    assert(predict(testTree, d2) === 1)
    assert(predict(testTree, d3) === 2)
    assert(predict(testTree, d4) === 2)

    val testTree2 = Branch(Leaf(2), Leaf(1), 0, 1.0, 1, (x: Either[Int, Double]) => x == Left(0), Vector(0,1,2,3))
    assert(predict(testTree2, d1) === 1)
    assert(predict(testTree2, d2) === 1)
    assert(predict(testTree2, d3) === 2)
    assert(predict(testTree2, d4) === 2)
  }

  test("findType basic cases") {
    assert(findType(1) === "Int")
    assert(findType(1.0) === "Double")
  }


}
