package decision_tree

import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.Matchers._

import Dtree._

@RunWith(classOf[JUnitRunner])
class DtreeSuite extends FunSuite {

  test("dataprep - simple cases") {
    assert(
      prepareData(Vector(Vector(1, 1))) === Vector(TreeData(features = Vector(1), label = 1, rowIndex = 0))
    )
    assert(
      prepareData(Vector(Vector(1, 1, 1), Vector(2, 1, 0))) ===
        Vector(TreeData(features = Vector(1, 1), label = 1, rowIndex = 0),
          TreeData(features = Vector(2, 1), label = 0, rowIndex = 1))
    )
  }

  test("dataRowIndexLookup - basic cases") {
    val td1 = TreeData(features = Vector(1, 1), label = 1, rowIndex = 0)
    val td2 = TreeData(features = Vector(2, 1), label = 0, rowIndex = 1)
    val td3 = TreeData(features = Vector(2, 2), label = 0, rowIndex = 2)

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

  test("log2 basic cases") {
    assert(log2(2.0) === 1.0)
    assert(log2(4.0) === 2.0)
  }

  test("entropy basic cases") {
    assert(entropy(Vector(1, 1, 2, 2)) === 1.0)
    assert(entropy(Vector(1, 1, 1, 1)) === 0.0)
    assert(entropy(Vector(1, 1, 1, 2)) === (.81 +- .002))
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
    val dt = prepareData(d)
    val splitResult = splitVariable(dt, 1, 1)
    assert(splitResult.get.leftDataRowIndexes === Vector(0, 1))
    assert(splitResult.get.rightDataRowIndexes === Vector(2, 3))
    assert(splitResult.get.iGain === 1.0)
    assert(splitResult.get.colIndex === 1)
    assert(splitResult.get.predicate(1) === true)
    assert(splitResult.get.predicate(0) === false)
  }

  test("find best split on another col -- splitVariable") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val dt = prepareData(d)
    val splitResult = splitVariable(dt, 0, 1)
    assert(splitResult.get.leftDataRowIndexes === Vector(0, 1, 3))
    assert(splitResult.get.rightDataRowIndexes === Vector(2))
    assert(splitResult.get.iGain === infoGain(Vector(1, 1, 2, 2), Vector(1, 1, 2), Vector(2)))
    assert(splitResult.get.colIndex === 0)
    assert(splitResult.get.predicate(1) === true)
    assert(splitResult.get.predicate(0) === false)
  }

  test("splitCategorical - basic case") {
    //splitCategorical(data: Vector[FeatureDataPoint], minSamplesSplit: Int): Option[Split]
    //FeatureDataPoint(feature: AnyVal, label: AnyVal, rowIndex: Int, colIndex: Int)
    val fd1 = FeatureDataPoint(feature = 1, label = 1, rowIndex = 0, colIndex = 0)
    val fd2 = FeatureDataPoint(feature = 1, label = 1, rowIndex = 1, colIndex = 0)
    val fd3 = FeatureDataPoint(feature = 2, label = 2, rowIndex = 2, colIndex = 0)
    val fd4 = FeatureDataPoint(feature = 2, label = 2, rowIndex = 3, colIndex = 0)
    val split = splitCategorical(Vector(fd1, fd2, fd3, fd4), minSamplesSplit = 1)
    assert(split.get.leftDataRowIndexes === Vector(0, 1))
    assert(split.get.rightDataRowIndexes === Vector(2, 3))
    assert(split.get.iGain === infoGain(Vector(1, 1, 2, 2), Vector(1, 1), Vector(2, 2)))
    assert(split.get.colIndex === 0)
    assert(split.get.predicate(1) === true)
    assert(split.get.predicate(0) === false)
  }

  test("splitCategorical - uneven split") {
    //splitCategorical(data: Vector[FeatureDataPoint], minSamplesSplit: Int): Option[Split]
    //FeatureDataPoint(feature: AnyVal, label: AnyVal, rowIndex: Int, colIndex: Int)
    val fd1 = FeatureDataPoint(feature = 1, label = 1, rowIndex = 0, colIndex = 0)
    val fd2 = FeatureDataPoint(feature = 1, label = 1, rowIndex = 1, colIndex = 0)
    val fd3 = FeatureDataPoint(feature = 1, label = 1, rowIndex = 2, colIndex = 0)
    val fd4 = FeatureDataPoint(feature = 2, label = 2, rowIndex = 3, colIndex = 0)
    val split = splitCategorical(Vector(fd1, fd2, fd3, fd4), minSamplesSplit = 1)
    assert(split.get.leftDataRowIndexes === Vector(0, 1, 2))
    assert(split.get.rightDataRowIndexes === Vector(3))
    assert(split.get.iGain === infoGain(Vector(1, 1, 1, 2), Vector(1, 1, 1), Vector(2)))
    assert(split.get.colIndex === 0)
    assert(split.get.predicate(1) === true)
    assert(split.get.predicate(0) === false)
  }

  test("find best split overall -- decideSplit") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val dt = prepareData(d)
    val splitResult = splitVariable(dt, 1, 1)
    assert(splitResult.get.leftDataRowIndexes === Vector(0, 1))
    assert(splitResult.get.rightDataRowIndexes === Vector(2, 3))
    assert(splitResult.get.iGain === 1.0)
    assert(splitResult.get.colIndex === 1)
    assert(splitResult.get.predicate(1) === true)
    assert(splitResult.get.predicate(0) === false)
  }

  test("fit basic") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val dt = prepareData(d)

    val fitResult = fit(dt, Vector(0,1,2,3))
    fitResult match {
      case Branch(l, r, depth, iGain, idx, splitPred, data) =>
        assert(l === Leaf(1))
        assert(r === Leaf(2))
        assert(depth === 0)
        assert(iGain === 1.0)
        assert(idx === 1)
        assert(splitPred(1) === true)
        assert(splitPred(0) === false)
        assert(data === d)
      case Leaf(_) => assert(false)
    }
  }

  test("test for maxDepth stopping of fit") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 1)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val dt = prepareData(d)
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

    val zeroEntropyData = prepareData(Vector(d1, d2))
    assert(terminateSplitting(zeroEntropyData, depth = 1, maxDepth = 10) === true)

    val nonZeroEntropyData = prepareData(Vector(d1, d2, d3))
    assert(terminateSplitting(nonZeroEntropyData, depth = 1, maxDepth = 10) === false)
  }

  test("test terminate splitting on depth") {
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val nonZeroEntropyData = prepareData(Vector(d2, d3))
    assert(terminateSplitting(nonZeroEntropyData, depth = 0, maxDepth = 1) === true)
    assert(terminateSplitting(nonZeroEntropyData, depth = 0, maxDepth = 10) === false)
  }

  test("minSampleSplit stops a split would otherwise happen") {
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val nonZeroEntropyData = prepareData(Vector(d2, d3))
    assert(fit(nonZeroEntropyData, dataRowIndexes = Vector(0, 1), depth = 0, maxDepth = 10, minSamplesSplit = 2)
      === Leaf(2))
  }

  test("predict with a tree of a single leaf") {
    val d1 = Vector(1, 1, 1, 1)
    assert(predict(Leaf(1), d1) === 1)
    assert(predict(Leaf(2), d1) === 2)
  }

  test("predict with a tree of two leaves and a single branch") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)
    val dt = prepareData(d)

    val testTree = Branch(Leaf(1), Leaf(2), 0, 1.0, 1, (x: AnyVal) => x == 1, Vector(0,1,2,3))
    assert(predict(testTree, d1) === 1)
    assert(predict(testTree, d2) === 1)
    assert(predict(testTree, d3) === 2)
    assert(predict(testTree, d4) === 2)

    val testTree2 = Branch(Leaf(2), Leaf(1), 0, 1.0, 1, (x: AnyVal) => x == 0, Vector(0,1,2,3))
    assert(predict(testTree2, d1) === 1)
    assert(predict(testTree2, d2) === 1)
    assert(predict(testTree2, d3) === 2)
    assert(predict(testTree2, d4) === 2)
  }

  test("findType basic cases") {
    assert(findType(1) === "Int")
    assert(findType(1.0) === "Double")
  }

  test("gini impurity - basic cases") {
    assert(giniImpurity(Vector(1, 1, 1)) === 0)
    assert(giniImpurity(Vector(1, 1, 0, 0)) === 0.5)
    giniImpurity(Vector(1, 0, 0, 0, 0, 0)) should be (0.278 +- .001)
  }

  test("misclassification error - basic cases") {
    assert(misclassificationError(Vector(1, 1, 1)) === 0)
    assert(misclassificationError(Vector(1, 1, 0, 0)) === 0.5)
    misclassificationError(Vector(1, 0, 0, 0, 0, 0)) should be (0.167 +- .001)
  }

  test("mean squared error - basic cases") {
    assert(meanSquaredError(Vector(1.1, 2.2, 3.2), Vector(1.1, 2.2, 3.2)) === 0.0)
    meanSquaredError(Vector(1.1, 2.2, 3.2), Vector(1.0, 2.0, 3.0)) should be (.03 +- .0001)
    meanSquaredError(Vector(1.0), Vector(2.0)) should be (1.0 +- .001)
  }
}
