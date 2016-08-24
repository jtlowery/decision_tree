package decision_tree

import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.Matchers._

import Dtree._

@RunWith(classOf[JUnitRunner])
class DtreeSuite extends FunSuite {

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

    val splitResult = splitVariable(d, 1, 1)
    assert(splitResult.get.leftData === Vector(d1, d2))
    assert(splitResult.get.rightData === Vector(d3, d4))
    assert(splitResult.get.iGain === 1.0)
    assert(splitResult.get.index === 1)
    assert(splitResult.get.predicate(1) === true)
    assert(splitResult.get.predicate(0) === false)
  }

  test("find best split on another col -- splitVariable") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)

    val splitResult = splitVariable(d, 0, 1)
    assert(splitResult.get.leftData === Vector(d1, d2, d4))
    assert(splitResult.get.rightData === Vector(d3))
    assert(splitResult.get.iGain === infoGain(Vector(1, 1, 2, 2), Vector(1, 1, 2), Vector(2)))
    assert(splitResult.get.index === 0)
    assert(splitResult.get.predicate(1) === true)
    assert(splitResult.get.predicate(0) === false)
  }

  test("find best split overall -- decideSplit") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)

    val splitResult = splitVariable(d, 1, 1)
    assert(splitResult.get.leftData === Vector(d1, d2))
    assert(splitResult.get.rightData === Vector(d3, d4))
    assert(splitResult.get.iGain === 1.0)
    assert(splitResult.get.index === 1)
    assert(splitResult.get.predicate(1) === true)
    assert(splitResult.get.predicate(0) === false)
  }

  test("fit basic") {
    val d1 = Vector(1, 1, 1, 1)
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val d4 = Vector(1, 0, 0, 2)
    val d = Vector(d1, d2, d3, d4)


    val fitResult = fit(d)
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

    val fitResult = fit(d, depth = 0, maxDepth = 1)
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

    val zeroEntropyData = Vector(d1, d2)
    assert(terminateSplitting(zeroEntropyData, depth = 1, maxDepth = 10) === true)

    val nonZeroEntropyData = Vector(d1, d2, d3)
    assert(terminateSplitting(nonZeroEntropyData, depth = 1, maxDepth = 10) === false)
  }

  test("test terminate splitting on depth") {
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val nonZeroEntropyData = Vector(d2, d3)
    assert(terminateSplitting(nonZeroEntropyData, depth = 0, maxDepth = 1) === true)
    assert(terminateSplitting(nonZeroEntropyData, depth = 0, maxDepth = 10) === false)
  }

  test("minSampleSplit stops a split would otherwise happen") {
    val d2 = Vector(1, 1, 0, 1)
    val d3 = Vector(0, 0, 1, 2)
    val nonZeroEntropyData = Vector(d2, d3)
    assert(fit(nonZeroEntropyData, depth = 0, maxDepth = 10, minSamplesSplit = 2) === Leaf(2))
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

    val testTree = Branch(Leaf(1), Leaf(2), 0, 1.0, 1, (x: AnyVal) => x == 1, d)
    assert(predict(testTree, d1) === 1)
    assert(predict(testTree, d2) === 1)
    assert(predict(testTree, d3) === 2)
    assert(predict(testTree, d4) === 2)

    val testTree2 = Branch(Leaf(2), Leaf(1), 0, 1.0, 1, (x: AnyVal) => x == 0, d)
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

}
