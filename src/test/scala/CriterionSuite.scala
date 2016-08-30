package decision_tree

import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.Matchers._

import Criterion._

@RunWith(classOf[JUnitRunner])
class CriterionSuite extends FunSuite {

  test("log2 basic cases") {
    assert(log2(2.0) === 1.0)
    assert(log2(4.0) === 2.0)
  }

  test("entropy basic cases") {
    assert(entropy(Vector(1, 1, 2, 2)) === 1.0)
    assert(entropy(Vector(1, 1, 1, 1)) === 0.0)
    assert(entropy(Vector(1, 1, 1, 2)) === (.81 +- .002))
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
