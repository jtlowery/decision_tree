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
    assert(entropy(Vector("a", "a", "b", "b")) === 1.0)
    assert(entropy(Vector("a", "a", "a", "a")) === 0.0)
    assert(entropy(Vector(1, 1, 1, 2)) === (.81 +- .002))
  }

  test("information gain simple cases") {
    assert(infoGain(Vector(1, 1, 2, 2), Vector(1,1), Vector(2,2)) === 1.0)
    assert(infoGain(Vector(1, 1, 2, 2), Vector(1,2), Vector(1,2)) === 0.0)
    assert(infoGain(Vector(1, 1, 2, 2), Vector(1,1,2), Vector(2)) === 0.3112 +- .002)
  }

  test("information gain and entropy return same result") {
    assert(entropy(Vector(1,1,2,2)) - .75 * entropy(Vector(1,1,2)) - .25*entropy(Vector(2)) ===
      (1.0 - 0.75*(.9182958340544896) - 0.25*0.0) +- .002)
  }

  test("findType basic cases") {
    assert(findType("ABC") === "String")
    assert(findType(1) === "Int")
    assert(findType(1.0) === "Double")
  }

}
