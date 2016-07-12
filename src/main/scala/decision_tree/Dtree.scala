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
    ???
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
      foldLeft(0.0)(_+_._2)
  }

  def infoGain(parent: Labels, split1: Labels, split2: Labels): Double = {
    val totalLength = (split1.length + split2.length).toDouble
    val splitsEntropy = (split1.length / totalLength) * entropy(split1) +
      (split2.length / totalLength) * entropy(split2)
    entropy(parent) - splitsEntropy
  }


}


