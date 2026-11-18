// Export one graph per source file (= one benchmark function) from a Joern CPG.
//
//   joern --script export_graphs.sc --param cpgFile=chunk.bin --param outFile=chunk.jsonl
//
// For each file we take its largest non-synthetic method (the benchmark function)
// and write one JSON line:
//   {"file": "<sample id>.<ext>", "method": name,
//    "nodes": [[id, label, code, name, line, order, typeFullName, isCfg]],
//    "edges": [[src, dst, type]]}           type in AST, CFG, CDG, REACHING_DEF
// Node ids are CPG ids; consumers re-index.
import io.shiftleft.codepropertygraph.generated.EdgeTypes

@main def exec(cpgFile: String, outFile: String) = {
  importCpg(cpgFile)
  val out = new java.io.PrintWriter(new java.io.File(outFile), "UTF-8")
  val keepEdges = Set(EdgeTypes.AST, EdgeTypes.CFG, EdgeTypes.CDG, EdgeTypes.REACHING_DEF)

  def q(s: String): String = ujson.write(ujson.Str(Option(s).getOrElse("")))

  val byFile = cpg.method.isExternal(false)
    .filterNot(m => m.name.startsWith("<") || m.name.contains("<lambda>"))
    .l.groupBy(_.filename)

  byFile.foreach { case (fname, ms) =>
    val m = ms.maxBy(x => x.ast.size)
    val nodes = m.ast.l
    val ids = nodes.map(_.id()).toSet
    val cfgIds = m.cfgNode.map(_.id()).toSet
    val sb = new StringBuilder
    sb.append("{\"file\":").append(q(new java.io.File(fname).getName))
      .append(",\"method\":").append(q(m.name)).append(",\"nodes\":[")
    nodes.zipWithIndex.foreach { case (n, i) =>
      if (i > 0) sb.append(',')
      val props = n.properties
      val code = props.get("CODE").map(_.toString).orNull
      val name = props.get("NAME").map(_.toString).orNull
      val tfn = props.get("TYPE_FULL_NAME").map(_.toString).orNull
      val line = props.get("LINE_NUMBER").map(_.toString).getOrElse("-1")
      val order = props.get("ORDER").map(_.toString).getOrElse("-1")
      sb.append('[').append(n.id()).append(',').append(q(n.label)).append(',').append(q(code))
        .append(',').append(q(name)).append(',').append(line).append(',').append(order)
        .append(',').append(q(tfn)).append(',').append(if (cfgIds.contains(n.id())) 1 else 0).append(']')
    }
    sb.append("],\"edges\":[")
    var first = true
    nodes.foreach { n =>
      n.outE.foreach { e =>
        val dst = e.dst.id()
        if (keepEdges.contains(e.label) && ids.contains(dst)) {
          if (!first) sb.append(',')
          first = false
          sb.append('[').append(n.id()).append(',').append(dst).append(',').append(q(e.label)).append(']')
        }
      }
    }
    sb.append("]}")
    out.println(sb.toString)
  }
  out.close()
}
