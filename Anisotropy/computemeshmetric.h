/** @file
 * @brief This function computes the implied metric, \c metric, using the nodes.
 * @param[out] - metric - Contains the metric of type Metric computed using the current mesh nodes. Size = nexD*(D+1)/2 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeMeshMetric(MeshMetric<D> & metric) {

  if (D != 2) {
    cout << "[ComputeMeshMetric] So far implemented for 2d only." << endl;
    exit(0);
  }

  double x1, x2, x3, y1, y2, y3;

  for (int i = 0; i < ne; i++) {
    ElementData<D,COMP> & ed = *eldata[i];

    x1 = ed.nodes[0];
    y1 = ed.nodes[1];
    x2 = ed.nodes[2];
    y2 = ed.nodes[3];
    x3 = ed.nodes[4];
    y3 = ed.nodes[5];
    metric.SetNodes(i, x1, y1, x2, y2, x3, y3);
  }
}
