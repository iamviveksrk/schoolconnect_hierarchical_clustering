import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

X = np.array([
    [5, 5], [5, 6],
    [7, 5], [8, 6],
    [6, 10], [5, 12], [8, 10]
])

from itertools import combinations

def agglomerative(X, linkage):

  clusters = [[tuple(i)] for i in X]
  n = len(X)

  X_i = dict(zip([tuple(i) for i in X], range(n)))
  merged = []
  linkage_matrix = []
  Zs = []

  for _ in range(n-1):

    min_dist = np.inf
    current_clusters = [i for i in range(len(clusters)) if i not in merged]

    # Compute linkage for all clusters pairwise to find best cluster-pair
    for c1, c2 in combinations(current_clusters, 2):

      linkage_val = linkage(clusters[c1], clusters[c2])

      # Find cluster-pair with smallest linkage
      if linkage_val < min_dist:
        min_dist = linkage_val
        best_pair = sorted([c1, c2])

    # Merge the best pair and append to clusters
    clusters.append(clusters[best_pair[0]] + clusters[best_pair[1]])

    # Add best pair clusters to merged
    merged += best_pair

    linkage_matrix.append(best_pair + [min_dist, len(clusters[-1])])

    # Append cluster indicator array Z to Zs
    Z = np.zeros(n)
    for c in current_clusters:
      for i in clusters[c]:
        Z[X_i[i]] = c

    Zs.append(Z)

  Zs.append([len(clusters)-1]*n)

  return np.array(linkage_matrix), np.array(Zs)

def single(cluster_1, cluster_2):
  single_linkage_val = np.inf

  for p1 in cluster_1:
    for p2 in cluster_2:

      p1_p2_dist = np.linalg.norm(np.array(p1)-np.array(p2))

      if single_linkage_val > p1_p2_dist:
        single_linkage_val = p1_p2_dist

  return single_linkage_val

def cluster_rename(Z):
  renamed_Z = []
  mapping = {}
  x = len(Z)

  for i in Z:
    try:
      renamed_Z.append(mapping[i])
    except:
      mapping[i] = x
      x -= 1
      renamed_Z.append(mapping[i])

  return renamed_Z

def plot_ellipse(X, ax):
  cov = np.cov(X[:, 0], X[:, 1])
  val, rot = np.linalg.eig(cov)
  val = np.sqrt(val)
  if min(val)<=0.01:
    val += 0.2 * max(val)
  center = np.mean([X[:, 0], X[:, 1]], axis=1)[:, None]

  t = np.linspace(0, 2.0 * np.pi, 1000)
  xy = np.stack((np.cos(t), np.sin(t)), axis=-1)

  return ax.plot(*(2 * rot @ (val * xy).T + center))