import pyvista as pv
from shapely.geometry.polygon import Polygon, orient
import numpy as np

APPROX_ZERO = 1e-4


class Node:
    """
        node class for connecting line segments
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end


def polygonsFromMesh(zLevel: float, mesh: pv.PolyData, cutoff: float = APPROX_ZERO) -> 'list[Polygon]':
    """
    slices a mesh along a plane parallel to xy plane at height zLevel

    Parameters
    ----------
    zLevel: float
        z height to slice at
    mesh: Obj
        environment mesh
    Returns
    -------
    list[Polygons]
        list of polygons resulting from z slice
    """
    # hist = []
    # k = 0
    points = np.array(mesh.points[mesh.faces.reshape(-1, 4)[:, 1:]])
    vectors = np.roll(points, 1, axis=1) - points
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.einsum('k, ijk->ij', [0, 0, 1], np.subtract(points, np.array(
            [[0, 0, zLevel]]))) / np.einsum('ijk, k->ij', -vectors, [0, 0, 1])
    indexLine = np.sum((t > 0) & (t < 1), axis=1) > 1
    intersections = np.sum(indexLine)
    indexIntersection = (t[indexLine] > 0) & (t[indexLine] < 1)
    p = np.reshape(points[indexLine][indexIntersection], [intersections, 2, 3])
    d = np.reshape(vectors[indexLine][indexIntersection], [
                   intersections, 2, 3])
    s = np.reshape(t[indexLine][indexIntersection], [intersections, 2])
    segments = np.zeros_like(p)
    for ii in range(p.shape[0]):
        for jj in range(p.shape[1]):
            segments[ii, jj, :] = p[ii, jj, :] + s[ii, jj] * d[ii, jj, :]
    # make polygons out of segments
    if len(segments) <= 0:
        return []
    ring = [Node(segments[0, 1, :].copy(), segments[0, 0, :].copy())]
    segments[0, :, :] = np.inf
    rings = []
    misses = 0
    while not np.isinf(segments).all():
        vec = np.linalg.norm(segments - ring[-1].end, axis=2)
        i = np.argmin(vec, axis=0)
        # check for duplicate segment
        a = vec[i, [0, 1]] < cutoff
        if i[0] == i[1] and a.all():
            segments[i] = np.inf
            continue

        # if the end matches
        if a.any():
            misses = 0
            if a[0]:
                segment = segments[i[0], :, :].copy()
                ring.append(Node(segment[0, :], segment[1, :]))
                segments[i[0], :, :] = np.inf
            else:
                segment = segments[i[1], :, :].copy()
                ring.append(Node(segment[1, :], segment[0, :]))
                segments[i[1], :, :] = np.inf

            # check to see if loop closed
            if np.linalg.norm(ring[0].start - ring[-1].end) < cutoff:
                rings.append(ring)
                if not np.isinf(segments).all():
                    i, _, _ = np.where(~np.isinf(segments))
                    ring = [Node(segments[i[0], 1, :].copy(),
                                 segments[i[0], 0, :].copy())]
                    segments[i[0], :, :] = np.inf
        else:
            misses += 1

        if misses > 0 and not np.isinf(segments).all():
            # bad ring
            if len(ring) > 1:
                # try again without last segment
                ring.pop()
            else:
                i, _, _ = np.where(~np.isinf(segments))
                ring = [Node(segments[i[0], 1, :].copy(),
                             segments[i[0], 0, :].copy())]
                segments[i[0], :, :] = np.inf
    # plotting debugging
    #     hist.append((segments.copy(), ring.copy()))
    #     k += 1
    # fig = plt.figure()
    # ax = fig.add_subplot()

    # def update(k):
    #     ax.clear()
    #     if k > len(hist):
    #         return
    #     for segment in hist[k][0]:
    #         ax.set_title(k)
    #         ax.plot(segment[:, 0], segment[:, 1])
    #     path = [hist[k][1][0].start]
    #     for segment in hist[k][1]:
    #         path.append(segment.end)
    #     path = np.array(path)
    #     ax.plot(path[:, 0], path[:, 1], marker='x', color='m')

    # fig.subplots_adjust(left=0.25, bottom=0.25)
    # sAx = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    # slider = Slider(sAx, 'step', 0, len(hist) - 1, 0, valstep=1)
    # slider.on_changed(update)
    # # ani = FuncAnimation(fig, update, frames=range(len(hist)), blit=False)
    # plt.show()
    polygons = []
    for ring in rings:
        ps = []
        for current in ring:
            if np.linalg.norm(current.start - current.end) > cutoff:
                ps.append(current.start)
        ps.append(current.end)
        if len(ps) < 3:
            continue
        polygons.append(orient(Polygon(shell=ps)))

    return polygons
