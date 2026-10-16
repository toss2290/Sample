"""Session-local work meshes and identity-based, recoverable edit-mesh patches."""

import bmesh
import bpy


DOMAINS = ("verts", "edges", "faces")
LAYER_KINDS = (
    "bool", "int", "float", "string", "float_vector", "color", "float_color",
    "deform", "uv",
)


def layer_pairs(source, target, create=True):
    """Match custom data by domain, type and name before creating any elements.

    Adding a layer can invalidate Python element wrappers. Never merge layouts
    inside an element-copy loop; copy_from also copies internal wrapper flags.
    """
    pairs = {}
    for domain in (*DOMAINS, "loops"):
        left = getattr(source, domain).layers
        right = getattr(target, domain).layers
        entries = []
        for kind in LAYER_KINDS:
            collection = getattr(left, kind, None)
            destination = getattr(right, kind, None)
            if collection is None or destination is None:
                continue
            for layer in collection:
                other = destination.active if kind == "deform" else destination.get(layer.name)
                if other is None:
                    if not create:
                        raise ValueError(f"Missing live {domain} layer: {kind}/{layer.name}")
                    other = destination.new(layer.name)
                entries.append((kind, layer, other))
        pairs[domain] = entries
    return pairs


def copy_values(source, target, pairs):
    if not isinstance(source, bmesh.types.BMLoop):
        target.select = source.select
        target.hide = source.hide
    if isinstance(source, bmesh.types.BMVert):
        target.co = source.co
    elif isinstance(source, bmesh.types.BMEdge):
        target.seam = source.seam
        target.smooth = source.smooth
    elif isinstance(source, bmesh.types.BMFace):
        target.material_index = source.material_index
        target.smooth = source.smooth
    for kind, left, right in pairs:
        if kind == "deform":
            values = target[right]
            values.clear()
            for group, weight in source[left].items():
                values[group] = weight
        elif kind == "uv":
            target[right].uv = source[left].uv
            if bpy.app.version < (5, 0, 0):
                target[right].pin_uv = source[left].pin_uv
                target[right].select = source[left].select
                target[right].select_edge = source[left].select_edge
            # UV selection/pinning live in bool layers in Blender 5+. Reading
            # the wrapper properties can lazily add layers and invalidate elements.
        else:
            target[right] = source[left]


def expanded_faces(paths, captured, rings=3):
    """Seed at both sides of the rails, then grow through vertex adjacency.

    Every face and edge incident to a path vertex is included, even on a
    non-manifold branch, so detachment checks see the same topology as the source.
    Captured dependencies are included explicitly rather than truncated at a radius.
    """
    path_vertices = {vertex for vertices, _, _ in paths for vertex in vertices}
    faces = {face for vertex in path_vertices for face in vertex.link_faces}
    faces.update(face for region in captured["regions"] for face in region["faces"])
    frontier = set(faces)
    for _ in range(rings):
        following = {
            linked for face in frontier for vertex in face.verts
            for linked in vertex.link_faces
        } - faces
        faces.update(following)
        frontier = following
        if not frontier:
            break
    edges = {edge for face in faces for edge in face.edges}
    edges.update(edge for vertex in path_vertices for edge in vertex.link_edges)
    edges.update(edge for region in captured["regions"] for edge in region["delete_edges"])
    vertices = {vertex for edge in edges for vertex in edge.verts} | path_vertices
    vertices.update(vertex for region in captured["regions"]
                    for vertex in region.get("outer_vertices") or ())
    vertices.update(region["pole_vertex"] for region in captured["regions"]
                    if region.get("pole_vertex") is not None)
    return vertices, edges, faces


class LocalPatch:
    """A pristine local source, stable source IDs and the last successful result."""

    def __init__(self, source, live, paths, captured):
        self.source = bmesh.new()
        self.previous = None
        self.previous_origins = None
        self.created = ([], [], [])
        try:
            pairs = layer_pairs(source, self.source)
            vertices, edges, faces = expanded_faces(paths, captured)
            # Preserve source order and source face records (including tie-breaking IDs).
            ordered = [sorted(elements, key=lambda item: item.index)
                       for elements in (vertices, edges, faces)]
            self.source_ids = tuple(tuple(item.index for item in elements) for elements in ordered)
            self.ids = tuple({old: new for new, old in enumerate(indices)}
                             for indices in self.source_ids)
            vertex_map = {}
            for vertex in ordered[0]:
                other = self.source.verts.new(vertex.co)
                copy_values(vertex, other, pairs["verts"])
                vertex_map[vertex] = other
            for edge in ordered[1]:
                other = self.source.edges.new(tuple(vertex_map[v] for v in edge.verts))
                copy_values(edge, other, pairs["edges"])
            for face in ordered[2]:
                other = self.source.faces.new(tuple(vertex_map[v] for v in face.verts))
                copy_values(face, other, pairs["faces"])
                for left, right in zip(face.loops, other.loops):
                    copy_values(left, right, pairs["loops"])
            for domain in DOMAINS:
                sequence = getattr(self.source, domain)
                sequence.index_update()
                sequence.ensure_lookup_table()
            self.source.normal_update()
            self.bind(live)
        except Exception:
            self.source.free()
            raise

    def remap_capture(self, cache):
        vertices, edges, faces = self.ids
        result = dict(cache)
        result["element_counts"] = tuple(len(getattr(self.source, d)) for d in DOMAINS)
        result["paths"] = tuple((tuple(vertices[i] for i in vs), tuple(edges[i] for i in es), closed)
                                for vs, es, closed in cache["paths"])
        regions = []
        for frozen in cache["regions"]:
            region = dict(frozen)
            region["face_indices"] = tuple(faces[i] for i in frozen["face_indices"])
            region["delete_edge_indices"] = tuple(edges[i] for i in frozen["delete_edge_indices"])
            outer = frozen["outer_vertex_indices"]
            region["outer_vertex_indices"] = None if outer is None else tuple(vertices[i] for i in outer)
            pole = frozen["pole_vertex_index"]
            region["pole_vertex_index"] = None if pole is None else vertices[pole]
            regions.append(region)
        result["regions"] = tuple(regions)
        return result

    def bind(self, live):
        """Only called initially or after a complete recovery from the source."""
        # Retain the Python BMesh owner: dropping the wrapper invalidates element
        # wrappers even though Blender still owns the underlying edit mesh.
        self.live_bm = live
        self.live = []
        self.created = ([], [], [])
        self.selection = ([], [], [])
        for domain, indices in zip(DOMAINS, self.source_ids):
            sequence = getattr(live, domain)
            sequence.index_update()
            sequence.ensure_lookup_table()
            self.live.append([sequence[index] for index in indices])
        self.selection = tuple([item for item in getattr(live, d) if item.select] for d in DOMAINS)

    def commit(self, live, work, origins):
        """Remove the last patch and publish a fully validated local result.

        The caller owns recovery because BMesh edits are not an atomic transaction.
        Original context faces survive; only deleted source elements and previously
        created elements are removed. Boundary vertices are matched by identity.
        """
        # Live element identity must survive schema matching. The working copy
        # inherits every layer from the source; unexpected additions require recovery.
        pairs = layer_pairs(work, live, create=False)
        removed = tuple(
            {item for item in self.created[offset] if item.is_valid}
            | {target for before, target in zip(origins[offset], self.live[offset])
               if not before.is_valid and target.is_valid}
            for offset in range(3)
        )
        if any(face not in removed[2] for edge in removed[1] for face in edge.link_faces):
            raise ValueError("Local patch would remove a face outside its captured region.")
        if any(edge not in removed[1] for vertex in removed[0] for edge in vertex.link_edges):
            raise ValueError("Local patch would remove an edge outside its captured region.")
        for group in self.selection:
            for item in group:
                if item.is_valid:
                    item.select = False
        for offset in (2, 1, 0):
            sequence = getattr(live, DOMAINS[offset])
            for item in self.created[offset]:
                if item.is_valid:
                    sequence.remove(item)
            for before, target in zip(origins[offset], self.live[offset]):
                if not before.is_valid and target.is_valid:
                    sequence.remove(target)

        maps = []
        created = ([], [], [])
        for offset, domain in enumerate(DOMAINS):
            original = {item: index for index, item in enumerate(origins[offset]) if item.is_valid}
            mapping = {}
            sequence = getattr(live, domain)
            for item in getattr(work, domain):
                index = original.get(item)
                target = None if index is None else self.live[offset][index]
                if target is None or not target.is_valid:
                    if offset == 0:
                        target = sequence.new(item.co)
                    else:
                        target = sequence.new(tuple(maps[0][v] for v in item.verts))
                    if index is None:
                        created[offset].append(target)
                    else:
                        self.live[offset][index] = target
                copy_values(item, target, pairs[domain])
                if offset == 2:
                    # Existing faces may start at a different loop; match by vertex.
                    loops = {loop.vert: loop for loop in target.loops}
                    for loop in item.loops:
                        copy_values(loop, loops[maps[0][loop.vert]], pairs["loops"])
                mapping[item] = target
            maps.append(mapping)
        self.created = created
        self.selection = tuple([target for item, target in mapping.items() if item.select]
                               for mapping in maps)
        for group in self.selection:
            for item in group:
                item.select_set(True)
        live.select_history.clear()
        live.normal_update()

    def accept(self, work, origins):
        if self.previous is not None:
            self.previous.free()
        self.previous = work
        self.previous_origins = origins

    def free(self):
        self.live_bm = None
        self.source.free()
        if self.previous is not None:
            self.previous.free()
