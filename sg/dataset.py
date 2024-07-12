from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import combinations
import networkx as nx
import numpy as np

from .base import BaseDataset


def boxOverlap(box1, box2):
	# get corners
	tl1 = (box1[0], box1[1])
	br1 = (box1[2], box1[3])
	tl2 = (box2[0], box2[1])
	br2 = (box2[2], box2[3])

	# separating axis theorem
	# left/right
	if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
		return False

	# top/down
	if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
		return False

	# overlap
	return True


def boxInclude(box1, box2):
	# get corners
	tl1 = (box1[0], box1[1])
	br1 = (box1[2], box1[3])
	tl2 = (box2[0], box2[1])
	br2 = (box2[2], box2[3])

	# separating axis theorem
	# left/right
	if tl1[0] <= tl2[0] and tl1[1] <= tl2[1] and br1[0] >= br2[0] and br1[1] >= br2[1]:
		return True

	return False

@dataclass
class AnnotationList:
	def subset(self, indices):
		data = self.__dict__.copy()
		for key, value in data.items():
			if key == 'relations':
				relations = []
				for head, relation, target in value:
					if head in indices and target in indices:
						relations.append((indices.index(head), relation, indices.index(target)))
				data[key] = relations
			else:
				if isinstance(value, list) or isinstance(value, np.ndarray):
					data[key] = [value[i] for i in indices]
				else:
					data[key] = value
		return self.__class__(**data)

	def small_bboxes(self, ratio=0.5, height=None, width=None):
		if height is None:
			height = max([box[3] for box in self.bboxes])
		if width is None:
			width = max([box[2] for box in self.bboxes])
		area = height * width
		indices = [i for i, box in enumerate(self.bboxes) if ((box[2] - box[0]) * (box[3] - box[1])) < area * ratio]
		return self.subset(indices)

	def non_including_bboxes(self):
		assert hasattr(self, 'bboxes'), "No bboxes."

		non_included = []
		for i, box1 in enumerate(self.bboxes):
			include = False
			for j, box2 in enumerate(self.bboxes):
				if i != j and boxInclude(box1, box2):
					include = True
					break
			if not include:
				non_included.append(i)

		return self.subset(non_included)

	def non_overlapping_bboxes(self):
		assert hasattr(self, 'bboxes'), "No bboxes."

		non_included = []
		for i, box1 in enumerate(self.bboxes):
			include = False
			for j, box2 in enumerate(self.bboxes):
				if i != j and boxInclude(box1, box2):
					include = True
					break
			if not include:
				non_included.append(i)

		overlapped, non_overlapped = set(), []
		for i, box1 in enumerate(self.bboxes):
			if i in non_included and i not in overlapped:
				overlap = False
				for j, box2 in enumerate(self.bboxes):
					if j in non_included and i != j and boxOverlap(box1, box2):
						overlap = True
						overlapped.add(i)
						overlapped.add(j)
						break
				if not overlap:
					non_overlapped.append(i)

		return self.subset(non_overlapped)

	def n_non_overlapping_bboxes(self, n):
		assert hasattr(self, 'bboxes'), "No bboxes."

		N = len(self.bboxes)
		adj_matrix = np.zeros((N, N))
		for i in range(N):
			box1 = self.bboxes[i]
			for j in range(i + 1, N):
				box2 = self.bboxes[j]
				if boxOverlap(box1, box2):
					adj_matrix[i, j] = 1
					adj_matrix[j, i] = 1

		object_combinations = combinations(range(N), n)
		non_overlapped_n_set = []
		for combination in object_combinations:
			combination = list(combination)
			adj = adj_matrix[combination][:, combination]
			if not np.any(adj):
				non_overlapped_n_set.append(combination)

		return non_overlapped_n_set


	def attributed_bboxes(self):
		assert hasattr(self, 'bboxes'), "No bboxes."
		assert hasattr(self, 'attributes'), "No attributes."

		attributed = [i for i, attr in enumerate(self.attributes) if len(attr) > 0]
		return self.subset(attributed)

@dataclass
class BoundBoxes(AnnotationList):
	bboxes: List[List[int]]
	labels: List[str]
	scores: Optional[List[float]]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, item):
		return self.bboxes[item], self.labels[item], self.scores[item]


class ObjectDetectionDataset(BaseDataset):

	def _load(self):
		self.annotations = [
			BoundBoxes(scores=label.get("scores", None), bboxes=label['bboxes'], labels=label['labels'])
			for label in self.annotations.copy()
		]


@dataclass
class Attributes(AnnotationList):
	bboxes: List[List[int]]
	labels: List[str]
	attributes: List[List[str]]
	scores: Optional[List[float]]

	def __len__(self):
		return len(self.labels)


class AttributeDataset(BaseDataset):
	def _load(self):
		self.annotations = [
			Attributes(scores=label.get("scores", None), bboxes=label['bboxes'], labels=label['labels'],
					   attributes=label['attributes'])
			for label in self.annotations.copy()
		]


@dataclass
class Relations(AnnotationList):
	"""
	relations: [(0, relation string, 1)]
	0 and 1 are id for bboxes and labels
	"""
	relations: List[Tuple[int, str, int]]
	bboxes: List[List[int]]
	labels: List[str]

	def __len__(self):
		return len(self.labels)


class RelationDataset(BaseDataset):
	def _load(self):
		self.annotations = [
			Relations(relations=label['relations'], bboxes=label['bboxes'], labels=label['labels'])
			for label in self.annotations.copy()
		]


@dataclass
class SceneGraph(AnnotationList):
	"""
	Annotation: Tuple[head, target, relationship]
	For: Scene Graph
	"""
	bboxes: List[List[int]]
	labels: List[str]
	attributes: List[List[str]]
	relations: List[Tuple[int, str, int]]
	scores: Optional[List[float]]

	def __len__(self):
		return len(self.labels)

	@property
	def graph(self):
		if not hasattr(self, 'graph_'):
			self.graph_ = self._create_graph()
		return self.graph_

	def _create_graph(self):
		scene_graph = nx.MultiDiGraph()
		for i, label in enumerate(self.labels):
			scene_graph.add_node(i, value=label, attributes=self.attributes[i])
		for head, relation, target in self.relations:
			scene_graph.add_edge(head, target, value=relation)
		return scene_graph

	def single_edge_scene_graph(self, rng):
		uv_to_relations = {}
		for head, relation, target in self.relations:
			if head < target:
				head_to_target = True
			else:
				head, target = target, head
				head_to_target = False
			if (head, target) not in uv_to_relations:
				uv_to_relations[(head, target)] = []
			uv_to_relations[(head, target)].append((relation, head_to_target))
		relations = []
		for (head, target), rels in uv_to_relations.items():
			if len(rels) == 1:
				selected_rel, head_to_target = rels[0]
			else:
				selected_rel, head_to_target = rng.choice(rels)
			if head_to_target:
				relations.append((head, selected_rel, target))
			else:
				relations.append((target, selected_rel, head))
		return SceneGraph(bboxes=self.bboxes, labels=self.labels, attributes=self.attributes, relations=relations, scores=self.scores)

	def decompose(self) -> List:
		"""
		Decompose scene graph to multiple disconnected subgraphs.
		"""
		subgraphs = []
		G = self.graph
		connected_nodes = nx.connected_components(G.to_undirected())
		for ids in sorted(connected_nodes, key=len):
			ids = list(ids)
			bboxes = [self.bboxes[i] for i in ids]
			labels = [self.labels[i] for i in ids]
			attributes = [self.attributes[i] for i in ids]
			relations = []
			for head, relation, target in self.relations:
				if head in ids and target in ids:
					relations.append((ids.index(head), relation, ids.index(target)))
			if self.scores is not None:
				scores = [self.scores[i] for i in ids]
			else:
				scores = None
			graph = SceneGraph(bboxes=bboxes, labels=labels, attributes=attributes, relations=relations, scores=scores)
			subgraphs.append(graph)
		return subgraphs

	def draw(self):
		import matplotlib.pyplot as plt

		def bezier_curve(P0, P1, P2, t):
			return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

		G = self.graph

		# Get node and edge labels
		node_labels = nx.get_node_attributes(G, 'label')
		edge_labels = {(u, v, key): data['label'] for u, v, key, data in G.edges(keys=True, data=True)}

		# Draw the graph
		pos = nx.spring_layout(G, k=3)  # Increase the value of k to spread out nodes
		plt.figure()
		# plt.figure(figsize=(8, 6))
		nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')

		# Draw curved edges with labels
		for (u, v, key), label in edge_labels.items():
			rad = 0.3 * (key - 1)  # Adjust the radius for each edge to avoid overlap
			P0, P2 = np.array(pos[u]), np.array(pos[v])
			ctrl_point = (P0 + P2) / 2 + rad * np.array([P2[1] - P0[1], P0[0] - P2[0]])

			# Compute points on the Bezier curve
			curve = np.array([bezier_curve(P0, ctrl_point, P2, t) for t in np.linspace(0, 1, 100)])
			plt.plot(curve[:, 0], curve[:, 1], 'k-')

			# Calculate the midpoint of the Bezier curve for the label
			mid_point = bezier_curve(P0, ctrl_point, P2, 0.5)
			plt.text(mid_point[0], mid_point[1], label, fontsize=10, color='red')

		plt.show()


class SceneGraphDataset(BaseDataset):

	def _load(self):
		self.annotations = [
			SceneGraph(attributes=label['attributes'], relations=label['relations'],
					   bboxes=label['bboxes'], labels=label['labels'], scores=label.get("scores", None))
			for label in self.annotations.copy()
		]
