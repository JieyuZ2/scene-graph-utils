from collections import Counter
from typing import List

import inflect

from .utils import make_and_description, make_one_data
from ..base import BaseGenerator
from ..dataset import BoundBoxes, ObjectDetectionDataset


class ExistsObjectGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "How many {name}?",
			"response": "{count}."
		}
	]
	des_templates = [
		{
			"description": "There {be} {count_name}."
		}
	]
	inflect_engine = inflect.engine()

	def __init__(self, dataset: ObjectDetectionDataset, numeric=True, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.numeric = numeric

	def _generate(self, annotation: BoundBoxes, templates: List) -> List:
		data_list = []

		for name, cnt in Counter(annotation.labels).items():
			be = 'is'
			if self.inflect_engine.singular_noun(name):
				if self.template_mode == 'qa':
					continue
				be = 'are'
				cnt_word = ''
				count_name = name
			else:
				if cnt > 1:
					be = 'are'
					name = self.inflect_engine.plural(name)
				cnt_word = str(cnt) if self.numeric else self.inflect_engine.number_to_words(cnt)
				count_name = cnt_word + ' ' + name

			data_list += make_one_data(
				{
					"name"      : name,
					"count_name": count_name,
					"be"        : be,
					"count"     : cnt_word
				},
				templates=templates,
				rng=self.rng,
				enumerate_templates=True
			)
		return data_list


class MostObjectGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "Among {candidates}, which is the most frequent object?",
			"response": "{name}"
		}
	]
	des_templates = [
		{
			"description": "Among {candidates}, {name} is the most frequent object."
		}
	]
	inflect_engine = inflect.engine()

	def __init__(self, dataset: ObjectDetectionDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n = n

	def _generate(self, annotation: BoundBoxes, templates: List) -> List:
		annotation = annotation.non_including_bboxes()

		labels = [label for label in annotation.labels if not self.inflect_engine.singular_noun(label)]

		cnt_dict = Counter(labels)
		if len(cnt_dict) < 2:
			return []
		name = max(cnt_dict, key=cnt_dict.get)
		max_cnt = cnt_dict[name]

		labels = [i for i in set(labels) if cnt_dict[i] < max_cnt]
		if len(labels) == 0:
			return []
		if len(labels) > self.n - 1:
			candidates = [name] + list(self.rng.choice(labels, self.n - 1))
		else:
			candidates = [name] + labels
		candidates = make_and_description(candidates, self.rng)
		return make_one_data(
			{
				"name"      : name,
				"candidates": candidates
			},
			templates=templates,
			rng=self.rng,
			enumerate_templates=True
		)


class LeastObjectGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "Among {candidates}, which is the least frequent object?",
			"response": "{name}"
		}
	]
	des_templates = [
		{
			"description": "Among {candidates}, {name} is the least frequent object."
		}
	]
	inflect_engine = inflect.engine()

	def __init__(self, dataset: ObjectDetectionDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n = n

	def _generate(self, annotation: BoundBoxes, templates: List) -> List:
		annotation = annotation.non_including_bboxes()

		labels = [label for label in annotation.labels if not self.inflect_engine.singular_noun(label)]

		cnt_dict = Counter(labels)
		if len(cnt_dict) < 2:
			return []
		name = min(cnt_dict, key=cnt_dict.get)
		min_cnt = cnt_dict[name]

		labels = [i for i in set(labels) if cnt_dict[i] > min_cnt]
		if len(labels) == 0:
			return []
		if len(labels) > self.n - 1:
			candidates = [name] + list(self.rng.choice(labels, self.n - 1))
		else:
			candidates = [name] + labels
		candidates = make_and_description(candidates, self.rng)
		return make_one_data(
			{
				"name"      : name,
				"candidates": candidates
			},
			templates=templates,
			rng=self.rng,
			enumerate_templates=True
		)


class LeftMostObjectGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "Among {candidates}, which is on the most left side?",
			"response": "{name}"
		}
	]
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most left side."
		}
	]

	def __init__(self, dataset: ObjectDetectionDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n = n

	def _generate(self, annotation: BoundBoxes, templates: List) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			left_most_candidate_label = ann.labels[min(bboxes_with_idx, key=lambda x: x[0][0])[1]]
			candidates = make_and_description(ann.labels, self.rng)

			data_list += make_one_data(
				{
					"name"      : left_most_candidate_label,
					"candidates": candidates
				},
				templates=templates,
				rng=self.rng,
				enumerate_templates=True
			)

		return data_list


class RightMostObjectGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "Among {candidates}, which is on the most right side?",
			"response": "{name}"
		}
	]
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most right side."
		}
	]

	def __init__(self, dataset: ObjectDetectionDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n = n

	def _generate(self, annotation: BoundBoxes, templates: List) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			right_most_candidate_label = annotation.labels[max(bboxes_with_idx, key=lambda x: x[0][2])[1]]
			candidates = make_and_description(ann.labels, self.rng)

			data_list += make_one_data(
				{
					"name"      : right_most_candidate_label,
					"candidates": candidates
				},
				templates=templates,
				rng=self.rng,
				enumerate_templates=True
			)

		return data_list


class TopMostObjectGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "Among {candidates}, which is on the most top side?",
			"response": "{name}"
		}
	]
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most top side."
		}
	]

	def __init__(self, dataset: ObjectDetectionDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n = n

	def _generate(self, annotation: BoundBoxes, templates: List) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			top_most_candidate_label = annotation.labels[min(bboxes_with_idx, key=lambda x: x[0][1])[1]]
			candidates = make_and_description(ann.labels, self.rng)

			data_list += make_one_data(
				{
					"name"      : top_most_candidate_label,
					"candidates": candidates
				},
				templates=templates,
				rng=self.rng,
				enumerate_templates=True
			)

		return data_list


class BottomMostObjectGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "Among {candidates}, which is on the most bottom side?",
			"response": "{name}"
		}
	]
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most bottom side."
		}
	]

	def __init__(self, dataset: ObjectDetectionDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n = n

	def _generate(self, annotation: BoundBoxes, templates: List) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			bottom_most_candidate_label = annotation.labels[max(bboxes_with_idx, key=lambda x: x[0][3])[1]]
			candidates = make_and_description(ann.labels, self.rng)

			data_list += make_one_data(
				{
					"name"      : bottom_most_candidate_label,
					"candidates": candidates
				},
				templates=templates,
				rng=self.rng,
				enumerate_templates=True
			)

		return data_list


ObjectGeneratorList = [
	ExistsObjectGenerator,
	MostObjectGenerator,
	LeastObjectGenerator,
	LeftMostObjectGenerator,
	RightMostObjectGenerator,
	TopMostObjectGenerator,
	BottomMostObjectGenerator
]
