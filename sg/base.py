import json
from abc import abstractmethod
from typing import Dict, List, Callable

import numpy as np


class BaseDataset:
	data_paths: List[str]
	annotations: List
	sources: List[str]

	def __init__(self, annotation_path):
		self.annotation_path = annotation_path

		with open(annotation_path, 'r') as f:
			annotation = json.load(f)
			self.data_paths = []
			self.annotations = []
			self.sources = []
			for ann in annotation:
				self.data_paths.append(ann['data_path'])
				self.annotations.append(ann['annotation'])
				self.sources.append(ann.get('source', None))

		self._load()

	@abstractmethod
	def _load(self):
		"""
		(Abstract method) load"
		"""

	def __getitem__(self, idx):
		return self.data_paths[idx], self.annotations[idx], self.sources[idx]

	def __len__(self):
		return len(self.data_paths)


class BaseGenerator:
	dataset: BaseDataset
	qa_templates = []
	des_templates = []

	def __init__(self, dataset: BaseDataset, template_mode: str = 'description', seed: int = 42):
		"""
		templates = 'qa' or 'description'
		"""
		self.dataset = dataset
		if template_mode == 'qa':
			self.templates = self.qa_templates
			self.template_mode = 'qa'
		elif template_mode == 'description':
			self.templates = self.des_templates
			self.template_mode = 'description'
		else:
			raise ValueError(f"Invalid template mode: {template_mode}")
		self.rng = np.random.default_rng(seed)

	def generate(self) -> List:
		if len(self.templates) == 0:
			return []
		data_list = []
		for data_path, annotation, source in (
				zip(self.dataset.data_paths, self.dataset.annotations, self.dataset.sources)
		):
			if len(annotation.labels) > 0:
				for data in self._generate(annotation, self.templates):
					data['data_path'] = data_path
					data['generator'] = self.__class__.__name__
					data_list.append(data)
		return data_list

	@abstractmethod
	def _generate(self, annotation, templates: List) -> List[Dict]:
		"""
		Abstract method
		"""


class JointGenerator(BaseGenerator):
	def __init__(self, dataset: BaseDataset, generators: List[Callable], **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.generators = [
			generator(dataset=dataset, **kwargs) for generator in generators
		]

	def generate(self) -> List:
		data_list = []
		for data_path, annotation, source in (
				zip(self.dataset.data_paths, self.dataset.annotations, self.dataset.sources)
		):
			for generator in self.generators:
				if len(annotation.labels) > 0:
					for data in generator._generate(annotation, generator.templates):
						data['data_path'] = data_path
						data['generator'] = generator.__class__.__name__
						data_list.append(data)
		return data_list

	def _generate(self, annotation, templates: List) -> List[Dict]:
		pass
