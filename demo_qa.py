from sg.base import *
from sg.dataset import *
from sg.generators.attribute import *
from sg.generators.object import *
from sg.generators.relation import *
from sg.generators.scene_graph import *

if __name__ == "__main__":

	sg_dataset = SceneGraphDataset("scene-graph-annotations.json")

	img = '2386621.jpg'

	# Objects
	gen = JointGenerator(
		dataset=sg_dataset,
		generators=ObjectGeneratorList,
		template_mode='qa'
	)
	for data in gen.generate():
		if data['data_path'] == img:
			print(data['generator'])
			print(data['prompt'], data['response'], sep='\n')

	# Attributes
	gen = JointGenerator(
		dataset=sg_dataset,
		generators=AttributeGeneratorList,
		template_mode='qa'
	)
	for data in gen.generate():
		if data['data_path'] == img:
			print(data['generator'])
			print(data['prompt'], data['response'], sep='\n')

	# Relations
	gen = JointGenerator(
		dataset=sg_dataset,
		generators=RelationGeneratorList,
		template_mode='qa'
	)
	for data in gen.generate():
		if data['data_path'] == img:
			print(data['generator'])
			print(data['prompt'], data['response'], sep='\n')

	# SG
	gen = SceneGraphQAGenerator(
		dataset=sg_dataset,
		template_mode='qa'
	)
	for data in gen.generate():
		if data['data_path'] == img:
			print(data['generator'])
			print(data['type'], data['prompt'], data['response'], sep='\n')
