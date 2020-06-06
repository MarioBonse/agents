# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A sum tree data structure.

Used for prioritized experience replay. See prioritized_replay_buffer.py
and Schaul et al. (2015).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.replay_buffers import table
import tensorflow as tf

import numpy as np


class TFSumTree(tf.Module):
	"""A sum tree data structure for storing replay priorities.

	A sum tree is a complete binary tree whose leaves contain values called
	priorities. Internal nodes maintain the sum of the priorities of all leaf
	nodes in their subtree.

	For capacity = 4, the tree may look like this:

							 +---+
							 |2.5|
							 +-+-+
								   |
				   +-------+--------+
				   |                |
			 +-+-+            +-+-+
			 |1.5|            |1.0|
			 +-+-+            +-+-+
				   |                |
		  +----+----+      +----+----+
		  |         |      |         |
	+-+-+     +-+-+  +-+-+     +-+-+
	|0.5|     |1.0|  |0.5|     |0.5|
	+---+     +---+  +---+     +---+

	This is stored in a single table.Table object. You can think of it as a single 
	array that contains eacv level concatenated to the previous one. Taking the example
	above the array would look like:
	[2.5, 1.5, 1.0, 0.5, 1.0, 0.5, 0.5]
	"""

	def __init__(self,
				 capacity,
				 name='TFSumTree'):
		"""Creates the sum tree data structure for the given replay capacity.

		Args:
		  capacity: int, the maximum number of elements that can be stored in this
				data structure.

		Raises:
		  ValueError: If requested capacity is not positive.
		"""
		super(TFSumTree, self).__init__(name=name)
		assert isinstance(capacity, (int, np.int64, np.int32))
		if capacity <= 0:
			raise ValueError('Sum tree capacity should be positive. Got: {}'.
							 format(capacity))

		self._tree_depth = int(np.ceil(np.log2(capacity))) + 1

		with self.name_scope:
			tensor_spec = tf.TensorSpec((), tf.float32)
			self._levels_offsets = tf.math.pow(
				2, tf.range(0, self._tree_depth, dtype=tf.int64)) - 1
			total_length = tf.math.reduce_sum(
				2**tf.range(0, self._tree_depth - 1)) + capacity
			self._table = table.Table(tensor_spec, total_length)
			self.max_recorded_priority = tf.constant(1.0, dtype=tf.float32)

	@tf.Module.with_name_scope
	def _total_priority(self):
		"""Returns the sum of all priorities stored in this sum tree.

		Returns:
		  float, sum of priorities stored in this sum tree.
		"""
		return self._table.read(0)

	@tf.function
	@tf.Module.with_name_scope
	def sample(self, shape=()):
		"""Samples an element from the sum tree.

		Each element has probability p_i / sum_j p_j of being picked, where p_i is
		the (positive) value associated with node i (possibly unnormalized).

		Args:
		  query_value: float in [0, 1], used as the random value to select a
		  sample. If None, will select one randomly in [0, 1).

		Returns:
		  int, a random element from the sum tree.

		Raises:
		  Exception: If the sum tree is empty (i.e. its node values sum to 0), or if
				the supplied query_value is larger than the total sum.
		"""
		print('SumTree.sample function is being executed in Pythonically.'
          '\nThis print should occur only once per script execution and possibly per process running the code.'
          '\nIf you see this print "a lot" call for help.')
		tf.debugging.assert_greater(self._total_priority(), 0.0,
									message='Cannot sample from an empty sum tree.')

		def choose_child(inputs):
			if inputs[1] < inputs[2]:
				return inputs[0]
			else:
				return inputs[0] + 1
		

		# Sample a value in range [0, R), where R is the value stored at the root.
		query_values = tf.random.uniform(shape=shape) * self._total_priority()

		# Now traverse the sum tree.
		node_indeces = tf.zeros(shape=shape, dtype=tf.int64)

		for i in range(1, self._tree_depth + 1):
			# Compute children of previous depth's node.
			level_offset = self._levels_offsets[i]
			left_children = node_indeces * 2
			left_sums = self._table.read(level_offset + left_children)
			node_indeces = tf.map_fn(choose_child, (node_indeces, query_values, left_sums), dtype=tf.int32)

		probabilities = self._table.read(node_indeces) / self._total_priority()

		return node_indeces, probabilities

	@tf.Module.with_name_scope
	def stratified_sample(self, batch_size):
		"""Performs stratified sampling using the sum tree.

		Let R be the value at the root (total value of sum tree). This method will
		divide [0, R) into batch_size segments, pick a random number from each of
		those segments, and use that random number to sample from the sum_tree. This
		is as specified in Schaul et al. (2015).

		Args:
		  batch_size: int, the number of strata to use.
		Returns:
		  list of batch_size elements sampled from the sum tree.

		Raises:
		  Exception: If the sum tree is empty (i.e. its node values sum to 0).
		"""
		raise NotImplementedError
		if self._total_priority() == 0.0:
			raise Exception('Cannot sample from an empty sum tree.')

		bounds = np.linspace(0., 1., batch_size + 1)
		assert len(bounds) == batch_size + 1
		segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
		query_values = [tf.random.uniform(x[0], x[1]) for x in segments]
		return [self.sample(query_value=x) for x in query_values]

	@tf.Module.with_name_scope
	def get(self, node_index):
		"""Returns the value of the leaf node corresponding to the index.

		Args:
		  node_index: The index of the leaf node.
		Returns:
		  The value of the leaf node.
		"""
		return self._table.read(node_index)

	@tf.Module.with_name_scope
	def set(self, node_index, value):
		"""Sets the value of a leaf node and updates internal nodes accordingly.

		This operation takes O(log(capacity)).
		Args:
		  node_index: Tensor, dtype=int64 the index of the leaf node to be updated.
		  value: Tensor, dtype=float32, the value which we assign to the node. This value must be
				nonnegative. Setting value = 0 will cause the element to never be
				sampled.

		Raises:
		  ValueError: If the given value is negative.
		"""
		tf.debugging.assert_greater(
			value,
			tf.constant(0, tf.float32),
			message='Sum tree values should be nonnegative. Got {}'.format(value))

		self.max_recorded_priority = tf.math.maximum(
			value, self.max_recorded_priority)
		delta_value = value - self._table.read(node_index)

		# Updating the priority value of the given leaf node and also of all its parent nodes
		# node that the first parent of node_index is at index (ceil(node_index/2) - 1) in the level
		# above, the second parend is at index (ceil(node_index/4) - 1). The variable indices computed
		# below corresponds to the index of every node to be updated in their respective level.
		# Example:
		# If I wanted to update the 5th element in a tree of depth 4 (which includes root level, so the
		# number of leaves is 2**(4-1)) then indices = [4, 2, 1, 0]
		divs = tf.math.pow(2, tf.range(0, self._tree_depth, dtype=tf.int64))
		indices = tf.cast(tf.math.ceil(node_index / divs), tf.int64) - 1
		rows = self._levels_offsets + indices

		self._table.write(rows, tf.repeat(delta_value, self._tree_depth))

