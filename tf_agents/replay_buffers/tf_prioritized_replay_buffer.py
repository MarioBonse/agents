# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A batched replay buffer of nests of Tensors which can be sampled uniformly.

- Each add assumes tensors have batch_size as first dimension, and will store
each element of the batch in an offset segment, so that each batch dimension has
its own contiguous memory. Within batch segments, behaves as a circular buffer.

The get_next function returns 'ids' in addition to the data. This is not really
needed for the batched replay buffer, but is returned to be consistent with
the API for a priority replay buffer, which needs the ids to update priorities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.replay_buffers import replay_buffer
from tf_agents.replay_buffers import table
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_sum_tree


DEFAULT_PRIORITY = 100.0         # copied from DeepMind implementation
MAXIMUM_SAMPLING_ATTEMPTS = 100
BufferInfo = collections.namedtuple('BufferInfo',
									['ids', 'probabilities'])

'''
Se non sai come funziona nel dettaglio il PRB chiedi a me che ti faccio un sunto del paper che l'ha presentato.
Un po' di note su come costruirlo:
a) Il PRB avrà dei metodi self.get/set_priority che di fatto non fanno altro che far riferimento all'oggetto
  self.sum_tree.get/set_priority. Vedi implementazione DeepMind in prioritized_replay_memory.py che mi sembra 
  sia abbastanza copia-incollabile. 
b) Il PRB ha due situazioni diverse in cui assegna la priorità alle transizioni (con cui poi estrae):
	1) Quando una transizione è appena aggiunta essa riceve una priorità di default (massima). Nota
		che un massimo per la priority non esiste visto che a regime essa dipende dalla loss (e quindi TD error)
		e quella può essere arbitraria... L'idea è di mettere un valore decisamente più alto di qualsiasi valore
		che la loss possa mai ragionevolmente ottenere.
		Quindi dentro il metodo add_batch (chiamato dal driver) avremmo del codice simile al seguente:
		...
		indices = self.add_transitions(batch)        # forse per questo si usa self.table_fn
		self.set_priority(indices, DEFAULT_PRIORITY)  # DEFAULT_PRIORITY=100 in implementazione DeepMind
		...
	2) Quando una transizione viene rivista in training, la sua priority viene updatata a seconda
		di quale sia la sua loss. Dobbiamo decidere/capire se fare questo dentro la funzione training 
		dell'agente o fuori usando la loss che ci viene ritornata... Io sarei a favore di farla fuori
		in modo da non dover cambiare la struttura della funzione train dell'agent. Questo è certamente
		possibile perchè gli elementi del dataset contengono le informazioni aggiuntive necessarie 
		(id che sono stati samplati)
		Leggi però le mie note sul training in cima (dopo gli import) al file eager_main.py
		Il codice per fare sta cosa sarà una roba del tipo:
		...(esegui training e prendi la loss)
		PRB.update_priority(indices, loss)
		...
		Il metodo update priority passerà la loss a SumTree a occhio
c) Il metodo self.get_priority sembra per il momento essere inutile. O meglio non viene chiamato da nessun altro metodo
	della classe e probabilmente serve solo a fini di debugging/ se per qualche altra ragione vuoi sapere qual è la 
	priority di qualcosa... Il metodo self._get_next (utilizzato per generare il dataset) in realtà sampla chiedendo gli
	indici direttamente all'oggetto sumtree, quindi il replay buffer di per sé non è mai interessato a sapere quali sono
	le priorities.... Quelle sono conosciute dal sum tree e il RB si accontenta di avere i sample estratti con quelle priority
d) Per samplare c'è il comodissimo metodo sample() di sum_tree che ti ritorna un indice basato sulle priority.
  nell'implementazione di DeepMind non fanno altro che chiamare questo metodo batch_size-volte (vedere prioritized_replay_memory.py
  a sample_index_batch()). Penso che questa implementazione sia però un pelo diversa da quella del paper che introdusse PRB.
  Se non sbaglio infatti nel paper originale il range [0, sum_of_priorities] viene diviso in batch_size sotto-intervalli di pari
  grandezza e si sampla a caso da ciascuno. Nell'implementazione DeepMind invece samplano direttamente con weight basato sulla priority...
  Non penso cambi molto e probabilmente chi se ne frega, quelli di DeepMind sapranno quello che fanno no?
		

Open Problems:
a) Se guardi nelle note sopra, i metodi get/set_priority prendono in input degli indici che ti dicono 
	quali sono gli indici nella memoria del PRB (delle transizioni che stai maneggiando). Come conciliare
	questo con PRB.as_dataset() e l'update delle priority? Ovvero se come suggerisco sopra l'update delle 
	priority viene chiamato nella funzione training dell'agent, la domanda è come fa l'agent a ricavare
	gli indici da passare alla funzione set_priority() partendo dal batch di osservazioni che riceve...
	Anche se facessimo l'update delle priority fuori dalla train dell'agent e dentro il for loop sul dataset
	non penso questo cambierebbe nulla... Mi sembra che come unica soluzione si debba passare l'indice della
	transizione insieme alla transizione quando il RB crea il batch con self.get_next()
b) Mi sembra di capire che nell'implementazione DeepMind il PRB consideri un elemento come
	obs, rew, action (e legal_actions e se lo stato è terminale, ma vabbeh), mentre tf-agents
	di default prende anche lo stato successivo. Not sure se questo ci creerà più problemi 
	nell'implementarlo, ma spero/penso di no
'''

"""
Notes on Prioritized Replay Buffer differences compared from "normal" Replay Buffers from tf-agents:
  - _add_batch method, doesn't really support batches (lol), this essentially because I haven't changed
	the SumTree object created by DeepMind. That object isn't written in TF code and it is unclear to me
	whether the calls to it would run when running in graph mode. It is also unclear when exactly TF 2.x *is*
	running in graph mode (I guess it has smth to do with tf.function) so I guess there's quite a bit of 
	uncertainty here... In any case when running eagerly the code should run as expected. The current "running"
	file eager_main.py (in the hanabi repo) definetly runs the Driver eagerly, and I think that the training functions
	are run in graph mode (or in any case they are compiled with tf.function). I'm not sure whether the tf.data.Dataset
	generated by the Replay Buffer (which must use the SumTree to generate it) is run eagerly or not. Haven't yet 
	started to develop that part of code so I guess I'll find out how this works wih the SumTree. 
	All of this to say that you can't run multiple environments and that the _add_batch method (I think) relies on being
	able to evaluate certain tensors.
	- If you wanted to make the Prioritized Replay Buffer batch-compliant also be careful about how indices are managed between
	  batches; i.e. if max_length = 100 and batch_size = 2, then the transition at index 99 in data_table definetly isn't followed
	  by the one at index 100. This is because index 100 actually corresponds to index 0 in the second batch. Looking closely at  
	  Uniform Replay Buffer's "_get_next" code might help to understand better how to deal with this and other fringe cases.
  - The SumTree already implicitly makes sure that memory which hasn't been filled isn't accessed (their priority is 0). Therefore
	the only thing to keep track of when sampling is the case in which the memory has already gone over max_length and has started
	substituing old trajectories. In this case there are two things to do right:
	  1) The trajectory stored at id=max_length-1 should be followed by trajectory at id=0
	  2) (for num_steps=2) The last trajectory added shouldn't be sampled since it doesn't have the required next trajectories.
"""
# FIXME It is unclear whether the SumTree associated with the PRB actually gets saved by checkpointer objects or not... I'd venture
# to say that it doesn't, especially since it isn't even written in TF code... Might raise an error when attempting to save
# (needs to be tested) or might have unexpected behaviour when loading. Do not trust checkpoints basically
# TODO convert all code in sample_ids_batch (and all functions called by it) into TF 2.x compliant code. Note that this includes
# rewriting the entire SumTree data structure to be TF 2.x compliant. Connected to this is the function update_priority because a batch
# of samples might well have twice the same sample and then you would have two different (possibly, I'm not sure) priorities trying to
# be assigned to the same item... That function should probably stay in non-TF 2.x code or (more simply) should enforce eager execution
# to update priorities sequentially and NOT at the same time
@gin.configurable
class TFPrioritizedReplayBuffer(replay_buffer.ReplayBuffer):
	"""A TFUniformReplayBuffer with batched adds and uniform sampling."""

	def __init__(self,
				 data_spec,
				 batch_size,
				 max_length=1000,
				 scope='TFPrioritizedReplayBuffer',
				 device='cpu:*',
				 table_fn=table.Table,
				 dataset_drop_remainder=False,
				 dataset_window_shift=None,
				 stateful_dataset=False):
		"""Creates a TFPrioritizedReplayBuffer.

		The TFPrioritizedReplayBuffer stores episodes in `B == batch_size` blocks of
		size `L == max_length`, with total frame capacity
		`C == L * B`.  Storage looks like:

		```
		block1 ep1 frame1
				   frame2
			   ...
			   ep2 frame1
				   frame2
			   ...
			   <L frames total>
		block2 ep1 frame1
				   frame2
			   ...
			   ep2 frame1
				   frame2
			   ...
			   <L frames total>
		...
		blockB ep1 frame1
				   frame2
			   ...
			   ep2 frame1
				   frame2
			   ...
			   <L frames total>
		```
		Multiple episodes may be stored within a given block, up to `max_length`
		frames total.  In practice, new episodes will overwrite old ones as the
		block rolls over its `max_length`.

		Args:
		  data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing a
			single item that can be stored in this buffer.
		  batch_size: Batch dimension of tensors when adding to buffer.
		  max_length: The maximum number of items that can be stored in a single
			batch segment of the buffer.
		  scope: Scope prefix for variables and ops created by this class.
		  device: A TensorFlow device to place the Variables and ops.
		  table_fn: Function to create tables `table_fn(data_spec, capacity)` that
			can read/write nested tensors.
		  dataset_drop_remainder: If `True`, then when calling
			`as_dataset` with arguments `single_deterministic_pass=True` and
			`sample_batch_size is not None`, the final batch will be dropped if it
			does not contain exactly `sample_batch_size` items.  This is helpful for
			static shape inference as the resulting tensors will always have
			leading dimension `sample_batch_size` instead of `None`.
		  dataset_window_shift: Window shift used when calling `as_dataset` with
			arguments `single_deterministic_pass=True` and `num_steps is not None`.
			This determines how the resulting frames are windowed.  If `None`, then
			there is no overlap created between frames and each frame is seen
			exactly once.  For example, if `max_length=5`, `num_steps=2`,
			`sample_batch_size=None`, and `dataset_window_shift=None`, then the
			datasets returned will have frames `{[0, 1], [2, 3], [4]}`.

			If `dataset_window_shift is not None`, then windows are created with a
			window overlap of `dataset_window_shift` and you will see each frame up
			to `num_steps` times.  For example, if `max_length=5`, `num_steps=2`,
			`sample_batch_size=None`, and `dataset_window_shift=1`, then the
			datasets returned will have windows of shifted repeated frames:
			`{[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]}`.

			For more details, see the documentation of `tf.data.Dataset.window`,
			specifically for the `shift` argument.

			The default behavior is to not overlap frames
			(`dataset_window_shift=None`) but users often want to see all
			combinations of frame sequences, in which case `dataset_window_shift=1`
			is the appropriate value.
		  stateful_dataset: whether the dataset contains stateful ops or not.
		"""
		if batch_size != 1:
			raise RuntimeError("Prioritized RB doesn't support batch_size != 1.\n"
							   "See comments in the code above this class for more info.")
		self._batch_size = batch_size
		self._max_length = max_length
		capacity = self._batch_size * self._max_length
		super(TFPrioritizedReplayBuffer, self).__init__(
			data_spec, capacity, stateful_dataset)

		self._id_spec = tensor_spec.TensorSpec([], dtype=tf.int64, name='id')
		self._capacity_value = np.int64(self._capacity)
		self._batch_offsets = (
			tf.range(self._batch_size, dtype=tf.int64) * self._max_length)
		self._scope = scope
		self._device = device
		self._table_fn = table_fn
		self._dataset_drop_remainder = dataset_drop_remainder
		self._dataset_window_shift = dataset_window_shift
		self.sum_tree = tf_sum_tree.TFSumTree(self._capacity_value)
		with tf.device(self._device), tf.compat.v1.variable_scope(self._scope):
			self._capacity = tf.constant(capacity, dtype=tf.int64)
			self._data_table = table_fn(self._data_spec, self._capacity_value)
			self._id_table = table_fn(self._id_spec, self._capacity_value)
			self._last_id = common.create_variable('last_id', -1)
			self._last_id_cs = tf.CriticalSection(name='last_id')

	def variables(self):
		return (self._data_table.variables() +
				self._id_table.variables() +
				[self._last_id])

	@property
	def device(self):
		return self._device

	@property
	def table_fn(self):
		return self._table_fn

	@property
	def scope(self):
		return self._scope

	# Methods defined in ReplayBuffer base class

	def _num_frames(self):
		num_items_single_batch_segment = self._get_last_id() + 1
		total_frames = num_items_single_batch_segment * self._batch_size
		return tf.minimum(total_frames, self._capacity)

	def _add_batch(self, items):
		"""Adds a batch of items to the replay buffer.

		Args:
		  items: A tensor or list/tuple/nest of tensors representing a batch of
		  items to be added to the replay buffer. Each element of `items` must match
		  the data_spec of this class. Should be shape [batch_size, data_spec, ...]
		Returns:
		  An op that adds `items` to the replay buffer.
		Raises:
		  ValueError: If called more than once.
		"""
		tf.nest.assert_same_structure(items, self._data_spec)

		with tf.device(self._device), tf.name_scope(self._scope):
			id_ = self._increment_last_id()
			write_rows = self._get_rows_for_id(id_)
			default_priorities = tf.ones_like(
				write_rows, dtype=tf.float32)*DEFAULT_PRIORITY
			self.tf_set_priority(write_rows, default_priorities)
			write_id_op = self._id_table.write(write_rows, id_)
			write_data_op = self._data_table.write(write_rows, items)
			return tf.group(write_id_op, write_data_op)

	def _get_next(self,
				  sample_batch_size=None,
				  num_steps=None,
				  time_stacked=True):
		"""Returns an item or batch of items sampled uniformly from the buffer.

		Sample transitions uniformly from replay buffer. When sub-episodes are
		desired, specify num_steps, although note that for the returned items to
		truly be sub-episodes also requires that experience collection be
		single-threaded.

		Args:
		  sample_batch_size: (Optional.) An optional batch_size to specify the
			number of items to return. See get_next() documentation.
		  num_steps: (Optional.)  Optional way to specify that sub-episodes are
			desired. See get_next() documentation.
		  time_stacked: Bool, when true and num_steps > 1 get_next on the buffer
			would return the items stack on the time dimension. The outputs would be
			[B, T, ..] if sample_batch_size is given or [T, ..] otherwise.
		Returns:
		  A 2 tuple, containing:
			- An item, sequence of items, or batch thereof sampled uniformly
			  from the buffer.
			- BufferInfo NamedTuple, containing:
			  - The items' ids.
			  - The sampling probability of each item.
		Raises:
		  ValueError: if num_steps is bigger than the capacity.
		"""
		with tf.device(self._device), tf.name_scope(self._scope):
			with tf.name_scope('get_next'):
				rows_shape = () if sample_batch_size is None else (sample_batch_size,)
				assert_nonempty = tf.debugging.assert_greater(
					self._get_last_id(),
					tf.constant(0, tf.int64),
					message='TFPrioritizedReplayBuffer is empty. Make sure to add items '
					'before sampling the buffer.')
				with tf.control_dependencies([assert_nonempty]):
					ids, probabilities = self.sample_ids_batch(
						rows_shape, num_steps)

				if num_steps is None:
					rows_to_get = tf.math.mod(ids, self._capacity)
					data = self._data_table.read(rows_to_get)
					data_ids = self._id_table.read(rows_to_get)
				else:
					if time_stacked:
						step_range = tf.range(num_steps, dtype=tf.int64)
						if sample_batch_size:
							step_range = tf.reshape(step_range, [1, num_steps])
							step_range = tf.tile(
								step_range, [sample_batch_size, 1])
							ids = tf.tile(tf.expand_dims(
								ids, -1), [1, num_steps])
						else:
							step_range = tf.reshape(step_range, [num_steps])

						rows_to_get = tf.math.mod(
							step_range + ids, self._capacity)
						data = self._data_table.read(rows_to_get)
						data_ids = self._id_table.read(rows_to_get)
					else:
						data = []
						data_ids = []
						for step in range(num_steps):
							steps_to_get = tf.math.mod(
								ids + step, self._capacity)
							items = self._data_table.read(steps_to_get)
							data.append(items)
							data_ids.append(self._id_table.read(steps_to_get))
						data = tuple(data)
						data_ids = tuple(data_ids)

				buffer_info = BufferInfo(ids=data_ids,
										 probabilities=probabilities)
		return data, buffer_info

	@gin.configurable(
		'tf_agents.tf_prioritized_replay_buffer.TFPrioritizedReplayBuffer.as_dataset')
	def as_dataset(self,
				   sample_batch_size=None,
				   num_steps=None,
				   num_parallel_calls=None,
				   single_deterministic_pass=False):
		return super(TFPrioritizedReplayBuffer, self).as_dataset(
			sample_batch_size, num_steps, num_parallel_calls,
			single_deterministic_pass=single_deterministic_pass)

	def _as_dataset(self,
					sample_batch_size=None,
					num_steps=None,
					num_parallel_calls=None):
		"""Creates a dataset that returns entries from the buffer in shuffled order.

		Args:
		  sample_batch_size: (Optional.) An optional batch_size to specify the
			number of items to return. See as_dataset() documentation.
		  num_steps: (Optional.)  Optional way to specify that sub-episodes are
			desired. See as_dataset() documentation.
		  num_parallel_calls: (Optional.) Number elements to process in parallel.
			See as_dataset() documentation.
		Returns:
		  A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
			- An item or sequence of items or batch thereof
			- Auxiliary info for the items (i.e. ids, probs).
		"""
		def get_next(_):
			return self.get_next(sample_batch_size, num_steps, time_stacked=True)

		dataset = tf.data.experimental.Counter().map(
			get_next, num_parallel_calls=num_parallel_calls)
		return dataset

	def _single_deterministic_pass_dataset(self,
										   sample_batch_size=None,
										   num_steps=None,
										   num_parallel_calls=None):
		"""Creates a dataset that returns entries from the buffer in fixed order.

		Args:
		  sample_batch_size: (Optional.) An optional batch_size to specify the
			number of items to return. See as_dataset() documentation.
		  num_steps: (Optional.)  Optional way to specify that sub-episodes are
			desired. See as_dataset() documentation.
		  num_parallel_calls: (Optional.) Number elements to process in parallel.
			See as_dataset() documentation.
		Returns:
		  A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
			- An item or sequence of items or batch thereof
			- Auxiliary info for the items (i.e. ids, probs).

		Raises:
		  ValueError: If `dataset_drop_remainder` is set, and
			`sample_batch_size > self.batch_size`.  In this case all data will
			be dropped.
		"""
		raise RuntimeError("This method is just a copy of what was in TFUniformReplayBuffer "
						   "and its implementation hasn't been updated for TFPrioritizedReplayBuffer.")
		static_size = tf.get_static_value(sample_batch_size)
		static_num_steps = tf.get_static_value(num_steps)
		static_self_batch_size = tf.get_static_value(self._batch_size)
		static_self_max_length = tf.get_static_value(self._max_length)
		if (self._dataset_drop_remainder
			and static_size is not None
			and static_self_batch_size is not None
				and static_size > static_self_batch_size):
			raise ValueError(
				'sample_batch_size ({}) > self.batch_size ({}) and '
				'dataset_drop_remainder is True.  In '
				'this case, ALL data will be dropped by the deterministic dataset.'
				.format(static_size, static_self_batch_size))
		if (self._dataset_drop_remainder
			and static_num_steps is not None
			and static_self_max_length is not None
				and static_num_steps > static_self_max_length):
			raise ValueError(
				'num_steps_size ({}) > self.max_length ({}) and '
				'dataset_drop_remainder is True.  In '
				'this case, ALL data will be dropped by the deterministic dataset.'
				.format(static_num_steps, static_self_max_length))

		def get_row_ids(_):
			"""Passed to Dataset.range(self._batch_size).flat_map(.), gets row ids."""
			with tf.device(self._device), tf.name_scope(self._scope):
				with tf.name_scope('single_deterministic_pass_dataset'):
					# Here we pass num_steps=None because _valid_range_ids uses
					# num_steps to determine a hard stop when sampling num_steps starting
					# from the returned indices.  But in our case, we want all the indices
					# and we'll use TF dataset's window() mechanism to get
					# num_steps-length blocks.  The window mechanism handles this stuff
					# for us.
					min_frame_offset, max_frame_offset = _valid_range_ids(
						self._get_last_id(), self._max_length, num_steps=None)
					tf.compat.v1.assert_less(
						min_frame_offset,
						max_frame_offset,
						message='TFUniformReplayBuffer is empty. Make sure to add items '
						'before asking the buffer for data.')

					min_max_frame_range = tf.range(
						min_frame_offset, max_frame_offset)

					drop_remainder = self._dataset_drop_remainder
					window_shift = self._dataset_window_shift

					def group_windows(ds_):
						return ds_.batch(num_steps, drop_remainder=drop_remainder)

					if sample_batch_size is None:
						def row_ids(b):
							# Create a vector of shape [num_frames] and slice it along each
							# frame.
							ids = tf.data.Dataset.from_tensor_slices(
								b * self._max_length + min_max_frame_range)
							if num_steps is not None:
								ids = (ids.window(num_steps, shift=window_shift)
									   .flat_map(group_windows))
							return ids
						return tf.data.Dataset.range(self._batch_size).flat_map(row_ids)
					else:
						def batched_row_ids(batch):
							# Create a matrix of indices shaped [num_frames, batch_size]
							# and slice it along each frame row to get groups of batches
							# for frame 0, frame 1, ...
							return tf.data.Dataset.from_tensor_slices(
								(min_max_frame_range[:, tf.newaxis]
								 + batch * self._max_length))

						indices_ds = (
							tf.data.Dataset.range(self._batch_size)
							.batch(sample_batch_size, drop_remainder=drop_remainder)
							.flat_map(batched_row_ids))

						if num_steps is not None:
							# We have sequences of num_frames rows shaped [sample_batch_size].
							# Window and group these to rows of shape
							# [num_steps, sample_batch_size], then
							# transpose them to get index tensors of shape
							# [sample_batch_size, num_steps].
							indices_ds = (indices_ds.window(num_steps, shift=window_shift)
										  .flat_map(group_windows)
										  .map(tf.transpose))

						return indices_ds

		# Get our indices as a dataset; each time we reinitialize the iterator we
		# update our min/max id bounds from the state of the replay buffer.
		ds = tf.data.Dataset.range(1).flat_map(get_row_ids)

		def get_data(id_):
			with tf.device(self._device), tf.name_scope(self._scope):
				with tf.name_scope('single_deterministic_pass_dataset'):
					data = self._data_table.read(id_ % self._capacity)
			buffer_info = BufferInfo(ids=id_, probabilities=())
			return (data, buffer_info)

		# Deterministic even though num_parallel_calls > 1.  Operations are
		# run in parallel but then the results are returned in original stream
		# order.
		ds = ds.map(get_data, num_parallel_calls=num_parallel_calls)

		return ds

	def _gather_all(self):
		"""Returns all the items in buffer, shape [batch_size, timestep, ...].

		Returns:
		  All the items currently in the buffer.
		"""
		with tf.device(self._device), tf.name_scope(self._scope):
			with tf.name_scope('gather_all'):
				# Make ids, repeated over batch_size. Shape [batch_size, num_ids, ...].
				min_val, max_val = _valid_range_ids(
					self._get_last_id(), self._max_length)
				ids = tf.range(min_val, max_val)
				ids = tf.stack([ids] * self._batch_size)
				rows = tf.math.mod(ids, self._max_length)

				# Make batch_offsets, shape [batch_size, 1], then add to rows.
				batch_offsets = tf.expand_dims(
					tf.range(self._batch_size, dtype=tf.int64) *
					self._max_length,
					1)
				rows += batch_offsets

				# Expected shape is [batch_size, max_length, ...].
				data = self._data_table.read(rows)
		return data

	def _clear(self, clear_all_variables=False):
		"""Return op that resets the contents of replay buffer.

		Args:
		  clear_all_variables: boolean indicating if all variables should be
			cleared. By default, table contents will be unlinked from
			replay buffer, but values are unmodified for efficiency. Set
			`clear_all_variables=True` to reset all variables including Table
			contents.

		Returns:
		  op that clears or unlinks the replay buffer contents.
		"""
		raise RuntimeError("This method is just a copy of what was in TFUniformReplayBuffer "
						   "and its implementation hasn't been updated for TFPrioritizedReplayBuffer."
						   " Specifically the SumTree object isn't resetted.")
		table_vars = self._data_table.variables() + self._id_table.variables()

		def _init_vars():
			assignments = [self._last_id.assign(-1)]
			if clear_all_variables:
				assignments += [v.assign(tf.zeros_like(v)) for v in table_vars]
			return tf.group(*assignments, name='clear')
		return self._last_id_cs.execute(_init_vars)

	#  Helper functions.
	def _increment_last_id(self, increment=1):
		"""Increments the last_id in a thread safe manner.

		Args:
		  increment: amount to increment last_id by.
		Returns:
		  An op that increments the last_id.
		"""
		def _assign_add():
			return self._last_id.assign_add(increment).value()
		return self._last_id_cs.execute(_assign_add)

	def _get_last_id(self):

		def last_id():
			return self._last_id.value()

		return self._last_id_cs.execute(last_id)

	def _get_rows_for_id(self, id_):
		"""Make a batch_size length list of tensors, with row ids for write."""
		id_mod = tf.math.mod(id_, self._max_length)
		rows = self._batch_offsets + id_mod
		return rows

	# Copied from DeepMind's implementation
	def tf_set_priority(self, indices, priorities):
		"""
		Sets the priorities for the given indices.

		Args:
		  indices: tensor of indices (int64), size k.
		  priorities: tensor of priorities (float32), size k.

		Returns:
		   A TF op setting the priorities according to Prioritized Experience
		   Replay.
		"""
		return self.set_priority(indices, priorities)

	# Copied from DeepMind's implementation (with minor adjustments)
	def set_priority(self, indices, priorities):
		"""Sets the priority of the given elements according to Schaul et al.

		Args:
		  indices: tensor of indices (int64), size k.
		  priorities: tensor of priorities (float32), size k.

		Note that the indices are most likely coming from ids in self._id_table.
		As such thet are not (mod capacity) and must be shifted to the correct index
		"""
		indices = tf.math.mod(indices, self._capacity)

		for i, memory_index in enumerate(indices):
			self.sum_tree.set(memory_index, priorities[i])

	# Copied from DeepMind's implementation (with minor adjustments)
	def get_priority(self, indices, sample_batch_size=None):
		"""Fetches the priorities correspond to a batch of memory indices.

		For any memory location not yet used, the corresponding priority is 0.

		Args:
		  indices: `np.array` of indices in range [0, replay_capacity).
		  sample_batch_size: int, requested number of items.
		Returns:
		  The corresponding priorities.
		"""
		if sample_batch_size is None:
			sample_batch_size = 1

		priority_batch = np.empty((sample_batch_size), dtype=np.float32)

		assert indices.dtype == np.int32, ('Indices must be integers, '
										   'given: {}'.format(indices.dtype))
		for i, memory_index in enumerate(indices):
			priority_batch[i] = self.sum_tree.get(memory_index)

		return priority_batch

	# Copied from DeepMind's implementation (with adjustments)
	def sample_ids_batch(self, sample_batch_size=(), num_steps=None):
		"""Returns a batch of valid indices.

		Args:
		  sample_batch_size: (Optional.) An optional batch_size to specify the
			number of items to return.
		  num_steps: (Optional.)  Optional way to specify that sub-episodes are
			desired. See get_next() documentation.

		Returns:
		  Tensors of shape (sample_batch_size,) containing valid indices and
		  corresponding sampling probabilities.
		"""

		def loop_cond(sampling_attempts_left, is_valid_flag, *_):
			return tf.math.logical_and(
				tf.math.greater(sampling_attempts_left, 0),
				tf.math.logical_not(is_valid_flag))
		
		def loop_body(sampling_attempts_left, is_valid_flag, indeces, probabilities):
			indeces, probabilities = self.sum_tree.sample(shape=sample_batch_size)
			sampling_attempts_left -= 1
			is_valid_flag = self.is_valid_transition(indeces, num_steps)
			return [sampling_attempts_left, is_valid_flag, indeces, probabilities]
		
		sampling_attempts_left = MAXIMUM_SAMPLING_ATTEMPTS

		#while sampling_attempts_left > 0:
		#	indeces, probabilities = self.sum_tree.sample(shape=sample_batch_size)

		#	if self.is_valid_transition(indeces, num_steps):
		#		break
		#	else:
		#		sampling_attempts_left -= 1


		indeces, probabilities = self.sum_tree.sample(shape=sample_batch_size)
		is_valid_flag = False
		[sampling_attemps_left, is_valid_flag, indeces, probabilities] = tf.nest.map_structure(
			tf.stop_gradient, tf.while_loop(
				loop_cond, loop_body, [sampling_attempts_left, is_valid_flag, indeces, probabilities]))


		if sampling_attemps_left == 0:
			raise RuntimeError("Why did it fail so sample so much?\n"
							   "Sampling attemps: {}"
							   "Batch size to sample: {}".format(MAXIMUM_SAMPLING_ATTEMPTS,
																 sample_batch_size))
		
		return indeces, probabilities

	# Copied from DeepMind's implementation (with heavy adjustments)
	def is_valid_transition(self, indeces, num_steps=None):
		"""Checks if the index contains a valid trajoctory.

		The index range needs to be valid

		Args:
		  indeces: Tensor of indeces to the trajoctories in self._data_table. Note that following trajectories
		  num_steps: (Optional.)  Optional way to specify that sub-episodes are
			desired. See get_next() documentation.

		Returns:
		  boolean Tensor, True if all trajoctories identified by indeces are valid

		"""
		last_id_added = tf.math.mod(self._get_last_id(), self._max_length)
		if num_steps is None:
			num_steps = tf.constant(1, tf.int64)
		
		# Range checks
		are_out_of_range = tf.math.logical_or(tf.math.less(indeces, 0),
											  tf.math.greater_equal(indeces, self._max_length))
		# if index < 0 or index >= self._max_length:
		#  raise RuntimeError("Why did this case occur? SumTree isn't supposed to return this index: {}".format(index))
		#  return False
		check_op = tf.debugging.Assert(
			tf.math.reduce_any(are_out_of_range), ["SumTree isn't supposed to return this index", indeces])

		# The trajectory and the following steps must be smaller than last_id_added
		lower_bound = tf.cast(last_id_added - num_steps + 1, tf.int64)
		upper_bound = tf.cast(last_id_added + 1, tf.int64)

		are_invalid = tf.math.logical_and(tf.math.less(lower_bound, indeces),
										tf.math.less(indeces, upper_bound))

		# if last_id_added - num_steps + 1 < index < last_id_added + 1:
		#  return False

		return tf.math.reduce_any(are_invalid)
