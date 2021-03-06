<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.batched_py_environment.BatchedPyEnvironment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="batched"/>
<meta itemprop="property" content="envs"/>
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="close"/>
<meta itemprop="property" content="current_time_step"/>
<meta itemprop="property" content="observation_spec"/>
<meta itemprop="property" content="render"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="time_step_spec"/>
</div>

# tf_agents.environments.batched_py_environment.BatchedPyEnvironment

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/batched_py_environment.py">View
source</a>

## Class `BatchedPyEnvironment`

Batch together multiple py environments and act as a single batch.

Inherits From: [`PyEnvironment`](../../../tf_agents/environments/py_environment/PyEnvironment.md)

<!-- Placeholder for "Used in" -->

The environments should only access shared python variables using
shared mutex locks (from the threading module).

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Batch together multiple (non-batched) py environments.

The environments can be different but must use the same action and
observation specs.

#### Args:

* <b>`envs`</b>: List python environments (must be non-batched).


#### Raises:

*   <b>`ValueError`</b>: If envs is not a list or tuple, or is zero length, or
    if one of the envs is already batched.
*   <b>`ValueError`</b>: If the action or observation specs don't match.

## Properties

<h3 id="batch_size"><code>batch_size</code></h3>

<h3 id="batched"><code>batched</code></h3>

<h3 id="envs"><code>envs</code></h3>

## Methods

<h3 id="__enter__"><code>__enter__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/py_environment.py">View
source</a>

``` python
__enter__()
```

Allows the environment to be used in a with-statement context.

<h3 id="__exit__"><code>__exit__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/py_environment.py">View
source</a>

``` python
__exit__(
    unused_exception_type,
    unused_exc_value,
    unused_traceback
)
```

Allows the environment to be used in a with-statement context.

<h3 id="action_spec"><code>action_spec</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/batched_py_environment.py">View
source</a>

``` python
action_spec()
```

<h3 id="close"><code>close</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/batched_py_environment.py">View
source</a>

``` python
close()
```

Send close messages to the external process and join them.

<h3 id="current_time_step"><code>current_time_step</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/py_environment.py">View
source</a>

``` python
current_time_step()
```

Returns the current timestep.

<h3 id="observation_spec"><code>observation_spec</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/batched_py_environment.py">View
source</a>

``` python
observation_spec()
```

<h3 id="render"><code>render</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/py_environment.py">View
source</a>

``` python
render(mode='rgb_array')
```

Renders the environment.

#### Args:

*   <b>`mode`</b>: One of ['rgb_array', 'human']. Renders to an numpy array, or
    brings up a window where the environment can be visualized.

#### Returns:

An ndarray of shape [width, height, 3] denoting an RGB image if mode is
`rgb_array`. Otherwise return nothing and render directly to a display
window.

#### Raises:

* <b>`NotImplementedError`</b>: If the environment does not support rendering.

<h3 id="reset"><code>reset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/py_environment.py">View
source</a>

``` python
reset()
```

Starts a new sequence and returns the first `TimeStep` of this sequence.

Note: Subclasses cannot override this directly. Subclasses implement
_reset() which will be called by this method. The output of _reset() will
be cached and made available through current_time_step().

#### Returns:

A `TimeStep` namedtuple containing: step_type: A `StepType` of `FIRST`. reward:
0.0, indicating the reward. discount: 1.0, indicating the discount. observation:
A NumPy array, or a nested dict, list or tuple of arrays corresponding to
`observation_spec()`.

<h3 id="seed"><code>seed</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/py_environment.py">View
source</a>

```python
seed(seed)
```

Seeds the environment.

#### Args:

*   <b>`seed`</b>: Value to use as seed for the environment.

<h3 id="step"><code>step</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/py_environment.py">View
source</a>

``` python
step(action)
```

Updates the environment according to the action and returns a `TimeStep`.

If the environment returned a `TimeStep` with
<a href="../../../tf_agents/trajectories/time_step/StepType.md#LAST"><code>StepType.LAST</code></a>
at the previous step the implementation of `_step` in the environment should
call `reset` to start a new sequence and ignore `action`.

This method will start a new sequence if called after the environment has been
constructed and `reset` has not been called. In this case `action` will be
ignored.

Note: Subclasses cannot override this directly. Subclasses implement
_step() which will be called by this method. The output of _step() will be
cached and made available through current_time_step().

#### Args:

*   <b>`action`</b>: A NumPy array, or a nested dict, list or tuple of arrays
    corresponding to `action_spec()`.

#### Returns:

A `TimeStep` namedtuple containing: step_type: A `StepType` value. reward: A
NumPy array, reward value for this timestep. discount: A NumPy array, discount
in the range [0, 1]. observation: A NumPy array, or a nested dict, list or tuple
of arrays corresponding to `observation_spec()`.

<h3 id="time_step_spec"><code>time_step_spec</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/batched_py_environment.py">View
source</a>

``` python
time_step_spec()
```
