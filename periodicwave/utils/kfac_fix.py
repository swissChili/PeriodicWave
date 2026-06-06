import jax
import numpy as np
import jax.numpy as jnp
from kfac_jax._src.utils import types
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

TArrayTree = types.TArrayTree

def device_put_replicated(x, devices):
  """jax.device_put_replicated is deprecated, and
  kfac_jax.replicate_all_local_devices uses it. This is a drop-in
  replacement from the Jax docs.

  https://docs.jax.dev/en/latest/migrate_pmap.html#drop-in-replacements
  """

  mesh = Mesh(np.array(devices), ('x',))
  sharding = NamedSharding(mesh, P('x'))
  return jax.tree.map(
      lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x
  )

def replicate_all_local_devices(
    obj: TArrayTree, axis_name: str | None = None
) -> TArrayTree:
  """Replicates `obj` to all local Jax devices.

  Args:
    obj: A pytree to replicate.
    axis_name: Optional axis name for sharding. When the result will be passed
      to a pmap with a specific axis_name, this should match to avoid mesh
      sharding mismatches.

  Returns:
    The replicated pytree.
  """
  if types.tree_is_empty(obj):
    return obj

  devices = jax.local_devices()

  # When no axis_name is provided, use the original device_put_replicated.
  if axis_name is None:
    return device_put_replicated(obj, devices=devices)

  mesh = jax.sharding.Mesh(devices, (axis_name,))
  sharding = jax.NamedSharding(mesh, jax.P(axis_name))

  def _replicate_with_axis(x):
    # Stack to add the device dimension, then device_put with sharding.
    stacked = jnp.stack([x] * len(devices))
    return jax.device_put(stacked, sharding)

  return jax.tree_util.tree_map(_replicate_with_axis, obj)

