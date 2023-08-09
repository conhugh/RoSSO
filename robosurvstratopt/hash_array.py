import jax
from jax import jit
from jax.tree_util import register_pytree_node
import functools

# Define a wrapper class that makes an array hashable
class HashableArray:
    def __init__(self, array):
        self.array = array

    def tree_flatten(self):
        return (self.array,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])
    
    def __getitem__(self, index):
        return self.array[index]
    
    # def __mul__(self, other):
    #     return HashableArray(self.array * other)

    # def __rmul__(self, other):
    #     return HashableArray(other * self.array)

register_pytree_node(
    HashableArray,
    HashableArray.tree_flatten,
    HashableArray.tree_unflatten
)


