import functools
import operator

def serialize(vec, tile_size):
  return sum((vec[i]*functools.reduce(operator.mul, tile_size[:i])
        for i in range(1, len(tile_size))),
         vec[0])

def serialize_iter(iterative, tile_size):
  return [serialize(x, tile_size) for x in iterative]

def deserialize(offset, tile_size):
  return tuple(deserialize_generator(offset, tile_size))

def deserialize_generator(offset, tile_size):
  for size in tile_size[:-1]:
    yield offset % size
    offset = offset // size
  yield offset
