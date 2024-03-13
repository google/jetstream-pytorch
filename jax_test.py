import time
import jax
import jax.numpy as jnp
import functools


def test1():
    @functools.partial(jax.jit,
        static_argnums=(2, ))
    def f(x, i, issum):
        if issum:
            return x + i
        else:
            return x - i

    x = jnp.ones( (10, ))
    print(f(x, 0, True))
    print('cache', f._cache_size())
    print(f(x, 1, False))
    print('cache', f._cache_size())

    class A:
        def __init__(self, a):
            self.a = a

        def incr(self):
            self.a += 1

    @jax.jit
    def f(x):
        a = A(x)
        a.incr()
        return a.a

    print(f(x))
    print(f(x))
    print(f(x))

from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils

def test2():
    batch, seq, heads, dim = 96, 2048, 40, 128
    sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
    sharding = sharding.reshape((1, 8, 1,  1))
    val_sharding = sharding.reshape((1, 8, 1, 1))
    caches_k = jnp.zeros((batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16)
    caches_v = jnp.zeros((batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16)

    def insert_cache(caches_k, caches_v, pos, key, val):
        # val is of shape b,h,d
        return caches_k.at[:, :, pos, :].set(key.squeeze(2)), caches_v.at[:, :, pos, :].set(val.squeeze(2))
        #return caches_k.at[:, :, pos:pos+1, :].set(key), caches_v.at[:, :, pos:pos+1, :].set(val)


    def insert_cache2(caches_k, caches_v, pos, key, val):
        # val is of shape b,h,d
        seqlen = caches_k.shape[2]
        val = jnp.broadcast_to(val, caches_k.shape)
        iota = jnp.arange(0, seqlen).reshape(1,1, seqlen, 1)
        iota = jnp.broadcast_to(iota, caches_k.shape)
        pos = jnp.broadcast_to(pos, (seqlen, ))
        return (jnp.where(iota == pos.reshape(1,1, seqlen, 1), caches_k, key),
                jnp.where(iota == pos.reshape(1,1, seqlen, 1), caches_v, val))

    def insert_cache3(caches_k, caches_v, pos, key, val):
        return (
            jax.lax.dynamic_update_slice(caches_k, key, (0, 0, pos, 0)),
            jax.lax.dynamic_update_slice(caches_k, key, (0, 0, pos, 0)),
        )


    insert_cache = jax.jit(
        insert_cache, 
        donate_argnums=(0, 1)
    )
    insert_cache2 = jax.jit(
        insert_cache2, 
        donate_argnums=(0, 1)
    )
    insert_cache3 = jax.jit(
        insert_cache3, 
        donate_argnums=(0, 1)
    )

    subkey = jax.random.PRNGKey(234) 
    to_insert = jax.device_put(
        jax.random.normal(
            subkey, (batch, heads, 1, dim), dtype=jnp.bfloat16),
        device=val_sharding).block_until_ready()
    j = jnp.int32(7).block_until_ready()

    print('====1====')
    print(insert_cache.lower(caches_k, caches_v, j, to_insert, to_insert).as_text())
    
    print('====2====')
    print(insert_cache2.lower(caches_k, caches_v, j, to_insert, to_insert).as_text())

    print('====3====')
    print(insert_cache3.lower(caches_k, caches_v, j, to_insert, to_insert).as_text())

    rng = jax.random.PRNGKey(0) 

    for func in (insert_cache, insert_cache2, insert_cache3):
        for i in range(10):
            all_times = 0
            for j in range(40):
                rng, subkey = jax.random.split(rng)
                key = jax.device_put(
                    jax.random.normal(
                        subkey, (batch, heads, 1, dim), dtype=jnp.bfloat16),
                    device=val_sharding).block_until_ready()
                val = jax.device_put(
                    jax.random.normal(
                        subkey, (batch, heads, 1, dim), dtype=jnp.bfloat16),
                    device=val_sharding).block_until_ready()
                j = jnp.int32(j).block_until_ready()
                start = time.perf_counter()
                caches_k, caches_v = func(caches_k, caches_v, j, key, val)
                caches_k.block_until_ready()
                caches_v.block_until_ready()
                end = time.perf_counter()
                all_times += (end - start)
            print(func.__name__, 'time is', all_times)

def test3():
    import torch
    import torch_xla2
    import torch_xla2.extra
    x = jnp.ones((10, 10, 10))
    y = jnp.ones((10, 10, 10))

    def f(x, y):
        return torch.einsum("ijm, ijn -> imn", [x, y])

    def g(x, y):
        return jnp.einsum("ijm, ijn -> imn", x, y)

    print('====== 1 ======')
    with torch_xla2.tensor.XLAFunctionMode():
        print(jax.jit(torch_xla2.extra.jax_view(f)).lower(x, y).as_text())
    print('====== 2 ======')
    print(jax.jit(g).lower(x, y).as_text())


test3()
