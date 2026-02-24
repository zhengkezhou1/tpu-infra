import time
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

def main():
    try:
        jax.distributed.initialize()
    except:
        pass

    process_id = jax.process_index()
    devices = jax.devices()
    num_devices = len(devices)

    # 定义切分策略 (Mesh)
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), ('data',))
    data_sharding = NamedSharding(mesh, PartitionSpec('data'))

    # --- 调整维度以适配内存 ---
    dim = 4096  # 减小维度，防止 OOM
    
    def loss_fn(params, x, y):
        # 通过在同一个参数上重复计算来人为增加 MXU 负载，而不增加内存占用
        h = x
        for _ in range(200): # 循环 200 次矩阵乘法
            h = jax.nn.relu(jnp.dot(h, params))
        preds = jnp.mean(h, axis=1)
        return jnp.mean((preds - y)**2)

    @jax.jit
    def train_step(params, opt_state, x, y):
        grads = jax.grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # 只需要一个大矩阵，节省内存
    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, (dim, dim))
    
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)

    if process_id == 0:
        print(f"🚀 启动 JAX 训练负载：{num_devices} 核心并行...")

    for step in range(1000): # 增加步数，使其运行更久以便观察 metrics
        step_key = jax.random.PRNGKey(step)
        # 增大 Batch Size (8192) 增加并行度
        raw_x = jax.random.normal(step_key, (8192, dim))
        raw_y = jax.random.normal(step_key, (8192,))
        
        # 切分数据到所有芯片
        data_x = jax.device_put(raw_x, data_sharding)
        data_y = jax.device_put(raw_y, data_sharding)

        start_time = time.time()
        params, opt_state = train_step(params, opt_state, data_x, data_y)
        jax.block_until_ready(params) # 强制等待 TPU 计算完成
        dt = time.time() - start_time
        
        if process_id == 0 and (step + 1) % 5 == 0:
            print(f"Step {step+1} | 耗时: {dt:.4f}s")

if __name__ == "__main__":
    main()
