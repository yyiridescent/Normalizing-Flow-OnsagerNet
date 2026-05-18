import jax
import jax.numpy as jnp
from jax import grad, vmap, random, jacfwd
import flax.linen as nn
import numpy as np
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# 生成数据
def generate_coefficients_np(dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    M_raw = rng.normal(size=(dim, dim))
    J_raw = rng.normal(size=(dim, dim))
    S_raw = rng.normal(size=(dim, dim))

    M_raw /= np.max(np.abs(M_raw))
    J_raw /= np.max(np.abs(J_raw))
    S_raw /= np.max(np.abs(S_raw))

    alpha_M = np.abs(rng.normal(size=dim))
    alpha_S = np.abs(rng.normal(size=dim))

    M = (np.diag(alpha_M) + M_raw @ M_raw.T)
    J = (J_raw - J_raw.T)
    S = (np.diag(alpha_S) + S_raw @ S_raw.T)
    print("M: ",M)
    print("J: ",J)
    print("S: ",S)
    return M, J, S


def euler_maruyama_linear_sde(
        M: np.ndarray,
        J: np.ndarray,
        S: np.ndarray,
        temperature: float,
        x0: np.ndarray,
        t0: float,
        t1: float,
        dt: float,
        seed: int = 0,
) -> np.ndarray:
    np.random.seed(seed)
    n_steps = int((t1 - t0) / dt) + 1
    xs = np.empty((n_steps, x0.shape[0]))
    xs[0] = x0
    drift_mat = -(M + J) @ S
    # drift_mat = -(M + J)
    diffusion_mat = np.sqrt(2.0 * temperature) * np.linalg.cholesky(M + 1e-12 * np.eye(M.shape[0]))
    for i in range(1, n_steps):
        dW = np.random.normal(scale=np.sqrt(dt), size=x0.shape)
        xs[i] = xs[i - 1] + drift_mat @ xs[i - 1] * dt + diffusion_mat @ dW
    return xs


def sample_data_from_sde(
        dim=2,
        num_trajectories=100,
        t0=0.0,
        t1=10.0,
        dt=0.01,
        temperature=1.0,
        init_scale=2.0,
        seed=42,
):
    M, J, S = generate_coefficients_np(dim=dim, seed=seed)

    all_trajs = []
    rng = np.random.default_rng(seed)
    init_conditions = init_scale * rng.standard_normal((num_trajectories, dim))

    for i in range(num_trajectories):
        traj = euler_maruyama_linear_sde(
            M, J, S, temperature,
            init_conditions[i],
            t0, t1, dt,
            seed=seed + i
        )
        all_trajs.append(traj)

    return all_trajs, M, J, S, temperature


def estimate_diffusion_matrix(z_t, z_next, dt):
    dx = np.array(z_next) - np.array(z_t)
    cov_matrix = np.cov(dx.T)
    D_est = cov_matrix / (2.0 * dt)
    return D_est


def ShiftedRePU(inputs):
    g = (jax.nn.relu(inputs)) ** 2 - (jax.nn.relu(inputs - 0.5)) ** 2
    return g

def neural_potential(X, weights, biases, Gamma, beta):
    # X: (batch, dim)
    # weights, biases 属于 U(Z) 网络

    # 1. 计算神经网络项 U(Z) -> 输出维度为 m
    H = X
    for l in range(len(weights) - 1):
        H = ShiftedRePU(jnp.matmul(H, weights[l]) + biases[l])
    U_Z = jnp.matmul(H, weights[-1]) + biases[-1]  # (batch, m)

    # 2. 计算线性权重项 sum(V_ij * Z_j)
    # Gamma 对应图片中的 V_ij，形状应为 (m, dim)
    linear_term = jnp.matmul(X, Gamma.T)  # (batch, m)

    # 3. 核心公式: 0.5 * sum( (U + linear)^2 )
    potential_term = 0.5 * jnp.sum((U_Z + linear_term) ** 2, axis=-1, keepdims=True)

    # 4. 正则项
    reg_term = beta * jnp.sum(X ** 2, axis=-1, keepdims=True)

    return potential_term + reg_term

# 2. Neural Network for M and W (Backbone)
def neural_A(X, weights, biases):
    H = X
    num_layers = len(weights)
    for l in range(num_layers):
        W = weights[l]
        b = biases[l]
        H = jnp.matmul(H, W) + b
        # 最后一层不加激活函数
        if l < num_layers - 1:
            H = jax.nn.tanh(H)
    return H

def neural_SymmAnti(X, weights_A, biases_A, dim):
    A_out = neural_A(X, weights_A, biases_A)
    A = A_out.reshape(-1, dim, dim)

    # 显式构造，增加微小的偏置防止奇异性
    L = jnp.tril(A)
    M = L @ jnp.transpose(L, (0, 2, 1)) + 1e-4 * jnp.eye(dim)

    # 强制 W 为纯反对称，并检查其量级
    W = (A - jnp.transpose(A, (0, 2, 1)))
    return M, W

# 3. Symmetric / Antisymmetric Decomposition(right)
def SymmAntiDecomposition(inputs, dim_low):
    A = jnp.reshape(inputs, [-1, dim_low, dim_low])

    lower_triangle = jnp.tril(A)
    upper_triangle = jnp.triu(A)

    symmetric = jnp.matmul(lower_triangle, jnp.transpose(lower_triangle, (0, 2, 1)))

    antisymmetric = upper_triangle - jnp.transpose(upper_triangle, (0, 2, 1))
    return symmetric, antisymmetric


# 5. Neural Network for Force Term(right)
def neural_force(X, weights, biases):
    num_layers = len(weights)
    H = X

    for l in range(num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = jax.nn.tanh(jnp.matmul(H, W) + b)

    W_out = weights[-1]
    b_out = biases[-1]
    Y = jnp.matmul(H, W_out) + b_out
    return Y

def neural_RHS(X, weights_potential, biases_potential, Gamma, beta,
               weights_force, biases_force, weights_A, biases_A, dim_low, alpha):
    M, W = neural_SymmAnti(X, weights_A, biases_A, dim_low)

    # 标量势能函数：将 (dim,) -> 标量
    def potential_scalar(x, w_p, b_p, Gamma, beta):
        # x: (dim,) 单个样本
        # neural_potential 接受 (1, dim) 返回 (1,1)，取 [0,0] 得标量
        return neural_potential(x[None, :], w_p, b_p, Gamma, beta)[0, 0]

    # 梯度函数：对 x 求梯度
    grad_fn = jax.grad(potential_scalar, argnums=0)
    # 对批量 X 进行 vmap，只映射第一个参数（X），其他参数不映射
    V_x = jax.vmap(grad_fn, in_axes=(0, None, None, None, None))(
        X, weights_potential, biases_potential, Gamma, beta
    )

    f = neural_force(X, weights_force, biases_force)
    MW_sum = M + W
    term1 = jnp.einsum('ijk,ik->ij', MW_sum, V_x)
    rhs = -term1 - alpha * V_x + 0 * f
    return rhs


class Coupling(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        s = nn.Dense(self.hidden_dim)(x)
        s = nn.relu(s)
        s = nn.Dense(dim, kernel_init=nn.initializers.zeros)(s)
        s = nn.tanh(s)

        t = nn.Dense(self.hidden_dim)(x)
        t = nn.relu(t)
        t = nn.Dense(dim, kernel_init=nn.initializers.zeros)(t)
        return s, t

class RealNVP(nn.Module):
    num_coupling: int = 4
    dim: int = 2

    def setup(self):
        masks = []
        for i in range(self.num_coupling):
            if i % 2 == 0:
                mask = jnp.array([0.0, 1.0])
            else:
                mask = jnp.array([1.0, 0.0])
            masks.append(mask)
        self.masks = jnp.stack(masks)
        self.couplings = [Coupling() for _ in range(self.num_coupling)]

    def __call__(self, x, inverse=False):
        log_det = jnp.zeros(x.shape[0])
        indices = reversed(range(self.num_coupling)) if inverse else range(self.num_coupling)

        for i in indices:
            mask = self.masks[i]
            x_masked = x * mask
            s, t = self.couplings[i](x_masked)
            s = s * (1.0 - mask)
            t = t * (1.0 - mask)

            if inverse:
                x = (x - t) * jnp.exp(-s)
                log_det -= jnp.sum(s, axis=-1)
            else:
                x = x_masked + (1.0 - mask) * (x * jnp.exp(s) + t)
                log_det += jnp.sum(s, axis=-1)

        return x, log_det

def compute_A_matrix(params, z0, dim):
    def drift_fn(x):
        # x: (dim,) -> 返回 (dim,)
        return neural_RHS(
            x.reshape(1, -1),                      # X
            params["weights_potential"],           # weights_potential
            params["biases_potential"],            # biases_potential
            params["Gamma"],                       # Gamma
            params["beta"],                        # beta
            params["weights_force"],               # weights_force
            params["biases_force"],                # biases_force
            params["weights_A"],                   # weights_A
            params["biases_A"],                    # biases_A
            dim,                                   # dim_low
            params["alpha"]                        # alpha
        ).reshape(-1)

    A = jax.jacfwd(drift_fn)(z0.reshape(-1))
    return A


def compute_M_W(params, z0, dim):
    M, W = neural_SymmAnti(
        z0.reshape(1, -1),
        params["weights_A"],
        params["biases_A"],
        dim
    )
    return M[0], W[0]


def compute_Hessian(params, z0, dim):
    def V_fn(x):
        # 这里的 x 形状是 (dim,)
        v_val = neural_potential_single(
            x,  # 注意：这里直接传 x 即可，不需要 reshape
            params["weights_potential"],
            params["biases_potential"],
            params["Gamma"],
            params["beta"]
        )
        return jnp.reshape(v_val, ())

    # 使用 jacfwd 的嵌套来计算 Hessian (dim, dim)
    H = jax.hessian(V_fn)(z0.reshape(-1))
    return H


def init_layer_params(rng, in_dim, out_dim, scale=0.1):
    W = scale * random.normal(rng, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return W, b


def init_mlp_params(rng, layer_sizes, scale=0.1):
    params_W = []
    params_b = []

    keys = random.split(rng, len(layer_sizes) - 1)

    for i in range(len(layer_sizes) - 1):
        W, b = init_layer_params(keys[i], layer_sizes[i], layer_sizes[i + 1], scale)
        params_W.append(W)
        params_b.append(b)

    return params_W, params_b


def init_onsager_params(rng, dim):
    k1, k2, k3, k4, k5, k6 = random.split(rng, 6)
    m = 64
    kS = random.PRNGKey(999)

    potential_layers = [dim, 64, 64, m]
    weights_potential, biases_potential = init_mlp_params(k1, potential_layers, scale=0.05)

    force_layers = [dim, 64, 64, dim]
    weights_force, biases_force = init_mlp_params(k2, force_layers, scale=0.1)

    A_layers = [dim, 64, 64, dim * dim]
    weights_A, biases_A = init_mlp_params(k3, A_layers, scale=0.01)

    # 改为数组，不要用列表
    # Gamma = 0.1 * random.normal(k4, (dim, dim))  # shape (dim, dim)
    Gamma = random.normal(k4, (m, dim)) * 1.0
    beta = jnp.array(0.01)  # scalar
    alpha = jnp.array(0.1)  # scalar

    S_raw = random.normal(kS, (dim, dim))
    S = S_raw @ S_raw.T + 0.1 * jnp.eye(dim)

    params = {
        "weights_potential": weights_potential,
        "biases_potential": biases_potential,
        "weights_force": weights_force,
        "biases_force": biases_force,
        "weights_A": weights_A,
        "biases_A": biases_A,
        "Gamma": Gamma,
        "beta": beta,
        "alpha": alpha,
        "S": S
    }
    return params


def sigma_net_apply(params, X, dim):
    # 1. 从现有网络得到 M(x)
    M, _ = neural_SymmAnti(
        X,
        params["weights_A"],
        params["biases_A"],
        dim
    )

    # 2. 数值稳定（防止非正定）
    eps = 1e-6
    I = jnp.eye(dim)
    M = M + eps * I

    # 3. Cholesky 分解: M = L L^T
    # vmap 处理 batch
    def chol(m):
        return jnp.linalg.cholesky(m)

    sigma = jax.vmap(chol)(M)
    return sigma


def neural_potential_single(z, weights, biases, Gamma, beta):
    # z: (dim,)
    z_in = z[None, :]  # (1, dim)

    # 1. 计算 U(Z)
    H = z_in
    for l in range(len(weights) - 1):
        H = ShiftedRePU(jnp.matmul(H, weights[l]) + biases[l])
    U_Z = jnp.matmul(H, weights[-1]) + biases[-1]  # (1, m)

    # 2. 计算线性项 V_ij * Z_j
    linear_term = jnp.matmul(z_in, Gamma.T)  # (1, m)

    # 3. 计算 0.5 * sum((U+linear)^2) + beta*|z|^2
    potential_energy = 0.5 * jnp.sum((U_Z + linear_term) ** 2)
    regularization = beta * jnp.sum(z ** 2)

    return potential_energy + regularization

def compute_learned_potential(x_grid, params):
    def V_fn(x):
        return neural_potential_single(
            x,
            params["weights_potential"],
            params["biases_potential"],
            params["Gamma"],
            params["beta"]
        )
    V_vals = jax.vmap(V_fn)(x_grid)
    return V_vals

def compute_true_potential(x_grid):
    # V(x) = 0.5 * ||x||^2
    return 0.5 * jnp.sum(x_grid**2, axis=1)

def create_grid(x_range=(-3, 3), num=100):
    x = jnp.linspace(x_range[0], x_range[1], num)
    y = jnp.linspace(x_range[0], x_range[1], num)
    X, Y = jnp.meshgrid(x, y)
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    return X, Y, grid

# 训练循环内
def total_loss_fn(nvp_p, ons_p, x_batch, x_next_batch):
    # 1. 映射到 z 空间并获取 Jacobian 的对数行列式
    z_batch, log_det = nvp_model.apply({'params': nvp_p}, x_batch)
    z_next_batch, _ = nvp_model.apply({'params': nvp_p}, x_next_batch)

    # 2. 计算 z 空间的 Onsager 损失
    loss_ons, _ = ons_loss_full(ons_p, z_batch, z_next_batch, dt, dim, T, x_batch, nvp_p)

    # 3. 核心：通过减去 log_det，将损失函数转换回 x 空间
    # 这样可以防止模型通过无限压缩空间来减小 loss
    total_loss = loss_ons - jnp.mean(log_det)

    return total_loss


def ons_loss_full(params, zt_batch, zn_batch, dt, dim, T, x_batch, nvp_params):
    # 调用更新后的 drift 计算
    drift, MW_sum = get_drift_with_correction(zt_batch, params, dim, T, x_batch, nvp_params)

    # 对称部分 = M
    M = 0.5 * (MW_sum + jnp.transpose(MW_sum, (0, 2, 1)))

    # 反对称部分 = W
    W = 0.5 * (MW_sum - jnp.transpose(MW_sum, (0, 2, 1)))

    # ===== 原本 likelihood =====
    cov = 2.0 * T * dt * M + 1e-6 * jnp.eye(dim)

    mean = zt_batch + dt * drift
    diff = zn_batch - mean

    sign, logdet = jnp.linalg.slogdet(cov)
    cov_inv = jnp.linalg.inv(cov)
    quad_form = jnp.einsum('bi,bij,bj->b', diff, cov_inv, diff)

    logp = -0.5 * (quad_form + logdet + dim * jnp.log(2 * jnp.pi))

    loss_total = -jnp.mean(logp)

    return loss_total, logp

def plot_potential_comparison(X, Y, V_true, V_learned):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 关键步骤：去掉常数偏移 ---
    # 将两个势能面都减去各自的平均值，使其中心对齐在 0
    V_true_centered = V_true - np.min(V_true)
    V_learned_centered = V_learned - np.min(V_learned)

    # 计算差值
    V_diff = V_learned_centered - V_true_centered

    # 统一颜色刻度范围，以提高可读性
    # vmax = max(np.max(np.abs(V_true_centered)), np.max(np.abs(V_learned_centered)))
    vmax = np.percentile(V_true_centered, 95)
    if vmax == 0: vmax = 1e-6

    titles = ["True Potential", "Learned Potential", "Difference"]
    data = [V_true_centered, V_learned_centered, V_diff]

    for i, ax in enumerate(axes):
        # 使用 imshow 或 contourf 绘图
        im = ax.imshow(data[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(titles[i], fontsize=16)
        ax.set_xlabel("x1", fontsize=14)
        ax.set_ylabel("x2", fontsize=14)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_matrix_comparison(true_mat, learned_mat, title_prefix, cmap='RdBu_r'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 计算差值
    diff_mat = learned_mat - true_mat
    print(learned_mat)
    print(true_mat)

    # 确定颜色范围，以最大绝对值为基准，保证三张图颜色尺度一致
    vmax = max(np.max(np.abs(true_mat)), np.max(np.abs(learned_mat)), np.max(np.abs(diff_mat)))
    # 避免 vmax 为 0 的情况
    if vmax == 0: vmax = 1e-6

    titles = ["True Value", "Learned Value", "Difference (Learned - True)"]
    matrices = [true_mat, learned_mat, diff_mat]

    for i, ax in enumerate(axes):
        im = ax.imshow(matrices[i], cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f"{title_prefix}: {titles[i]}", fontsize=16)
        ax.set_xlabel("Dimension Index", fontsize=14)
        ax.set_ylabel("Dimension Index", fontsize=14)

        # 添加数值标注
        rows, cols = matrices[i].shape
        for r in range(rows):
            for c in range(cols):
                text_val = matrices[i][r, c]
                # 根据背景色决定文字颜色
                color = "white" if abs(text_val) > vmax * 0.6 else "black"
                ax.text(c, r, f"{text_val:.2f}", ha="center", va="center", color=color, fontsize=12)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def get_avg_jacobian(params_nvp, x_samples):
    def single_forward(x):
        # 确保输出是 (dim,) 向量
        z, _ = nvp_model.apply({'params': params_nvp}, x[None, :])
        return z[0]

    # 使用 vmap 对整个 batch 并行计算 jacfwd
    jac_fn = jax.vmap(jax.jacfwd(single_forward))
    Js = jac_fn(x_samples)

    # 返回平均 Jacobian 矩阵 (dim, dim)
    return jnp.mean(Js, axis=0)

def get_drift_with_correction(zt_batch, params, dim, T, x_batch, nvp_params):
    # --- A. Onsager 漂移部分 ---
    def V_for_grad(z):
        return neural_potential_single(z, params["weights_potential"], params["biases_potential"],
                                       params["Gamma"], params["beta"])

    V_z_grad = jax.vmap(jax.grad(V_for_grad))(zt_batch)
    M_mat, W_mat = neural_SymmAnti(zt_batch, params["weights_A"], params["biases_A"], dim)
    MW_sum = M_mat + W_mat
    b_z_base = -jnp.einsum('bij,bj->bi', MW_sum, V_z_grad)

    # --- B. Ito 修正项: 0.5 * Tr(D_x @ H_z_x) ---
    # 定义从 x 到 z 的映射，用于求 Hessian
    def forward_fn(x):
        z, _ = nvp_model.apply({'params': nvp_params}, x[None, :])
        return z[0]

    # 计算 Hessian: (batch, dim_z, dim_x, dim_x)
    H_z_x = jax.vmap(jax.hessian(forward_fn))(x_batch)

    # 扩散矩阵 D_x = T * M_x (这里使用 S 作为估计，或者传入真实的 M)
    D_x = T * params["S"]

    # 修正项: b_ito = 0.5 * Tr(D_x @ H_z_x)
    # 对应 Einstein 求和: 0.5 * D_{jk} * H_{ijk}
    ito_corr = 0.5 * jnp.einsum('jk,bijk->bi', D_x, H_z_x)

    # --- C. 散度修正 (Gamma 项) ---
    def MW_single(z):
        m, w = neural_SymmAnti(z[None, :], params["weights_A"], params["biases_A"], dim)
        return m[0] + w[0]

    MW_jac = jax.vmap(jax.jacfwd(MW_single))(zt_batch)
    gamma_corr = T * jnp.einsum('bijk,jk->bi', MW_jac, jnp.eye(dim))

    # 最终总漂移: 包含物理漂移 + Ito修正 + 散度修正
    full_drift = b_z_base + ito_corr + gamma_corr

    return full_drift, MW_sum

def compute_rho_on_grid(kde, grid):
    """
    grid: (N_grid, dim)
    return: rho (N_grid,)
    """
    log_rho = kde.score_samples(np.array(grid))  # log ρ
    rho = np.exp(log_rho)
    return rho, log_rho

def compute_potential_from_density(log_rho, shape):
    V = -log_rho.reshape(shape)
    return V

def forward_map(x):
    z, _ = nvp_model.apply({'params': nvp_params}, x[None, :])
    return z[0]


def plot_learned_potential_only(X, Y, V_learned):
    plt.figure(figsize=(8, 6))

    # 使用 contourf 绘制填充等高线，这比 imshow 更容易看清形状趋势
    # levels=50 增加平滑度
    cp = plt.contourf(X, Y, V_learned, levels=50, cmap='RdBu_r')

    # 添加等高线线条，帮助识别椭圆主轴
    line = plt.contour(X, Y, V_learned, levels=10, colors='black', linewidths=0.5, alpha=0.5)

    plt.colorbar(cp, label='Potential Value')
    plt.title("Learned Potential", fontsize=16)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14)

    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compute_onsager_potential_x(x_grid, nvp_params, onsager_params):

    z, log_det = nvp_model.apply({'params': nvp_params}, x_grid)

    Vz = compute_learned_potential(z, onsager_params)

    # density correction
    Vx = Vz - log_det

    return Vx

if __name__ == "__main__":
    dim = 2
    dt = 0.02
    trajs, M, J, S, T = sample_data_from_sde(
        dim=dim,
        num_trajectories=50,
        t0=0.0,
        t1=10.0,
        dt=dt,
        temperature=1.0,
        init_scale=2.0,
        seed=42,
    )

    data_array = np.vstack(trajs)

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    plt.scatter(data_array[:, 0], data_array[:, 1], s=1, alpha=0.3, c='red')
    plt.title("Samples from SDE Trajectories")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    x_t = jnp.vstack([t[:-1] for t in trajs])
    x_next = jnp.vstack([t[1:] for t in trajs])
    # 将所有轨迹的所有点拼接在一起，形成一个大的数据集
    x_static = jnp.concatenate([x_t, x_next[-1:]], axis=0)

    nvp_model = RealNVP(dim=dim)
    # nvp_params = nvp_model.init(random.PRNGKey(0), jnp.ones((1, dim)))['params']
    dummy = jnp.ones((128, dim))  # 用真实 batch size 初始化
    nvp_params = nvp_model.init(random.PRNGKey(0), dummy)['params']
    nvp_opt = optax.adam(1e-3)
    nvp_state = nvp_opt.init(nvp_params)

    onsager_params = init_onsager_params(random.PRNGKey(1), dim)
    ons_opt = optax.adam(1e-4)
    ons_state = ons_opt.init(onsager_params)

    num_samples = x_t.shape[0]
    batch_size = 128
    num_epochs = 50

    rng = random.PRNGKey(42)
    epoch_losses = []  # 用于存储每个 epoch 的平均损失

    for epoch in tqdm(range(num_epochs)):
        rng, perm_rng = random.split(rng)
        perm = random.permutation(perm_rng, num_samples)

        epoch_loss_sum = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = x_t[idx]
            x_next_batch = x_next[idx]


            # 修改这里的定义，确保它能接收 nvp_params 并传递给下层
            def total_loss_fn(nvp_p, ons_p):
                # 1. 计算 z 空间映射
                z_batch, log_det = nvp_model.apply({'params': nvp_p}, x_batch)
                z_next_batch, _ = nvp_model.apply({'params': nvp_p}, x_next_batch)

                # 2. 关键修改：将 x_batch 和 nvp_p (即当前的 nvp_params) 传给 ons_loss_full
                # 注意：这里传的是 nvp_p，它是 grad_fn 正在求导的当前参数版本
                loss_onsager, _ = ons_loss_full(ons_p, z_batch, z_next_batch, dt, dim, T, x_batch, nvp_p)

                # 3. 别忘了加入 Jacobian 行列式修正项，防止 z 空间塌缩
                # return loss_onsager - jnp.mean(log_det)
                return loss_onsager


            grad_fn = jax.value_and_grad(total_loss_fn, argnums=(0, 1))
            loss_val, (grad_nvp, grad_ons) = grad_fn(nvp_params, onsager_params)

            epoch_loss_sum += loss_val
            num_batches += 1

            updates_nvp, nvp_state = nvp_opt.update(grad_nvp, nvp_state)
            nvp_params = optax.apply_updates(nvp_params, updates_nvp)
            updates_ons, ons_state = ons_opt.update(grad_ons, ons_state)
            onsager_params = optax.apply_updates(onsager_params, updates_ons)

        # 一个 epoch 结束后，计算平均损失并存储
        avg_loss = epoch_loss_sum / num_batches
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} done. Avg Loss: {avg_loss:.4f}")

    # 绘图
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, num_epochs + 1), epoch_losses)  # epoch_losses 长度 = num_epochs
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()

    print("Training Finished.")

    # 1. 基础映射：计算训练数据在隐空间的表示
    z_t, _ = nvp_model.apply({'params': nvp_params}, x_t)
    z_next, _ = nvp_model.apply({'params': nvp_params}, x_next)

    # 2. 计算 z 空间的扩散矩阵 D_z (对应你原本的 D_learned)
    dx = z_next - z_t
    D_learned_z = (dx.T @ dx) / (2.0 * dt * dx.shape[0])

    # 3. 计算 z 空间的核心矩阵 (在 z 的中心点计算)
    z0_mean = jnp.mean(z_t, axis=0)
    A_learned_z = compute_A_matrix(onsager_params, z0_mean, dim)
    M_learned_z, W_learned_z = compute_M_W(onsager_params, z0_mean, dim)
    H_learned_z = compute_Hessian(onsager_params, z0_mean, dim)

    print("正在通过平均 Jacobian 进行坐标变换...")

    # 定义内部 forward_map 以便计算 Jacobian
    def forward_map_internal(x):
        z, _ = nvp_model.apply({'params': nvp_params}, x[None, :])
        return z[0]

    # 随机采样一部分数据点用于估计平均 J
    sample_idx = np.random.choice(len(x_t), size=min(1000, len(x_t)), replace=False)
    x_estimate_j = x_t[sample_idx]

    # 计算平均 J (使用 vmap 加速)
    J_batch = jax.vmap(jax.jacfwd(forward_map_internal))(x_estimate_j)
    J_avg = jnp.mean(J_batch, axis=0)

    # 稳定求逆
    J_inv = jnp.linalg.inv(J_avg + 1e-6 * jnp.eye(dim))

    # --- 转换回物理空间 x ---
    # A_x = J^-1 @ A_z @ J
    # A_learned_x = J_inv @ A_learned_z @ J_inv.T
    #
    # # D_x = J^-1 @ D_z @ J^-T (注意扩散项的协方差变换规律)
    # D_learned_x = J_inv @ D_learned_z @ J_inv.T
    #
    # # M 和 W 的变换 (基于 MW_sum = M + W)
    # M_learned_x = J_inv @ M_learned_z @ J_inv.T
    # W_learned_x = J_inv @ W_learned_z @ J_inv.T

    # 4. 计算误差并绘图
    def rel_error(true, pred):
        return jnp.linalg.norm(true - pred) / jnp.linalg.norm(true)

    z_t, _ = nvp_model.apply({'params': nvp_params}, x_t)
    z_next, _ = nvp_model.apply({'params': nvp_params}, x_next)

    A_true = -(M + J)# drift
    D_true = T * M # diffusion

    dx = z_next - z_t# 间隔
    D_learned = (dx.T @ dx) / (2.0 * dt * dx.shape[0])

    # def forward_map(x):
    #     z, _ = nvp_model.apply({'params': nvp_params}, x[None, :])
    #     return z[0]

    # 多点平均 Jacobian（关键优化）
    sample_idx = np.random.choice(len(x_t), size=200, replace=False)
    x_sample = x_t[sample_idx]

    J_batch = jax.vmap(jacfwd(forward_map))(x_sample)
    Ja = jnp.mean(J_batch, axis=0)

    # 稳定求逆
    Ja_inv = jnp.linalg.inv(Ja + 1e-6 * jnp.eye(dim))

    z0 = jnp.mean(z_t, axis=0)

    A_learned = compute_A_matrix(onsager_params, z0, dim)
    M_learned, W_learned = compute_M_W(onsager_params, z0, dim)
    H_learned = compute_Hessian(onsager_params, z0, dim)

    print("Training Finished.")

    def ensure_2d(mat, dim):
        mat = jnp.array(mat)
        if mat.ndim == 1:
            return jnp.diag(mat)
        return mat

    A_learned = ensure_2d(A_learned, dim)
    D_learned = ensure_2d(D_learned, dim)
    H_learned = ensure_2d(H_learned, dim)
    M_learned = ensure_2d(M_learned, dim)
    W_learned = ensure_2d(W_learned, dim)

    A_learned_x = Ja @ A_learned + H_learned @ M_learned
    D_learned_x = Ja_inv @ D_learned @ Ja_inv.T
    H_learned_x = Ja.T @ H_learned @ Ja

    MW_sum_z = M_learned + W_learned
    MW_sum_x = Ja_inv @ MW_sum_z @ Ja_inv.T
    M_learned_x = (MW_sum_x + MW_sum_x.T) / 2
    W_learned_x = (MW_sum_x - MW_sum_x.T) / 2

    hessian_single = jax.hessian(forward_map)
    H = jax.vmap(hessian_single)(x_sample)  # (n_samples, dim, dim, dim)
    ito_term_per_sample = jnp.einsum('bijk,ij->bk', H, D_learned)  # (n_samples, dim)
    ito_term = jnp.mean(ito_term_per_sample, axis=0)  # (dim,)
    # 4. 修正后的 drift
    f_x = J_inv @ (A_learned_z - ito_term)
    b_z = A_learned_z @ z0
    b_x = J_inv @ (b_z - ito_term)

    def rel_error(true, pred):
        return jnp.linalg.norm(true - pred) / jnp.linalg.norm(true)


    print(f"A error: {rel_error(A_true, A_learned_x):.4f}")
    print(f"D error: {rel_error(D_true, D_learned_x):.4f}")
    print(f"M error: {rel_error(M, M_learned_x):.4f}")
    print(f"J error: {rel_error(J, W_learned_x):.4f}")

    print("变换完成，开始绘图...")
    # print("Drift Matrix:")
    # plot_matrix_comparison(np.array(A_true), np.array(A_learned_x), "Drift Matrix")
    # print("Diffusion Matrix:")
    # plot_matrix_comparison(np.array(D_true), np.array(D_learned_x), "Diffusion Matrix")
    print("Matrix D:")
    plot_matrix_comparison(np.array(M), np.array(M_learned_x), "Matrix D")
    print("Matrix Q:")
    plot_matrix_comparison(np.array(J), np.array(W_learned_x), "Matrix Q")
    print("Jacobi matrix: ")
    print("开始计算 Potential...")

    def compute_flow_potential(x_grid, nvp_params):
        z, log_det = nvp_model.apply({'params': nvp_params}, x_grid)

        # latent Gaussian energy
        Vz = 0.5 * jnp.sum(z ** 2, axis=1)

        # V(x) = -log rho(x)
        Vx = Vz - log_det

        return Vx


    ################################################################################
    def compute_true_potential_from_S(x_grid, S):
        # 计算 V(x) = 0.5 * x^T @ S @ x
        return 0.5 * jnp.einsum('ni,ij,nj->n', x_grid, S, x_grid)


    # --- 2. 准备网格数据 ---
    X, Y, grid = create_grid()

    # --- 3. 计算学习到的势能 (来自 RealNVP) ---
    V_learned_flat = compute_onsager_potential_x(
        grid,
        nvp_params,
        onsager_params
    )
    V_learned = np.array(V_learned_flat).reshape(X.shape)

    # --- 4. 修改部分：计算基于 S 矩阵的真实势能 ---
    # 注意：这里的 S 必须是 sample_data_from_sde 返回的那个真实矩阵 S
    V_true_flat = compute_true_potential_from_S(grid, S)
    V_true = np.array(V_true_flat).reshape(X.shape)

    # --- 5. 归一化（对齐零点） ---
    # 减去中心点的值，使得中心位置的势能为 0，方便对比形状
    V_true = V_true - V_true[50, 50]
    V_learned = V_learned - V_learned[50, 50]

    # --- 6. 绘图对比 ---
    plot_potential_comparison(X, Y, V_true, V_learned)
    plot_learned_potential_only(X, Y, V_learned)
