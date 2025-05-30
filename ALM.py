import numpy as np
import matplotlib.pyplot as plt
import os
# 参数定义
N = 3  # 时间步数
DELTA = 0.05
SIGMA = 0.5  # 初始惩罚参数
BETA = 1.3   # sigma 更新因子
ALPHA = 0.0001  #学习率
EPSILON_optimality = 0.0001  #容差
EPSILON_feasibility = 0.001
MAX_ITER = 50000000

MAX_PAINT = 30000
U0 = 2.175
U1 = -0.25
def generate_filename():
    return f"obj__N_{N}__SIGMA_{SIGMA}__BETA_{BETA}__ALPHA_{ALPHA}__MAXITER_{MAX_ITER}__\
    EPSILON1_{EPSILON_optimality}__EPSILON2_{EPSILON_feasibility}__U0_{U0}__U1_{U1}"
filename = generate_filename()

save_dir = './' + filename
if not os.path.exists(filename):
    os.makedirs(filename)

# 初始状态
x0 = np.array([0.0, -1.0, 0.0])

# 参考控制
ur = np.array([2.3, 0.0])

# 动态生成参考轨迹
xr = np.zeros((N+1, 3))
xr[0, 0] = 0
xr[0, 1] = 0
xr[0, 2] = 0
for k in range(N):
    xr[k + 1, 0] = xr[k, 0] + DELTA * ur[0] * np.cos(xr[k, 2])
    xr[k + 1, 1] = xr[k, 1] + DELTA * ur[0] * np.sin(xr[k, 2])
    xr[k + 1, 2] = xr[k, 2] + DELTA * ur[1]

# 控制输入（初始值）
u = np.zeros((N, 2))
for k in range(N):
    u[k, 0] = U0
    u[k, 1] = U1

# 拉格朗日乘子
gamma = np.zeros((7, N))

# 系统动态
def compute_next_state(x_current, v, omega):
    x_next = np.zeros(3)
    x_next[0] = x_current[0] + DELTA * v * np.cos(x_current[2])
    x_next[1] = x_current[1] + DELTA * v * np.sin(x_current[2])
    x_next[2] = x_current[2] + DELTA * omega
    return x_next

# 约束函数
def compute_constraints(x, u):
    c = np.zeros((7, N))
    for k in range(N):
        c[0, k] = 2.0 - u[k, 0]
        c[1, k] = u[k, 0] - 2.35
        c[2, k] = -1.5 - u[k, 1]
        c[3, k] = u[k, 1] - 1.0
        c[4, k] = 0.61**2 - (x[k, 0] - 3.0)**2 - x[k, 1]**2
        c[5, k] = 0.81**2 - (x[k, 0] - 6.1)**2 - (x[k, 1] + 1.0)**2
        c[6, k] = 1.02**2 - (x[k, 0] - 10.0)**2 - (x[k, 1] - 0.4)**2

    return c

# 解析梯度计算
def compute_gradient(x, u, lambda_next, gamma, sigma):
    grad = np.zeros((N, 2))
    c = compute_constraints(x, u)
    for k in range(N):
        v = u[k, 0]
        omega = u[k, 1]
        theta = x[k, 2]
        
        # 对 v_k 的梯度
        grad[k, 0] = 2.2 * (v - 2.3)
        grad[k, 0] += DELTA * lambda_next[k + 1, 0] * np.cos(theta)
        grad[k, 0] += DELTA * lambda_next[k + 1, 1] * np.sin(theta)
        grad[k, 0] -= max(0,(gamma[0, k] + sigma * c[0, k]))
        grad[k, 0] += max(0,(gamma[1, k] + sigma * c[1, k]))
                
        # 对 omega_k 的梯度
        grad[k, 1] = 0.2 * omega
        grad[k, 1] += DELTA * lambda_next[k + 1, 2]
        grad[k, 1] -= max(0,(gamma[2, k] + sigma * c[2, k]))
        grad[k, 1] += max(0,(gamma[3, k] + sigma * c[3, k]))

    return grad

# 协态变量计算
def compute_costate(x, u, lambda_k, gamma, sigma):
    
    c = compute_constraints(x, u)
    for k in range(N - 1, -1, -1):
        lambda_k[k, 0] = 2 * (x[k, 0] - xr[k, 0]) + lambda_k[k + 1, 0]
        lambda_k[k, 0] -= 2 * max(0,(gamma[4, k] + sigma * c[4, k])) * (x[k, 0] - 3.0)
        lambda_k[k, 0] -= 2 * max(0,(gamma[5, k] + sigma * c[5, k])) * (x[k, 0] - 6.1)
        lambda_k[k, 0] -= 2 * max(0,(gamma[6, k] + sigma * c[6, k])) * (x[k, 0] - 10.0)
        
        lambda_k[k, 1] = 2 * (x[k, 1] - xr[k, 1]) + lambda_k[k + 1, 1]
        lambda_k[k, 1] -= 2 * max(0,(gamma[4, k] + sigma * c[4, k])) * x[k, 1]
        lambda_k[k, 1] -= 2 * max(0,(gamma[5, k] + sigma * c[5, k])) * (x[k, 1] + 1.0)
        lambda_k[k, 1] -= 2 * max(0,(gamma[6, k] + sigma * c[6, k])) * (x[k, 1] - 0.4)
        
        lambda_k[k, 2] = 2 * (x[k, 2] - xr[k, 2]) + lambda_k[k + 1, 2]
        lambda_k[k, 2] -= DELTA * u[k, 0] * lambda_k[k + 1, 0] * np.sin(x[k, 2])
        lambda_k[k, 2] += DELTA * u[k, 0] * lambda_k[k + 1, 1] * np.cos(x[k, 2])
    return lambda_k

def compute_ALM_objective(x, u, xr, ur):
    J = 0.0
    c = compute_constraints(x, u)
    # 计算原始目标函数 J
    for k in range(N):
        state_error = x[k] - xr[k]
        J += np.dot(state_error, state_error)  # (x_k - x_{r,k})^T (x_k - x_{r,k})
        J += 1.1 * (u[k, 0] - ur[0])**2  # 2.2 (v_k - 2.3)^2
        J += 0.1 * u[k, 1]**2  # 0.2 omega_k^2

    for k in range(N):
        for i in range(7):
            J += (1/(2*sigma)) * (max(0, gamma[i,k] + sigma * c[i,k])**2 - gamma[i,k]**2)
    return J

def compute_objective(x, u, xr, ur):
    J = 0.0
    c = compute_constraints(x, u)
    # 计算原始目标函数 J
    for k in range(N):
        state_error = x[k] - xr[k]
        J += np.dot(state_error, state_error)  # (x_k - x_{r,k})^T (x_k - x_{r,k})
        J += 1.1 * (u[k, 0] - ur[0])**2  # 2.2 (v_k - 2.3)^2
        J += 0.1 * u[k, 1]**2  # 0.2 omega_k^2
    return J

# 画图代码
def results_paint(N, DELTA, ur, xr, u, x):
    #time = np.arange(0, N * DELTA, DELTA)
    time = np.linspace(0, N * DELTA, len(xr)-1)
    v_ref = np.full(N, ur[0])
    omega_ref = np.full(N, ur[1])
    obstacles = np.array([[3.0, 0.0, 0.61], [6.1, -1.0, 0.81], [10.0, 0.4, 1.02]])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(xr[:, 0], xr[:, 1], 'b--', label='Reference')
    axs[0, 0].plot(x[:, 0], x[:, 1], 'r-', label='Optimized')
    axs[0, 0].scatter(obstacles[:, 0], obstacles[:, 1], c='k', marker='o', label='Obstacles')
    # 绘制障碍物和周围的虚线圆圈
    for obs in obstacles:
        # 绘制障碍物位置
        axs[0, 0].scatter(obs[0], obs[1], c='k', marker='o')
        
        # 绘制障碍物周围的虚线圆圈
        circle = plt.Circle((obs[0], obs[1]), obs[2], 
                           fill=False, linestyle='--', color='black', linewidth=1.5)
        axs[0, 0].add_patch(circle)
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_xlabel('X Position')
    axs[0, 0].set_ylabel('Y Position')
    axs[0, 0].legend()
    axs[0, 0].set_title('Tracking Results')

    axs[0, 1].plot(time, xr[:-1, 2], 'b--', label='Reference')
    axs[0, 1].plot(time, x[:-1, 2], 'r-', label='Optimized')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Orientation (rad)')
    axs[0, 1].legend()
    axs[0, 1].set_title('Orientation')

    axs[1, 0].plot(time, v_ref, 'b--', label='Reference')
    axs[1, 0].plot(time, u[:, 0], 'r-', label='Optimized')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Linear Velocity (m/s)')
    axs[1, 0].legend()
    axs[1, 0].set_title('Linear Velocity')

    axs[1, 1].plot(time, omega_ref, 'b--', label='Reference')
    axs[1, 1].plot(time, u[:, 1], 'r-', label='Optimized')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axs[1, 1].legend()
    axs[1, 1].set_title('Angular Velocity')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"optimization_results_{filename}s.png"))
    print(f"图表已保存为 'optimization_results_{filename}.png'")
    plt.show()

# 绘制目标函数随迭代次数的变化图
def objective_function_paint(filename, J_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(J_history)), J_history, label='Augmented Objective Function')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\tilde{J}$')
    plt.title('Augmented Objective Function vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"objective_function_plot_{filename}.png"))
    print(f"目标函数图已保存为 'objective_function_plot_{filename}.png'")
    plt.show()

# 保存数据到 TXT 文件
def data_save(filename, save_dir, data_to_save):
    data_to_save = np.array(data_to_save)
    np.savetxt(os.path.join(save_dir, f"data_{filename}.txt"), data_to_save, 
           header="Iteration   J   norm   sigma   J_o", 
           fmt=['%.1f', '%.18e', '%.18e', '%.2f','%.18e']
)
    print(f"数据已保存为 'data_{filename}.txt'")

# 输出结果
def U_save(N, filename, save_dir, u, U_to_save):
    print("最优控制输入 (N =", N, "):")
    for k in range(N):
        U_to_save.append([k, u[k, 0],  u[k, 1]])
        print("时间步", k, ": v_k =", u[k, 0], ", omega_k =", u[k, 1])

    U_to_save = np.array(U_to_save)
    np.savetxt(os.path.join(save_dir, f"U_{filename}.txt"), U_to_save, 
           header="Iteration v_k omega_k",
           fmt=['%.1f', '%.18e', '%.18e'])
    print(f"数据已保存为 'U_{filename}.txt'")

#sum_if输出
def SUMIF_save(N, filename, save_dir, SUMIF_to_save):
    print("最优控制输入 (N =", N, "):")
    SUMIF_to_save = np.array(SUMIF_to_save)
    np.savetxt(os.path.join(save_dir, f"sumif_{filename}.txt"), SUMIF_to_save, 
           header="Iteration   sum_if", 
           fmt=['%.f', '%.18e'])
    print(f"数据已保存为 'sumif_{filename}.txt'")

#gamma迭代
def gamma_iter(N, gamma, sigma, c):
    for k in range(N):
        for i in range(7):
            gamma[i, k] = max(0,gamma[i, k] + sigma * c[i, k])
    return gamma

#sigma迭代
def sigma_iter(BETA, sigma):
    sigma *= BETA
    print(f"Iter {iteration}: 增大 sigma 至 {sigma:.2f}")
    return sigma


sigma = SIGMA  # sigma 在循环中更新
iteration = 0
J_history = []  # 用于保存_目标函数值_的列表
J_r_history = []
data_to_save = []  # 用于保存_数据_的列表
U_to_save = []  # 用于保存_控制变量_的列表
SUMIF_to_save = []
try:
   
    while iteration < MAX_ITER:
        # 计算状态轨迹
        x = np.zeros((N + 1, 3))
        x[0, 0] = x0[0]
        x[0, 1] = x0[1]
        x[0, 2] = x0[2]
        for k in range(N):
            x[k + 1, 0] = x[k, 0] + DELTA * u[k, 0] * np.cos(x[k, 2])
            x[k + 1, 1] = x[k, 1] + DELTA * u[k, 0] * np.sin(x[k, 2])
            x[k + 1, 2] = x[k, 2] + DELTA * u[k, 1]
        # 计算协态方程
        lambda_k = np.zeros((N + 1, 3))
        lambda_k = compute_costate(x, u, lambda_k, gamma, sigma)
        # 计算梯度
        grad = compute_gradient(x, u, lambda_k, gamma, sigma)
        #print(f"Iter {iteration}: grad[0] = {grad[0, 0]:.6f}, grad[1] = {grad[0, 1]:.6f}")
        
        
        
        # 计算梯度范数
        if iteration % 100 == 0:
            norm = 0.0
            for k in range(N):
                norm = norm + grad[k, 0]**2 + grad[k, 1]**2
            norm = np.sqrt(norm)
            #if iteration % 20 == 0:
            print("norm:",norm)

            # 检查收敛
            if norm < EPSILON_optimality:
                #print("收敛于第", iteration + 1, "次迭代")
                print("梯度范数小于容差")

                #gamma与sigma的迭代
                c = compute_constraints(x, u)
                sum_if = 0
                for k in range(N):
                    for i in range(7):
                        temp = max(c[i, k],-(gamma[i, k]/sigma))
                        sum_if += temp ** 2
                        #print(f"c[{i},{k}] _{c[i, k]}____-(gamma[{i}, {k}]/{sigma}) _{-(gamma[i, k]/sigma)}")

                print(f"sum_if {sum_if}")
                SUMIF_to_save.append([iteration,  sum_if])

                if sum_if < EPSILON_feasibility:
                    print("收敛于第", iteration + 1, "次迭代")
                    break    
                else:
                    print("更新了gamma与sigma")
                    gamma = gamma_iter(N, gamma, sigma, c)
                    sigma = sigma_iter(BETA, sigma)
        #    else:
        # 更新控制输入
        for k in range(N):
            u[k, 0] = u[k, 0] - ALPHA * grad[k, 0]
            u[k, 1] = u[k, 1] - ALPHA * grad[k, 1]
        # 画目标函数值趋势图
        if iteration % 100 == 0:
            J = compute_ALM_objective(x, u, xr, ur)
            J_o = compute_objective(x, u, xr, ur)
            if J > MAX_PAINT:
                J_history.append(0)
                raise ValueError("数据过大，无法继续执行！")
            J_history.append(J)

            print(f"Iter {iteration}: \t J = {J:.10f} \t JO = {J_o:.10f} \t", end=' ')
            data_to_save.append([iteration,  J,  norm, sigma, J_o])

        iteration = iteration + 1

finally:
    data_save(filename, save_dir, data_to_save)
    U_save(N, filename, save_dir, u, U_to_save)
    SUMIF_save(N, filename, save_dir, SUMIF_to_save)
    objective_function_paint(filename, J_history)
    results_paint(N, DELTA, ur, xr, u, x)
    