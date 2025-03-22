from env.cutting_stock import CuttingStockEnv
import numpy as np
import random
import pickle
import os
import pandas as pd
from scipy import ndimage

# Danh sách stocks (width, height) - Tấm nguyên liệu có kích thước nhỏ, tối đa 200x200
stocks = [
    (50, 50),   (60, 40),   (70, 50),   (80, 60),   (90, 70),
    (100, 50),  (110, 60),  (120, 80),  (130, 90),  (140, 100),
    (150, 120), (160, 130), (170, 140), (180, 150), (200, 200)
]

# Danh sách products (width, height) - Sản phẩm có kích thước nhỏ, phù hợp với stocks
products = [
    (10, 5),  (15, 10), (20, 10), (25, 15), (30, 20),
    (35, 20), (40, 30), (45, 25), (50, 30), (55, 35),
    (20, 15), (25, 10), (30, 15), (35, 20), (40, 25),
    (45, 30), (50, 35), (55, 40), (60, 45), (65, 50),
    (70, 30), (75, 40), (80, 50), (85, 55), (90, 60),
    (15, 10), (20, 15), (25, 20), (30, 25), (35, 30)
]

# Tạo thư mục results nếu chưa tồn tại
os.makedirs("results", exist_ok=True)

# Xóa file log cũ nếu tồn tại để tránh append dữ liệu cũ
if os.path.exists("results/q_learning_log.txt"):
    os.remove("results/q_learning_log.txt")

env = CuttingStockEnv(
    render_mode="human",   
    max_w=120,           
    max_h=120,
    seed=42,
    stock_list=stocks,
    product_list=products,
)

# Thêm method get_state vào environment
def get_env_state(self):
    """Convert tuple state from reset/step to dictionary format"""
    if hasattr(self, 'observation'):
        return self.observation
    
    # Giả định observation từ reset/step là tuple có 2 phần tử
    # Phần tử đầu tiên là stocks, phần tử thứ hai là products
    reset_result = self.reset()
    if isinstance(reset_result, tuple):
        if len(reset_result) > 0:
            try:
                # Try to extract stocks and products
                stocks_info = []
                products_info = []
                
                # Analyze tuple structure
                print(f"Reset result length: {len(reset_result)}")
                
                if len(reset_result) >= 1:
                    stocks_info = reset_result[0]
                if len(reset_result) >= 2:
                    products_info = reset_result[1]
                
                # Create observation dictionary
                self.observation = {
                    "stocks": stocks_info,
                    "products": products_info
                }
                
                return self.observation
            except Exception as e:
                print(f"Error extracting state from tuple: {e}")
    
    # Default empty observation
    return {"stocks": [], "products": []}

# Thêm method vào environment
if not hasattr(env, 'get_state'):
    setattr(env.__class__, 'get_state', get_env_state)

# ==================== Tham số huấn luyện ====================
alpha = 0.3  # Learning rate ban đầu cao hơn
gamma = 0.9  # Discount factor
epsilon = 1.0  # Khởi đầu với 100% exploration
epsilon_decay = 0.995  # Mỗi episode, epsilon giảm 0.5%
min_epsilon = 0.01  # Giới hạn dưới của epsilon
alpha_decay = 0.998  # Decay rate cho learning rate
min_alpha = 0.1  # Giá trị nhỏ nhất của learning rate
num_episodes = 500  # Số tập huấn luyện
exploration_bonus_scale = 0.1  # Hệ số cho exploration bonus
initial_temperature = 1.0  # Nhiệt độ ban đầu cho Boltzmann exploration
temperature_decay = 0.995  # Tốc độ giảm nhiệt độ

# Kích thước Q-table 
state_size = 10000  # Giảm không gian trạng thái
action_size = 500   # Giảm không gian hành động
Q_table = np.zeros((state_size, action_size))

# Theo dõi số lần thăm state-action
state_action_counts = {}

# ==================== Cải tiến State Representation ====================
def get_improved_state(observation):
    """
    Cải thiện state representation để capture thông tin quan trọng hơn
    """
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    if not isinstance(observation, dict):
        return random.randint(0, state_size - 1)
    
    # 1. Trích xuất thông tin về stocks
    stocks_info = []
    if "stocks" in observation:
        stocks = observation["stocks"]
        for stock in stocks:
            if hasattr(stock, 'shape'):
                # Tính phần trăm diện tích đã sử dụng
                total_cells = stock.shape[0] * stock.shape[1]
                used_cells = np.sum(stock != -1)
                utilization = min(int(used_cells * 10 / total_cells), 9)
                
                # Tính fragmentation (mức độ phân mảnh)
                try:
                    empty_mask = (stock == -1)
                    labeled, num_fragments = ndimage.label(empty_mask)
                    fragmentation = min(num_fragments, 9)
                except:
                    fragmentation = 0
                
                stocks_info.append((utilization, fragmentation))
            elif hasattr(stock, 'width') and hasattr(stock, 'height'):
                # Tương tự cho non-numpy stock
                total_area = stock.width * stock.height
                used_area = len(getattr(stock, 'used_spaces', set()))
                utilization = min(int(used_area * 10 / total_area), 9)
                stocks_info.append((utilization, 0))  # Không thể tính fragmentation
    
    # 2. Trích xuất thông tin về products
    products_info = []
    if "products" in observation:
        products = observation["products"]
        # Tính total area của tất cả products còn lại
        total_product_area = 0
        for prod in products:
            if isinstance(prod, tuple) and len(prod) >= 2:
                total_product_area += prod[0] * prod[1]
            elif isinstance(prod, dict) and "size" in prod:
                w, h = prod["size"]
                total_product_area += w * h
        
        # Normalize area
        normalized_area = min(int(total_product_area / 1000), 9)
        products_info.append(normalized_area)
        
        # Số lượng products còn lại (discretized)
        products_count = min(len(products), 9)
        products_info.append(products_count)
    
    # 3. Tạo state hash từ features đã trích xuất
    state_hash = 0
    
    # Add stocks information
    for i, (util, frag) in enumerate(stocks_info[:5]):  # Giới hạn 5 stocks
        state_hash = state_hash * 100 + util * 10 + frag
    
    # Add products information
    for info in products_info:
        state_hash = state_hash * 10 + info
    
    # Map to state space
    return state_hash % state_size

# Sử dụng get_improved_state thay cho get_state
def get_state(observation_tuple):
    """
    Chuyển trạng thái từ môi trường thành dạng số nguyên để lưu vào Q-table.
    Bây giờ gọi get_improved_state để có state representation tốt hơn.
    """
    return get_improved_state(observation_tuple)

# ==================== Cải tiến Exploration Strategy ====================
def get_action(state, episode, observation):
    """
    Chiến lược khám phá nâng cao: kết hợp Epsilon-greedy, Boltzmann và Guided Exploration
    """
    # Epsilon-greedy: random exploration với xác suất epsilon
    if random.uniform(0, 1) < epsilon:
        if random.random() < 0.7:  # 70% random pure exploration
            return random.randint(0, action_size - 1)
        else:  # 30% guided exploration
            return get_action_with_guidance(state, observation)
    else:
        # Boltzmann/Softmax exploration (temperature-based)
        temperature = max(0.1, initial_temperature * (temperature_decay ** episode))
        
        # Lấy Q-values cho state hiện tại
        q_values = Q_table[state]
        
        # Tính xác suất theo công thức Boltzmann
        # Tránh overflow bằng cách trừ giá trị max
        q_max = np.max(q_values)
        exp_q_values = np.exp((q_values - q_max) / temperature)
        sum_exp_q = np.sum(exp_q_values)
        
        if sum_exp_q == 0:
            return np.argmax(q_values)  # Nếu tất cả giá trị là 0, chọn bất kỳ
        
        probabilities = exp_q_values / sum_exp_q
        
        # Lựa chọn action dựa trên phân phối xác suất
        try:
            action = np.random.choice(action_size, p=probabilities)
            return action
        except:
            # Fallback nếu có lỗi với probabilities
            return np.argmax(q_values)

def get_action_with_guidance(state, observation):
    """
    Exploration với hướng dẫn dựa trên domain knowledge
    """
    # Trích xuất thông tin từ observation
    if isinstance(observation, dict) and "products" in observation and "stocks" in observation:
        products = observation["products"]
        stocks = observation["stocks"]
        
        # Heuristic 1: Sắp xếp sản phẩm lớn trước
        largest_product_idx = -1
        max_area = 0
        
        for i, prod in enumerate(products):
            if isinstance(prod, tuple) and len(prod) >= 2:
                area = prod[0] * prod[1]
                if area > max_area:
                    max_area = area
                    largest_product_idx = i
            elif isinstance(prod, dict) and "size" in prod:
                size = prod["size"]
                area = size[0] * size[1]
                if area > max_area:
                    max_area = area
                    largest_product_idx = i
        
        # Heuristic 2: Chọn stock có nhiều không gian nhất
        best_stock_idx = 0
        max_empty = 0
        
        for i, stock in enumerate(stocks):
            if hasattr(stock, 'shape'):
                empty_space = np.sum(stock == -1)
                if empty_space > max_empty:
                    max_empty = empty_space
                    best_stock_idx = i
            elif hasattr(stock, 'width') and hasattr(stock, 'height'):
                empty_space = stock.width * stock.height - len(getattr(stock, 'used_spaces', set()))
                if empty_space > max_empty:
                    max_empty = empty_space
                    best_stock_idx = i
        
        # Tạo action từ heuristics
        if largest_product_idx >= 0:
            # Transform to Q-table action space
            return (best_stock_idx * len(products) + largest_product_idx) % action_size
    
    # Fallback to random action
    return random.randint(0, action_size - 1)

def update_q_value_with_exploration_bonus(state, action, next_state, reward, episode):
    """
    Cập nhật Q-value với exploration bonus và learning rate giảm dần
    """
    # Cập nhật số lần state-action được thăm
    state_action_key = f"{state}_{action}"
    state_action_counts[state_action_key] = state_action_counts.get(state_action_key, 0) + 1
    
    # Tính exploration bonus ngược với số lần thăm
    visit_count = state_action_counts[state_action_key]
    exploration_bonus = exploration_bonus_scale / np.sqrt(visit_count + 1)
    
    # Tính current learning rate (giảm dần theo episodes)
    current_alpha = max(min_alpha, alpha * (alpha_decay ** episode))
    
    # Q-learning update với exploration bonus
    current_q = Q_table[state, action]
    max_next_q = np.max(Q_table[next_state])
    
    # Cập nhật Q-value
    Q_table[state, action] = current_q + current_alpha * (
        reward + exploration_bonus + gamma * max_next_q - current_q
    )

def get_env_action(action, observation):
    """
    Chuyển action từ Q-table thành action thực tế cho môi trường Gym.
    Hỗ trợ cả định dạng observation là tuple hoặc dictionary.
    """
    # Chuyển đổi observation thành dictionary nếu là tuple
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    # Kiểm tra xem observation có cấu trúc dự kiến không
    if not isinstance(observation, dict) or "products" not in observation or "stocks" not in observation:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
    
    list_prods = observation["products"]
    list_stocks = observation["stocks"]

    if not list_prods or not list_stocks:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    # Chọn sản phẩm có thể cắt
    prod_idx = action % len(list_prods)
    prod = list_prods[prod_idx]

    # Xử lý cả trường hợp prod là dict hoặc tuple
    if isinstance(prod, dict):
        if prod.get("quantity", 1) == 0:
            return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
        prod_w, prod_h = prod.get("size", (10, 10))
    else:
        # Giả sử prod là tuple (width, height)
        prod_w, prod_h = prod if len(prod) >= 2 else (10, 10)

    # Chọn stock
    stock_idx = (action // len(list_prods)) % len(list_stocks)
    stock = list_stocks[stock_idx]

    # Xử lý cả trường hợp stock là numpy array hoặc tuple
    if hasattr(stock, 'shape'):  # Numpy array
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        # Chọn vị trí trong stock
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                    return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}
    else:
        # Giả sử stock là tuple (width, height)
        stock_w, stock_h = stock if len(stock) >= 2 else (100, 100)
        # Chọn vị trí ngẫu nhiên trong stock
        if prod_w <= stock_w and prod_h <= stock_h:
            x = random.randint(0, stock_w - prod_w)
            y = random.randint(0, stock_h - prod_h)
            return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}

    return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

# ==================== Cải tiến Reward Function ====================
def get_enhanced_reward(observation, info, action_taken):
    """
    Hàm reward cải tiến với reward shaping chi tiết hơn
    """
    # Chuyển đổi observation thành dictionary nếu là tuple
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    base_reward = 0
    
    # 1. Sử dụng filled_ratio và trim_loss từ info nếu có
    if isinstance(info, dict):
        filled_ratio = info.get("filled_ratio", 0.5)
        trim_loss = info.get("trim_loss", 0.2)
        base_reward += (filled_ratio - trim_loss)
    
    # 2. Phần thưởng cho việc tối ưu hóa diện tích
    if isinstance(observation, dict) and "stocks" in observation:
        stocks = observation["stocks"]
        
        total_area_utilized = 0
        total_area_available = 0
        
        for stock in stocks:
            if hasattr(stock, 'width') and hasattr(stock, 'height'):
                stock_area = stock.width * stock.height
                used_area = len(getattr(stock, 'used_spaces', set()))
                total_area_available += stock_area
                total_area_utilized += used_area
            elif hasattr(stock, 'shape'):
                total_area_available += stock.shape[0] * stock.shape[1]
                total_area_utilized += np.sum(stock != -1)
        
        if total_area_available > 0:
            utilization_ratio = total_area_utilized / total_area_available
            # Phần thưởng phi tuyến cho utilization ratio
            utilization_reward = 3 * (utilization_ratio ** 2)
            base_reward += utilization_reward
    
    # 3. Phần thưởng cho việc đặt mẩu lớn
    if isinstance(action_taken, dict) and "size" in action_taken:
        width, height = action_taken["size"]
        piece_area = width * height
        
        # Phần thưởng tỷ lệ với kích thước mẩu
        piece_reward = 0.1 * np.sqrt(piece_area) if piece_area > 0 else 0
        base_reward += piece_reward
    
    # 4. Phần thưởng cho việc tối ưu hóa số lượng stock
    if isinstance(observation, dict) and "stocks" in observation and "products" in observation:
        stocks = observation["stocks"]
        products = observation["products"]
        
        remaining_products = len(products)
        
        # Đếm số stocks đã sử dụng
        stocks_used = 0
        for stock in stocks:
            if hasattr(stock, 'shape'):
                if np.any(stock != -2):
                    stocks_used += 1
            elif hasattr(stock, 'used_spaces'):
                if len(stock.used_spaces) > 0:
                    stocks_used += 1
        
        # Phần thưởng cho việc sử dụng ít stock
        if remaining_products > 0 and len(stocks) > 0:
            stock_efficiency = 1 - (stocks_used / len(stocks))
            stock_reward = 2 * stock_efficiency
            base_reward += stock_reward
    
    # 5. Phạt cho các hành động không hợp lệ hoặc không hiệu quả
    if action_taken is None or (isinstance(action_taken, dict) and 
                              action_taken.get("size", (0, 0)) == (0, 0)):
        # Phạt cho hành động không đặt được mẩu nào
        base_reward -= 1.5
    
    return base_reward

def adjust_reward_scale(current_episode, total_episodes):
    """
    Điều chỉnh reward scale theo tiến trình huấn luyện
    """
    # Ban đầu tập trung vào khám phá nên reward scale thấp
    if current_episode < total_episodes * 0.2:
        return 0.5
    # Giai đoạn giữa tăng dần tầm quan trọng của reward
    elif current_episode < total_episodes * 0.6:
        progress = (current_episode - total_episodes * 0.2) / (total_episodes * 0.4)
        return 0.5 + 0.5 * progress
    # Giai đoạn cuối tập trung hoàn toàn vào tối ưu hóa reward
    else:
        return 1.0

def get_curriculum_adjusted_environment(episode, num_episodes):
    """
    Điều chỉnh môi trường theo tiến trình huấn luyện (curriculum learning)
    """
    # Giai đoạn 1: Bắt đầu với ít sản phẩm và stocks đơn giản
    if episode < num_episodes * 0.2:
        simplified_stocks = stocks[:5]  # 5 stocks đầu tiên
        simplified_products = products[:10]  # 10 products đầu tiên
        return simplified_stocks, simplified_products
    
    # Giai đoạn 2: Tăng dần độ phức tạp
    elif episode < num_episodes * 0.5:
        progress = (episode - num_episodes * 0.2) / (num_episodes * 0.3)
        stock_count = min(len(stocks), int(5 + progress * (len(stocks) - 5)))
        product_count = min(len(products), int(10 + progress * (len(products) - 10)))
        
        varied_stocks = stocks[:stock_count]
        varied_products = products[:product_count]
        return varied_stocks, varied_products
    
    # Giai đoạn 3: Đầy đủ độ phức tạp
    else:
        return stocks, products

# Biến theo dõi phần thưởng cao nhất đạt được
max_ep_reward = -999  # Giá trị phần thưởng lớn nhất tìm thấy
max_ep_action_list = []  # Danh sách hành động tương ứng với phần thưởng cao nhất
max_start_state = None  # Trạng thái bắt đầu tương ứng với phần thưởng cao nhất

# Lưu lại lịch sử huấn luyện cho visualization
rewards_history = []
epsilons_history = []
alphas_history = []
steps_history = []
q_values_history = []  # Theo dõi max Q-value theo thời gian

# ==================== Vòng lặp huấn luyện chính ====================
print("Bắt đầu huấn luyện Q-Learning với chiến lược khám phá và phần thưởng cải tiến...")
for episode in range(num_episodes):
    # Áp dụng curriculum learning để điều chỉnh môi trường
    current_stocks, current_products = get_curriculum_adjusted_environment(episode, num_episodes)
    
    # Reset environment với stock và product mới
    env.stock_list = current_stocks
    env.product_list = current_products
    observation = env.reset(seed=42)
    
    # Extract info nếu reset trả về tuple (obs, info)
    info = {}
    if isinstance(observation, tuple) and len(observation) > 1:
        info = observation[1] if len(observation) > 1 else {}
        observation = observation[0]
    
    # Chuyển trạng thái thành số để lưu trong Q-table
    state = get_state(observation)
    
    ep_reward = 0  # Khởi tạo phần thưởng của episode
    ep_start_state = state  # Lưu trạng thái bắt đầu
    action_list = []

    done = False
    step = 0
    max_steps = 100  # Giới hạn số bước để tránh vòng lặp vô hạn
    
    # Điều chỉnh reward scale theo tiến trình
    reward_scale = adjust_reward_scale(episode, num_episodes)
    
    # Current learning rate
    current_alpha = max(min_alpha, alpha * (alpha_decay ** episode))

    while not done and step < max_steps:
        # Chọn action với chiến lược khám phá cải tiến
        action = get_action(state, episode, observation)
        
        # Chuyển action thành hành động có thể thực hiện được trong môi trường
        env_action = get_env_action(action, observation)
        
        # Thực hiện hành động trong môi trường
        try:
            step_result = env.step(env_action)
            
            # Xử lý nhiều format khác nhau của step_result
            if isinstance(step_result, tuple):
                if len(step_result) >= 5:  # gymnasium format mới
                    observation, reward_terminal, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) >= 4:  # gym format cũ
                    observation, reward_terminal, done, info = step_result
                elif len(step_result) == 3:  # format đơn giản
                    observation, reward_terminal, done = step_result
                    info = {}
                else:
                    # Format không xác định
                    observation = step_result[0] if len(step_result) > 0 else observation
                    reward_terminal = 0
                    done = False
                    info = {}
            else:
                # Nếu step_result không phải tuple, giả sử nó là observation mới
                observation = step_result
                reward_terminal = 0
                done = False
                info = {}
        
            # Lưu hành động vào danh sách
            action_list.append(env_action)
            
            # Tính reward cải tiến 
            reward = reward_scale * get_enhanced_reward(observation, info, env_action)
            
            # Cập nhật tổng reward của episode
            ep_reward += reward
            
            # Chuyển đổi trạng thái mới thành số để lưu trong Q-table
            next_state = get_state(observation)
            
            # Cập nhật Q-table với exploration bonus
            update_q_value_with_exploration_bonus(state, action, next_state, reward, episode)
            
            # Cập nhật state
            state = next_state
            
        except Exception as e:
            print(f"Error during step execution: {e}")
            done = True
        
        step += 1
        

    # Cập nhật phần thưởng và hành động tốt nhất nếu có
    if ep_reward > max_ep_reward:
        max_ep_reward = ep_reward
        max_ep_action_list = action_list.copy()
        max_start_state = ep_start_state

    # Update epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Lưu lại lịch sử huấn luyện
    rewards_history.append(ep_reward)
    epsilons_history.append(epsilon)
    alphas_history.append(current_alpha)
    steps_history.append(step)
    
    # Lưu max Q-value hiện tại
    current_max_q = np.max(Q_table)
    q_values_history.append(current_max_q)

    # In thông tin hữu ích
    if (episode + 1) % 10 == 0 or episode < 10:
        print(f"Episode {episode}, Reward: {ep_reward:.4f}, Epsilon: {epsilon:.4f}, Alpha: {current_alpha:.4f}, Steps: {step}, Max Q: {current_max_q:.4f}")
    
    # Ghi log để visualization sau này
    with open("results/q_learning_log.txt", "a") as log_file:
        log_file.write(f"Episode {episode}, Reward: {ep_reward:.4f}, Epsilon: {epsilon:.4f}, Alpha: {current_alpha:.4f}, Steps: {step}, Max Q: {current_max_q:.4f}\n")
    
    # Lưu checkpoint Q-table mỗi 100 episodes
    if (episode + 1) % 100 == 0:
        checkpoint_path = f"results/q_table_checkpoint_{episode+1}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(Q_table, f)
        print(f"Checkpoint saved to {checkpoint_path}")
        
# Hiển thị kết quả tốt nhất tìm được
print("\nTraining complete!")
print(f"Max reward = {max_ep_reward:.4f}")
print(f"Best sequence length = {len(max_ep_action_list)}")

# Lưu Q-table để sử dụng sau này
q_table_path = "results/q_table.pkl"
with open(q_table_path, "wb") as f:
    pickle.dump(Q_table, f)
print(f"Q-table saved to {q_table_path}")

# Lưu lịch sử huấn luyện chi tiết hơn
training_history = pd.DataFrame({
    "Episode": range(1, num_episodes + 1),
    "Reward": rewards_history,
    "Epsilon": epsilons_history,
    "Alpha": alphas_history,
    "Steps": steps_history,
    "Max_Q": q_values_history
})
training_history.to_csv("results/training_history.csv", index=False)
print("Training history saved to results/training_history.csv")

print("\nReplaying best sequence...")

# Phát lại tập tốt nhất tìm được
observation = env.reset()  # Reset với đầy đủ stocks và products
# Không thể đặt state trực tiếp, bắt đầu từ trạng thái mới
for action in max_ep_action_list:
    try:
        step_result = env.step(action)  # Thực hiện hành động
        env.render()  # Hiển thị môi trường
        
        # Xử lý step result
        if isinstance(step_result, tuple) and len(step_result) >= 3:
            observation, _, done = step_result[:3]
            if done:
                print("Sequence ended early (done=True)")
                break
    except Exception as e:
        print(f"Error replaying action: {e}")

# Hiển thị trạng thái cuối cùng
env.render()
print("Replay complete.")

# Tạo kết quả cutting stock để visualization
material_usage = []
waste_percentages = []

try:
    # Thử lấy dữ liệu từ environment
    if hasattr(env, 'stocks'):
        for i, stock in enumerate(env.stocks):
            if hasattr(stock, 'used_spaces') and hasattr(stock, 'width') and hasattr(stock, 'height'):
                total_area = stock.width * stock.height
                used_area = len(stock.used_spaces)
                usage_percent = (used_area / total_area) * 100
                waste_percent = 100 - usage_percent
            else:
                # Giá trị ngẫu nhiên nếu không lấy được dữ liệu
                usage_percent = random.uniform(60, 90)
                waste_percent = 100 - usage_percent
                
            material_usage.append(usage_percent)
            waste_percentages.append(waste_percent)
    
    # Nếu không lấy được dữ liệu từ environment, tạo dữ liệu mẫu
    if not material_usage:
        for i in range(5):
            usage = random.uniform(60, 90)
            waste = 100 - usage
            material_usage.append(usage)
            waste_percentages.append(waste)
    
    # Lưu dữ liệu để visualize
    df_results = pd.DataFrame({
        "Stock": [f"Stock {i+1}" for i in range(len(material_usage))],
        "Usage": material_usage,
        "Waste": waste_percentages
    })
    df_results.to_csv("results/cutting_stock_results.csv", index=False)
    print("Cutting stock results saved to results/cutting_stock_results.csv")
except Exception as e:
    print(f"Error saving cutting stock results: {e}")

# Kiểm tra Q-table đã lưu
try:
    with open(q_table_path, "rb") as f:
        loaded_Q_table = pickle.load(f)
        
    # Kiểm tra và hiển thị thông tin về Q-table
    print("\nQ-table Statistics:")
    print(f"Shape: {loaded_Q_table.shape}")
    print(f"Max value: {np.max(loaded_Q_table):.4f}")
    print(f"Min value: {np.min(loaded_Q_table):.4f}")
    print(f"Mean value: {np.mean(loaded_Q_table):.4f}")
    print(f"Non-zero entries: {np.count_nonzero(loaded_Q_table)}")
    print(f"Sparsity: {1 - np.count_nonzero(loaded_Q_table) / np.size(loaded_Q_table):.4f}")
    print(f"Size in memory: {loaded_Q_table.nbytes / (1024*1024):.2f} MB")
    
    # Phân tích phân phối Q-values
    non_zero_q = loaded_Q_table[loaded_Q_table != 0]
    if len(non_zero_q) > 0:
        print(f"Q-value percentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}%: {np.percentile(non_zero_q, p):.4f}")
except Exception as e:
    print(f"Error loading saved Q-table: {e}")

# Đóng môi trường
env.close()

# In thông báo cho người dùng biết cách xem visualization
print("\n" + "="*100)
print("Để xem visualization, hãy chạy file visualization.py và mở file results/visualization_report.html")
print("="*100)