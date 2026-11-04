import numpy as np
import copy
import mimicgen
import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.tensor_utils as TensorUtils
from scipy.spatial.transform import Rotation

from robomimic.config import config_factory


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert not abs_env.env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')
    
    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({'states': states[0]})
        for i in range(len(states)):
            obs = env.reset_to({'states': states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()['states'])
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info

class RobomimicObsConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        # config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        # ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # env_meta['env_kwargs']['camera_names'] = ['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']

        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'agentview_full', 'robot0_robotview', 'robot0_eye_in_hand'], 
            camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand'], 
            camera_height=84, 
            camera_width=84, 
            reward_shaping=False,
        )
        # env = EnvUtils.create_env_from_metadata(
        #     env_meta=env_meta,
        #     render=True, 
        #     render_offscreen=True,
        #     use_image_obs=True,
        # )

        self.env = env
        self.file = h5py.File(dataset_path, 'r')
    
    def __len__(self):
        return len(self.file['data'])

    def convert_obs(self, initial_state, states):
        obss = []
        self.env.reset()
        obs = self.env.reset_to(initial_state)
        obss.append(obs)
        for i in range(1, len(states)):
            obs = self.env.reset_to({'states': states[i]})
            obss.append(obs)
        return TensorUtils.list_of_flat_dict_to_dict_of_list(obss)

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input

        states = demo['states'][:]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo.attrs["model_file"]

        # generate abs actions
        obss = self.convert_obs(initial_state, states)
        del obss['birdview_image']
        del obss['birdview_depth']
        del obss['agentview_depth']
        del obss['sideview_image']
        del obss['sideview_depth']
        del obss['robot0_eye_in_hand_depth']
        return obss


# import numpy as np
# import copy
# import h5py
# import mimicgen
# import robomimic.utils.obs_utils as ObsUtils
# import robomimic.utils.file_utils as FileUtils
# import robomimic.utils.env_utils as EnvUtils
# import robomimic.utils.tensor_utils as TensorUtils
# from scipy.spatial.transform import Rotation

# # --- 新增：用于获取相机矩阵 ---
# def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
#     cam_id = sim.model.camera_name2id(camera_name)
#     fovy = sim.model.cam_fovy[cam_id]
#     f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
#     K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
#     return K

# def get_camera_extrinsic_matrix(sim, camera_name):
#     cam_id = sim.model.camera_name2id(camera_name)
#     #相机位置和旋转矩阵
#     cam_pos = sim.data.cam_xpos[cam_id]
#     cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
#     # 构建 4x4 变换矩阵 T_world_to_cam
#     T = np.eye(4)
#     T[:3, :3] = cam_rot
#     T[:3, 3] = cam_pos
#     return T
# # ---------------------------

# class RobomimicAbsoluteActionConverter:
#     # ... (保持这部分代码不变) ...
#     def __init__(self, dataset_path, algo_name='bc'):
#         # ... (保持原样)
#         pass 
#     # ... (保持原样)

# class RobomimicObsConverter:
#     def __init__(self, dataset_path, algo_name='bc'):
#         # 获取原始元数据
#         env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        
#         # --- 修改开始：通过 env_meta 强制开启深度图 ---
#         if 'env_kwargs' not in env_meta:
#             env_meta['env_kwargs'] = {}
            
#         # 强制设置 use_depth_obs 为 True
#         env_meta['env_kwargs']['use_depth_obs'] = True
#         # 确保 use_camera_obs 也是 True
#         env_meta['env_kwargs']['use_camera_obs'] = True 
#         # ----------------------------------------

#         # 调用函数时去掉 use_depth_obs 参数，它会从 env_meta 里读取配置
#         env = EnvUtils.create_env_for_data_processing(
#             env_meta=env_meta,
#             camera_names=['agentview', 'robot0_eye_in_hand'], 
#             camera_height=84, 
#             camera_width=84, 
#             reward_shaping=False,
#             # use_depth_obs=True,  <-- 这一行删掉，不要传这个参数
#         )

#         self.env = env
#         self.file = h5py.File(dataset_path, 'r')
#         self.camera_names = ['agentview', 'robot0_eye_in_hand']
#         self.num_points = 1024 # 目标点云数量

#     def __len__(self):
#         return len(self.file['data'])

#     def convert_obs(self, initial_state, states):
#         obss = []
#         self.env.reset()
#         obs = self.env.reset_to(initial_state)
        
#         # 修改 2: 在每一步处理中加入点云生成
#         obs['point_cloud'] = self.get_point_cloud(obs)
#         obss.append(obs)
        
#         for i in range(1, len(states)):
#             obs = self.env.reset_to({'states': states[i]})
#             obs['point_cloud'] = self.get_point_cloud(obs)
#             obss.append(obs)
            
#         return TensorUtils.list_of_flat_dict_to_dict_of_list(obss)

#     # 新增: 点云生成逻辑
#     def get_point_cloud(self, obs):
#         all_points = []
#         all_colors = []
        
#         sim = self.env.env.sim
#         h, w = 84, 84
        
#         for cam_name in self.camera_names:
#             # 获取 RGB 和 Depth
#             rgb = obs[f'{cam_name}_image'] # (C, H, W)
#             depth = obs[f'{cam_name}_depth'] # (1, H, W)
            
#             # 调整维度为 (H, W, C)
#             rgb = np.moveaxis(rgb, 0, -1)
#             depth = np.moveaxis(depth, 0, -1).squeeze(-1)
            
#             # 真实深度值转换 (robosuite depth is normalized, need real distance)
#             # 注意: create_env_for_data_processing 返回的 depth 通常是真实距离 
#             # 如果是归一化的 (0-1)，需要反归一化，但 robosuite wrapper 通常返回真实值
#             real_depth = depth 

#             # 获取矩阵
#             K = get_camera_intrinsic_matrix(sim, cam_name, h, w)
#             T_cam_world = get_camera_extrinsic_matrix(sim, cam_name) # T_world_to_cam (camera pose)
            
#             # 反投影
#             # 创建像素网格
#             u, v = np.meshgrid(np.arange(w), np.arange(h))
#             u = u.flatten()
#             v = v.flatten()
#             z = real_depth.flatten()
            
#             # 过滤掉无效深度 (背景)
#             valid = (z > 0.01) & (z < 2.0) # 根据场景调整阈值
#             u, v, z = u[valid], v[valid], z[valid]
            
#             # 像素坐标 -> 相机坐标
#             x = (u - K[0, 2]) * z / K[0, 0]
#             y = (v - K[1, 2]) * z / K[1, 1]
#             points_cam = np.stack([x, y, z], axis=1) # (N, 3)
            
#             # 相机坐标 -> 世界坐标
#             # P_world = T * P_cam
#             # T_cam_world 是相机在世界坐标系下的位姿，所以直接用它变换
#             # R = T_cam_world[:3, :3], t = T_cam_world[:3, 3]
#             points_world = points_cam @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
            
#             # 颜色
#             color = rgb.reshape(-1, 3)[valid]
            
#             all_points.append(points_world)
#             all_colors.append(color)

#         if not all_points:
#             return np.zeros((self.num_points, 6), dtype=np.float32)

#         all_points = np.concatenate(all_points, axis=0)
#         all_colors = np.concatenate(all_colors, axis=0)
        
#         # 采样 (FPS 或 随机采样)
#         # 为了速度这里演示随机采样，如果需要更高质量请实现 FPS
#         if all_points.shape[0] > self.num_points:
#             indices = np.random.choice(all_points.shape[0], self.num_points, replace=False)
#             all_points = all_points[indices]
#             all_colors = all_colors[indices]
#         else:
#             # 补零
#             pad_len = self.num_points - all_points.shape[0]
#             all_points = np.concatenate([all_points, np.zeros((pad_len, 3))], axis=0)
#             all_colors = np.concatenate([all_colors, np.zeros((pad_len, 3))], axis=0)

#         # 拼接 XYZ + RGB (归一化到 0-1 的 RGB)
#         point_cloud = np.concatenate([all_points, all_colors], axis=-1).astype(np.float32)
#         return point_cloud

#     def convert_idx(self, idx):
#         # ... (保持大部分原样，除了最后删除 depth 的部分)
#         file = self.file
#         demo = file[f'data/demo_{idx}']
#         states = demo['states'][:]
#         initial_state = dict(states=states[0])
#         initial_state["model"] = demo.attrs["model_file"]

#         obss = self.convert_obs(initial_state, states)
        
#         # 修改 3: 删除不需要的 Depth，但保留 point_cloud
#         # 注意：这里只删除了我们不想保存到 hdf5 的 depth 键
#         keys_to_delete = [k for k in obss.keys() if 'depth' in k]
#         for k in keys_to_delete:
#             del obss[k]
        
#         # 其他多余的图像也可以删掉，视你的 config 而定
#         if 'sideview_image' in obss: del obss['sideview_image']
#         if 'birdview_image' in obss: del obss['birdview_image']

#         return obss