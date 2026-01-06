import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import gstaichi as ti


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False,show_camera=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.contact_links = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        # add terrain type either plane or noisy
        # noisy terrain
        if self.env_cfg.get("terrain_type") == "active_noisy":
             self.terrain = self.scene.add_entity(
                gs.morphs.Terrain(
                    n_subterrains=(3, 3),
                    subterrain_size=(10, 10),
                    pos=(-15, -15, 0),
                    horizontal_scale=0.25,
                    vertical_scale=0.005 ,
                    subterrain_types="random_uniform_terrain",
                    randomize=True
                )
             )
        else:
             self.terrain = self.scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0), fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                links_to_keep=self.contact_links,
            ),
        )
        self.contact_sensors = []
        for link in self.contact_links:
            if link in self.robot.links:
                self.contact_sensors.append(self.scene.add_sensor(gs.sensors.ContactForce(
                    entity_idx=self.robot.idx,
                    link_idx_local=self.robot.get_link(link).idx_local,
                    draw_debug=True,
                )))

        # random forces
        self.push_interval_steps = int(self.env_cfg.get("push_interval_s", 5.0) / self.dt)
        self.push_vel_range = self.env_cfg.get("push_vel_range", [-1.0, 1.0])
        self.lin_vel_shift_range = self.env_cfg.get("lin_vel_shift_range", [-0.1, 0.1])
        self.ang_vel_shift_range = self.env_cfg.get("ang_vel_shift_range", [-0.2, 0.2])

        # add camera
        if show_camera:
            self.cam_0 = self.scene.add_camera(
                res=(640, 480),
                pos=(3.0, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=60,
                GUI=False
            )
        else:
            self.cam_0 = None
        # build
        self.scene.build(n_envs=num_envs)

        if self.cam_0 is not None:
            self.cam_0.follow_entity(self.robot, smoothing=0.05)


        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel_shift = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel_shift = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["height_range"], (len(envs_idx),), self.device)

    def sample_contacts(self, envs_idx=None) -> torch.Tensor:
        """
        Sample contact information from all contact sensors across environments.
        
        Parameters
        ----------
        envs_idx : int, list, or None, optional
            Environment indices to query. If None, queries all environments.
            Defaults to None.
        
        Returns
        -------
        torch.Tensor, shape (n_envs, n_links)
            Boolean tensor indicating contact for each link in each environment.
            True means the link is in contact, False means no contact.
            Order matches self.contact_links: [FR_foot, FL_foot, RR_foot, RL_foot]
        """
        if len(self.contact_sensors) == 0:
            # No contact sensors, return zeros
            n_envs = self.num_envs if envs_idx is None else len(envs_idx) if isinstance(envs_idx, (list, tuple)) else 1
            return torch.zeros((n_envs, len(self.contact_links)), device=self.device, dtype=gs.tc_float)
        
        # Read from all contact sensors
        # Each sensor.read() returns shape (n_envs,) when envs_idx=None
        contact_data = []
        for sensor in self.contact_sensors:
            contact = sensor.read(envs_idx=envs_idx)  # Shape: (n_envs,) or scalar if single env
            # Ensure it's a tensor with proper shape
            if not isinstance(contact, torch.Tensor):
                contact = torch.tensor(contact, device=self.device, dtype=gs.tc_float) * 1.0
            # Handle single environment case
            if contact.dim() == 0:
                contact = contact.unsqueeze(0)
            contact_data.append(contact)
        
        # Stack along the links dimension: (n_envs, n_links)
        contacts = torch.stack(contact_data, dim=-1)  # Shape: (n_envs, n_links)
        
        return contacts


    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        # Sample contact information (optional - can be called when needed)
        # self.contact_buffers = self.sample_contacts()

        # random pushes
        if self.episode_length_buf[0] % self.push_interval_steps == 0:
             self._apply_random_forces()

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                (self.base_ang_vel + self.base_ang_vel_shift) * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 4
                
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self._resample_shifts(envs_idx)

    def _resample_shifts(self, envs_idx):
        self.base_lin_vel_shift[envs_idx] = gs_rand_float(*self.lin_vel_shift_range, (len(envs_idx), 3), self.device)
        self.base_ang_vel_shift[envs_idx] = gs_rand_float(*self.ang_vel_shift_range, (len(envs_idx), 3), self.device)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _apply_random_forces(self):
        random_vel = torch.rand((self.num_envs, 3), device=self.device) * (self.push_vel_range[1] - self.push_vel_range[0]) + self.push_vel_range[0]
        random_vel[:, 2] = 0 # keep z velocity zero for now
        current_vel = self.robot.get_vel()
        self.robot.set_vel(current_vel + random_vel)

    def query_terrain_heights(self, x_min, x_max, y_min, y_max, voxel_size=0.1):
        """
        Query terrain heights over a rectangular region.
        
        This function samples the terrain height field over a specified rectangular
        region using bilinear interpolation. If no terrain exists (e.g., using a
        flat plane), it returns zeros.
        
        Parameters
        ----------
        x_min, x_max : float
            X-axis bounds of the query region in world coordinates (meters)
        y_min, y_max : float
            Y-axis bounds of the query region in world coordinates (meters)
        voxel_size : float, optional
            Size of each voxel in meters. If None, uses the terrain's horizontal_scale.
            If specified, the region will be sampled at this resolution.
        
        Returns
        -------
        heights : torch.Tensor, shape (nx, ny)
            Height values in world coordinates (meters) for each voxel in the region
        x_coords : torch.Tensor, shape (nx,)
            X coordinates of each voxel center
        y_coords : torch.Tensor, shape (ny,)
            Y coordinates of each voxel center
        
        Example
        -------
        >>> # Query a 2m x 2m region around the origin with 0.1m voxels
        >>> heights, x_coords, y_coords = env.query_terrain_heights(
        ...     x_min=-1.0, x_max=1.0,
        ...     y_min=-1.0, y_max=1.0,
        ...     voxel_size=0.1
        ... )
        >>> # heights[i, j] gives the height at position (x_coords[i], y_coords[j])
        """
        if not hasattr(self, 'terrain') or self.terrain is None:
            # If no terrain, return flat plane at z=0
            if voxel_size is None:
                voxel_size = 0.25
            nx = int((x_max - x_min) / voxel_size) + 1
            ny = int((y_max - y_min) / voxel_size) + 1
            x_coords = torch.linspace(x_min, x_max, nx, device=self.device)
            y_coords = torch.linspace(y_min, y_max, ny, device=self.device)
            heights = torch.zeros((nx, ny), device=self.device)
            return heights, x_coords, y_coords
        
        # Get height field and metadata from terrain
        terrain_geom = self.terrain.geoms[0]
        height_field = terrain_geom.metadata["height_field"]
        horizontal_scale = terrain_geom.metadata["horizontal_scale"]
        vertical_scale = terrain_geom.metadata.get("vertical_scale", 0.005)
        
        # Get terrain position (offset)
        # Try to get from morph, otherwise try entity position, otherwise default to (0, 0, 0)
        try:
            if hasattr(self.terrain, 'morph') and hasattr(self.terrain.morph, 'pos'):
                terrain_pos = torch.tensor(self.terrain.morph.pos, device=self.device)
            elif hasattr(self.terrain, '_morph') and hasattr(self.terrain._morph, 'pos'):
                terrain_pos = torch.tensor(self.terrain._morph.pos, device=self.device)
            else:
                # Try to get from entity position
                terrain_pos = self.terrain.get_pos()[0] if hasattr(self.terrain, 'get_pos') else torch.tensor([0.0, 0.0, 0.0], device=self.device)
        except:
            terrain_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        terrain_x_offset = terrain_pos[0]
        terrain_y_offset = terrain_pos[1]
        
        # Use voxel_size if provided, otherwise use terrain's horizontal_scale
        if voxel_size is None:
            voxel_size = horizontal_scale
        
        # Create query grid
        nx = int((x_max - x_min) / voxel_size) + 1
        ny = int((y_max - y_min) / voxel_size) + 1
        x_coords = torch.linspace(x_min, x_max, nx, device=self.device)
        y_coords = torch.linspace(y_min, y_max, ny, device=self.device)
        
        # Convert world coordinates to height field indices
        # Height field indices are relative to terrain position
        x_indices = ((x_coords - terrain_x_offset) / horizontal_scale).clamp(0, height_field.shape[0] - 1)
        y_indices = ((y_coords - terrain_y_offset) / horizontal_scale).clamp(0, height_field.shape[1] - 1)
        
        # Create meshgrid for interpolation
        x_idx_grid, y_idx_grid = torch.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Convert to integer indices for indexing
        x_idx_floor = x_idx_grid.floor().long()
        y_idx_floor = y_idx_grid.floor().long()
        x_idx_ceil = (x_idx_floor + 1).clamp(0, height_field.shape[0] - 1)
        y_idx_ceil = (y_idx_floor + 1).clamp(0, height_field.shape[1] - 1)
        
        # Get fractional parts for bilinear interpolation
        x_frac = x_idx_grid - x_idx_floor.float()
        y_frac = y_idx_grid - y_idx_floor.float()
        
        # Convert height field to tensor if needed
        if isinstance(height_field, np.ndarray):
            height_field_tensor = torch.tensor(height_field, dtype=gs.tc_float, device=self.device)
        else:
            height_field_tensor = height_field.to(self.device)
        
        # Bilinear interpolation
        h00 = height_field_tensor[x_idx_floor, y_idx_floor]
        h10 = height_field_tensor[x_idx_ceil, y_idx_floor]
        h01 = height_field_tensor[x_idx_floor, y_idx_ceil]
        h11 = height_field_tensor[x_idx_ceil, y_idx_ceil]
        
        # Interpolate
        h0 = h00 * (1 - x_frac) + h10 * x_frac
        h1 = h01 * (1 - x_frac) + h11 * x_frac
        heights = h0 * (1 - y_frac) + h1 * y_frac
        
        # Convert from height field units to world coordinates
        heights = heights * vertical_scale
        
        return heights, x_coords, y_coords
