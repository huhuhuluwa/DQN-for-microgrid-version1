"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np
import pandas as pd
import random

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer

Actions = [-1, -0.5, 0, 0.5, 1]

class Battery:
    #simulates the battery system of the microgrid
    def __init__(self, capacity, Pmax, dissipation, E):
        self.capacity = capacity #电池容量MWh
        self.Pmax = Pmax #电池充放电最大功率MW
        # self.dischargeC = dischargeC #放电效率
        # self.chargeC = chargeC #充电效率
        self.disspation = dissipation #耗散系数
        self.E = E #电池电量kWh

    def reset(self, E):
        self.E = E

    def charge(self, action):
        # if self.SoC < 0.01:
        #     self.E = 0
        self.E -= Actions[action] * self.Pmax

    def get_E(self):
        return self.E
    
    def get_P(self):
        return self.Pmax
    
    @property
    def SoC(self):
        return self.E/self.capacity

class WindPower:
    def __init__(self):
        power_pd = pd.read_csv('wind.csv',usecols=[0,1])
        self.power = np.array(power_pd)
    
    def CurrentGeneration(self,time):
        return self.power[time,1]

class PhotoVoltage:
    def __init__(self):
        power_pd = pd.read_csv('distributed-pv-2021.csv',usecols=[2,3])
        self.power = np.array(power_pd)
    
    def CurrentGeneration(self,time):
        return self.power[time,1]

class Load:
    def __init__(self):
        load_df = pd.read_csv('operational-demand-2021.csv',usecols=[2,3])
        self.Load = np.array(load_df)

    def currentload(self,time):
        return self.Load[time,1]

class Grid:
    def __init__(self):
        GridPrice = pd.read_csv('stem-summary-2021.csv',usecols=[2,7])
        self.Price = np.array(GridPrice)

    def price(self,time):
        return self.Price[time,1]


class MicroGrid(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    #该类使用了类型注解来指定输入和输出的数据类型
    #接受一个np.ndarray类型的观测值，并返回一个Union[int, np.ndarray]类型的动作。
    #其中，动作可以是一个整数，也可以是一个np.ndarray类型的向量。
    """
    ### Description
    微电网调度问题通过控制储能设备的充放电行为，达到微电网成本最低

    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` 
    表示电池的充放电功率
    | Num | Action                  |
    |-----|-------------------------|
    |-1   | Pmax 放电               |
    |-0.5 | 0.5 * Pmax 放电         |
    | 0   | 不充电不放电             |
    | 0.5 | 0.5 * Pmax 充电         |
    | 1   | Pmax 充电               |
    **Note**: k+1时刻的电池电量E取决于k时刻的电池电量和采取的动作

    ### Observation Space
    The observation is a `ndarray` with shape `(6,)` 
    电池电量 + 风电 + 光伏 + 负荷 + 电价 
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 1   | 电池电量E              | 0                   | 13600 kwh         |
    | 2   | 风电功率               | 0                   | 1763  kw          |
    | 3   | 光伏功率               | 0                   | 1560  kw          |
    | 4   | 负荷功率               | 700                 | 3869  kw          |
    | 5   | 电价                   | -175                | 138  $/kwh        |

    ### Rewards
    每小时用电成本

    ### Starting State
    从历史数据随机选择

    ### Episode End
    终止条件：
        1、储能设备不能满足当前选择的设备
        2、每一回合运行步数超过1000
    """

    def __init__(self, render_mode: Optional[str] = None):#对象创建时可以传入一个可选的渲染模式（render_mode），它的值可以是一个字符串。如果未提供render_mode参数，它的值将为None。
        self.battery = Battery(capacity=8000, Pmax=1000, dissipation=0.001, E=4000)
        self.BatPmax = 1000
        self.wind = WindPower()
        self.pv = PhotoVoltage()
        self.Load = Load()
        self.grid = Grid()
        self.time = 0
        self.SocMin = 0
        self.SocMax = 1.01
        self.end = 0
        # 电池SOC上下限为10%，90%
        low = np.array([-100 ,self.SocMin, 0, 0, 0, -1000],dtype=np.float32)
        high = np.array([100 ,self.SocMax, 2000, 2000, 4000, 1000],dtype=np.float32)
        self.action_space = spaces.Discrete(5) #0-4
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.render_mode = render_mode
        self.state = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False
    ):
        super().reset(seed=seed)
        # self.time = 0       
        self.time = random.randint(0,3)*48
        self.end = self.time + 48
        # print("time:",self.time)
        self.battery.reset(E=4000)
        BatSoc = self.battery.SoC
        Wind = self.wind.CurrentGeneration(self.time)
        pv = self.pv.CurrentGeneration(self.time)
        load = self.Load.currentload(self.time)
        price = self.grid.price(self.time)
        self.state = (self.time%48, BatSoc, Wind, pv, load, price)

        # self.renderer.reset()
        # self.renderer.render_step()
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
        
    def step(self, action):
        # err_msg = f"{action!r} ({type(action)}) invalid" #判断action是否为有效类型
        # assert self.action_space.contains(action), err_msg #用于确保action在可接受的动作空间中
        # assert self.state is not None, "Call reset before using step method."
        self.battery.charge(action)
        self.time += 1
        # print("time",self.time)
        BatSoc = self.battery.SoC
        # print(BatSoc)
        Wind = self.wind.CurrentGeneration(self.time)
        pv = self.pv.CurrentGeneration(self.time)
        load = self.Load.currentload(self.time)
        price = self.grid.price(self.time)
        self.state = (self.time%48, BatSoc, Wind, pv, load, price)
        if self.time < self.end:
            terminated = bool( self.battery.SoC < 0 or self.battery.SoC > 1)
        else:
            terminated = True

        if not terminated:
                reward = (Wind + pv + Actions[action] * self.BatPmax - load) * price#千刀
        elif terminated:
            if self.time < self.end: # 说明没有完成48个时刻的调度
                reward = -1e5
            else:                    # 说明完成了48个时刻的调度
                reward = 5e5
        # self.renderer.render_step()
        return np.array(self.state, dtype=np.float32), reward, terminated

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode="human"):
        return 

    def close(self):
        """ 
        Nothing to be done here, but has to be defined 
        """
        return

    def seed(self, seed=None):
        np.random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return self._np_random

# 环境测试
# env = MicroGrid()
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# print(state_dim, action_dim)
# state = env.reset()
# print("state:",state)
# for i in range(4):
#     state = env.step(0.5)
#     print("state:",i,state)