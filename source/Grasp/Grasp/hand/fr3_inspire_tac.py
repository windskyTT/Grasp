import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

# 定义 Franka FR3 + Inspire 手的配置
FR3_INSPIRE_TAC_CFG = ArticulationCfg(
    # 1. 资产加载 (Asset Loading)
    # 推荐在 Isaac Lab 中将 URDF 提前转为 USD，但也可以直接加载 URDF
    spawn=sim_utils.UsdFileCfg(
        asset_path="inspire_tac/fr3_inspire_tac_L_right_safety.usd",
        # 如果你已经转换成了 USD，请使用 UsdFileCfg:
        # asset_path="inspire_tac/fr3_inspire_tac_L_right_safety.usd",
        make_instanceable=True,
    ),
    
    # 2. 初始状态 (Initial State)
    # 将 default_dof_pos 映射到具体的关节名称
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), # 机器人基座的世界坐标位置
        joint_pos={
            # 机械臂初始位置
            "fr3_joint1": 0.0,
            "fr3_joint2": 0.0,
            "fr3_joint3": 0.0,
            "fr3_joint4": -1.6,
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.6,
            "fr3_joint7": 0.0,
            # 灵巧手初始位置 (利用正则表达式匹配所有 right_ 开头的关节设为 0)
            "right_.*_joint": 0.0, 
        },
    ),
    
    eef_link="fr3_link8",

    # 3. 驱动器配置 (Actuators)
    # 将 numActions (13) 拆分为机械臂(7)和手(6)的控制组
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-7]"],
            effort_limit=87.0, # 根据实际电机修改
            velocity_limit=2.175,
            stiffness=400.0, # PD 控制的 P 参数
            damping=40.0,    # PD 控制的 D 参数
        ),
        "hand": ImplicitActuatorCfg(
            # 对应原 YAML 中的 hardware_active_hand_dof_names
            joint_names_expr=[
                "right_little_1_joint", 
                "right_ring_1_joint", 
                "right_middle_1_joint", 
                "right_index_1_joint", 
                "right_thumb_2_joint", 
                "right_thumb_1_joint"
            ],
            stiffness=100.0,
            damping=10.0,
        ),
    },
)