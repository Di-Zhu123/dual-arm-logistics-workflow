# 三相机标定工具

该目录提供固定头部相机的 eye-to-hand 标定、左右腕部相机的 eye-in-hand 标定、内参健康检查以及三路点云对齐验证。所有采集工具保持只读，不发送机械臂或夹爪动作命令。

## 准备

1. 将 `config/camera_identity.example.json` 复制为仓库外的本地配置。
2. 填写 head、left、right 三台相机的真实序列号；USB 枚举编号只能用于观察，不能作为设备身份。
3. 按 `config/calibration_board.example.json` 核对标定板尺寸。
4. 准备机器人数据采集工程路径与其本地环境配置。

先运行离线自检：

```bash
PYTHONPATH=src python -m calibration.self_test
```

检查相机身份与流配置：

```bash
python -m calibration.inspect_realsense
python -m calibration.preview_three_cameras --host 127.0.0.1 --port 65090
```

## 数据采集

采集入口支持固定标定板和手持标定板两类流程。实际参数以 `--help` 为准：

```bash
python -m calibration.capture_dataset --help
```

建议将采集结果写入仓库外的独立目录，并在每轮采集后保存：

- 三路 RGB 与深度图；
- 相机内参、深度尺度、时间戳和序列号；
- 机器人关节状态与末端位姿；
- 标定板检测结果和本轮操作备注。

## 求解与验证

```bash
python -m calibration.solve_head_eye_to_hand --help
python -m calibration.solve_wrist_eye_in_hand_via_head --help
python -m calibration.solve_wrist_fixed_board --help
python -m calibration.validate_intrinsics --help
python -m calibration.validate_three_camera_alignment --help
```

上线前应同时满足重投影误差、跨相机点云对齐、静态场景时间同步和设备序列号检查。任何一路相机缺少深度、内参或世界外参时，该帧都不能参与三维融合。

