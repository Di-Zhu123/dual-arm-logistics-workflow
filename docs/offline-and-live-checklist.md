# 离线验证与上真机门禁

## 当前可执行的离线验证

核心离线演示只依赖 Python 标准库。完整测试集还需要安装视觉可选依赖：

```powershell
$env:PYTHONPATH='src'
python -m pip install -e ".[vision]"
python -m unittest discover -s tests -v
python scripts/run_offline_demo.py --approve-simulation
```

第一条覆盖三相机完整性/时效/同步、head 外参缺失、计划篡改、碰撞余量、关节限制、一次性批准、过期或错误 token、场景变化、执行失败、验证失败，以及本地 loopback TCP 分片、截断和超长响应。第二条只对内存模拟器执行两次显式的 simulated approval，不访问网络或真机。

不带 `--approve-simulation` 运行 demo 时，它输出第一个待审核计划后安全停止。

## 上真机前必须全部满足

- [ ] 为固定 head camera 完成内参、深度尺度和 `world_from_camera` 标定；给三台相机记录 serial number。
- [ ] 修改后的采集副本保存三路 RGB/depth/intrinsics/时间戳/serial，并做静态场景点云对齐验收。
- [ ] 定义 API v2 schema：request ID、scene ID、API/model/calibration version、单位、坐标系、超时和结构化错误。
- [ ] 检测服务返回 boxes/masks/labels/scores；不能再依赖当前 `ovd`/`seg_open` 歧义。
- [ ] 抓取服务输入只包含目标 mask 内及安全上下文点云，输出 score/宽度/深度/世界姿态；候选逐一做 IK 和碰撞过滤。
- [ ] 仿真规划返回带时间的轨迹、速度/加速度、完整碰撞集合、最小间距和 attached object 结果。
- [ ] 预览 MP4 保存到受控运行目录并计算 SHA-256；UI 展示的 digest 与批准 token 绑定。
- [ ] 抓取与放置分别批准，批准后执行前再次确认 joints/scene/TTL。
- [ ] 放置规划使用箱内自由体积、箱沿和物体尺寸，而非硬编码中心点。
- [ ] 有急停人员在场；先使用假物体、低速、单臂、扩大 clearance 做 shadow/dry-run。
- [ ] 硬件 adapter 只允许执行 plan 中的 waypoints，禁止接受 LLM 直接生成的任意 `movej/movel`。
- [ ] 记录任务、输入帧摘要、标定版本、模型版本、计划摘要、审核人、执行反馈和验证证据。

在这些门禁满足之前，`hardware_execution_enabled` 保持 false。
