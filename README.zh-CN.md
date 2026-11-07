# TacSE3 · 触觉演示

实时查看触觉图像、接触深度、剪切响应和各传感器的三维反馈。
适合通过手转小球观察接触与局部运动响应的原型演示。

[English](README.md) · [使用指南](docs/ball-demo.md)

## 启动

在桌面终端运行，并预先安装厂商提供的 `dmrobotics` SDK。

```bash
pip install -e ".[viz]"
python -m tacse3.ball_demo --sensor-serial SERIAL1 SERIAL2 --pose3d on
```

将序列号替换为实际设备。先让传感器空载完成初始化，再把小球放在固定的
传感器表面之间，轻缓转动。只需要二维界面时，安装 `.[demo]` 并使用 `--pose3d off`。

| 按键 | 操作 |
| --- | --- |
| Z | 显示归零 |
| R | 切换原始 / 平滑显示 |
| S | 保存快照 |
| Q / Esc | 退出 |

快捷键在二维窗口中生效。三维窗口的灰色坐标轴为参考，彩色坐标轴表示当前
局部响应。采集记录保存在 `runs/` 下。

显示数值采用标称传感器标定，表示局部响应，不代表物体的绝对或累计位姿。
具体单位见[单位与标定说明](UNIT_CALIBRATION.md)。

## 更多

- [显示参数与记录格式](docs/ball-demo.md)
- [原有监控与机器人接口](docs/legacy-usage.md)
- [架构](docs/architecture.md) · [开发说明](CONTRIBUTING.md)

基于 [yuanzero/tacSE3_code](https://github.com/yuanzero/tacSE3_code)。
采用 [Apache-2.0](LICENSE) 许可证；厂商 SDK 和机器人驱动需独立安装。
