# 历史 chain_repair 运行实现

这里是从原实验树复制出来的实现、源码配置与许可声明。保留 `phase*/…` 的原有层级，用于固定版本、动态 import、worker 和渲染器快照的兼容定位。原始 `outputs/.../chain_repair/` 未删除、移动或改写。

发布版保留公开入口可达的实现和兼容模块。已排除的历史源码路径记录在 [archived_sources.json](../_shared/archived_sources.json)，共享路径适配器禁止静默回退加载旧实验目录中的这些实现。该运行配置只含相对路径，不依赖本地审计记录。

- 路径与外部资源配置：仓库顶层 [README.md](../../README.md)。
- 平台入口：[`integrations/scene-platform/README.md`](../scene-platform/README.md)。

这里只迁移实现和小型源码配置，不包含训练数据、检查点、原生运行时二进制、GPU 缓存、视频和实验评分。需要这些数据的模块使用 `HUGSIM_OUTPUT_ROOT` 或 `HUGSIM_EXTERNAL_ROOT`；新源码不能重用旧代码哈希冒充旧实验版本。当前仅完成 CPU 迁移验证，未做 GPU 闭环回归。

模块中的共享 `Path` 适配器将历史逻辑路径映射到本目录；源码旁缺少的数据仍从外部实验树读取。它不修改全局 `pathlib`，也不更改驾驶和评分公式。许可文件保留各自上游声明，不能据此推断所有第三方实现可适用同一新许可。
