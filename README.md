# HUGSIM 编辑与闭环仿真接入平台

将场景编辑、观测生成、冻结 UniAD 推理、HUGSIM 运动学推进、原生评分和结果展示接入统一任务接口。当前可用 JSON 模拟 LLM 的工具调用；具体模块的可调度范围和限制以注册信息及验收记录为准。

**从 [统一平台使用说明](integrations/scene-platform/README.md) 开始。** 可选扫描模块见 [SENSORS.md](integrations/scene-platform/SENSORS.md)。

## 目录

| 路径 | 职责 |
| --- | --- |
| `code/HUGSIM/` | HUGSIM 仿真环境、车辆运动学、跟踪与原生评分；作为外层仓库的普通源码追踪，上游 Git 元数据单独备份 |
| `code/UniAD_SIM/`、`code/NAVSIM/` | 驾驶模型及相关评测接入代码 |
| `integrations/scene-platform/` | 统一任务协议、工具调度、模拟 LLM、编辑与驾驶生命周期、结果回执 |
| `integrations/chain-repair/` | 从历史实验目录复制出来的运行实现和源码配置，保留 `phase*/…` 层级以兼容既有版本；原实验文件保留 |
| `integrations/_shared/` | 仓库根目录、源码与外部数据路径适配、Shell 环境入口 |
| `integrations/sr-hugsim/` 等 | 渲染、感知及仿真后端适配 |
| `integrations/novel-view-*/` | 新自车轨迹、编辑视角、伪 GT、离线蒸馏及展示 |
| `integrations/light-*/`、`integrations/*patch*/` | 分版本驾驶与外部补丁研究实现；版本号不代表已验收的默认方案 |
| `waymo23-native-import/` | 原生数据、传感器和运动学接入公共依赖 |
| `tools/` | 本地环境工具与开源目录迁移工具 |
| `outputs/` | 本地实验、模型中间产物、日志、视频和冻结证据；不是源码发布位置 |

## 本地路径与外部资源

源码不要求固定安装目录。默认根目录由 `integrations/_shared/paths.py` 所在位置确定，可用以下环境变量覆盖：

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `HUGSIM_ROOT` | 当前仓库根目录 | 仓库搬迁或显式部署根目录 |
| `HUGSIM_CHAIN_CODE_ROOT` | `$HUGSIM_ROOT/integrations/chain-repair` | 已迁移的 phase 实现 |
| `HUGSIM_OUTPUT_ROOT` | `$HUGSIM_ROOT/outputs` | 实验数据与运行输出 |
| `HUGSIM_EXTERNAL_ROOT` | `$HUGSIM_ROOT/assets/external` | 不在仓库内的原始数据、模型和其他研究项目 |
| `HUGSIM_PYTHON` | 已存在的 HUGSIM Pixi Python，否则 `python3` | 运行解释器 |

Shell 使用 `source integrations/_shared/env.sh`。JSON 中的 `${HUGSIM_ROOT}` 等路径令牌通过共享配置读取边界展开；OmegaConf YAML 使用 `${oc.env:HUGSIM_ROOT,.}` 等标准变量表达式。旧实验清单按原样保存，读取时支持其历史路径格式；不会改写证据文件或绕过哈希检查。

历史 phase 同一目录中混放实现和数据。迁移模块显式使用共享 `Path` 兼容类：优先定位 `integrations/chain-repair` 中的实现，源码旁缺失的数据回退到 `HUGSIM_OUTPUT_ROOT`。该类不全局修改 Python 的 `pathlib`。这使旧 phase 布局可继续定位依赖，同时保持源码与实验资产分离。训练数据、检查点、SMPL 资源、编译扩展和冻结实验发布包仍需按各模块说明单独提供；目录整理不包含模型或数据发布，也不保证上游依赖已安装。

## 最小检查

从仓库根目录运行，不启动 GPU 或驾驶：

```bash
python3 -B -m integrations._shared.paths --help
python3 -B integrations/scene-platform/cli.py --help
python3 -B integrations/scene-platform/cli.py capabilities
```

实际 GPU 运行及依赖安装参见各模块文档。历史任务绑定了代码与配置哈希；本次迁移后的源码不能冒充旧版本继续通过冻结核验。旧任务及证据保留，迁移版本需创建新任务、重新做对应准入，不能重签旧实验来声称已完成 GPU 回归。

## 发布范围

本仓库发布源码及运行配置，不包含本地环境、权重、数据集或完整实验输出。上游源码作为普通目录保存，并保留各自许可；本地 Git 备份、迁移审计和临时预检记录不进入发布包。

### 单独获取的模型和程序

以下文件由 `.gitignore` 排除，不随源码上传；本机已有完整文件继续保留。先取得与各模型配置兼容、许可允许使用的资源，再放到表中路径：

| 资源 | 获取方式 | 仓库内存放位置 |
| --- | --- | --- |
| UniAD stage-2 权重 | [UniAD v1.0.1 官方发布](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth)，亦见本地 [INSTALL.md](code/UniAD_SIM/docs/INSTALL.md) | `code/UniAD_SIM/ckpts/uniad_base_e2e.pth` |
| VAD-Base 权重 | 从 [VAD 官方 Models 表](https://github.com/hustvl/VAD#models) 的 VAD-Base 下载项取得，按本地配置重命名 | `code/VAD_SIM/ckpts/VAD_base.pth` |
| STF 相机提取程序（重建侧） | 从原数据工具维护方取得获准分发、与运行环境兼容的 `dump_camera_image_virtual_pkg` 发布包；本仓库没有该二进制的构建源码或已核实的公共下载地址 | `code/scene_reconstruction/third_party/StfTools/dump_camera_image_virtual_pkg/dump_camera_image_virtual` |
| 同一 STF 程序（独立接入侧） | 同上，可从本机已有合法发布包复制；按需保留包内 `libs/` 与配置 | `integrations/sr-origin-main/third_party/StfTools/dump_camera_image_virtual_pkg/dump_camera_image_virtual` |

UniAD 权重可在仓库根目录下载（只在用户主动执行时下载）：

```bash
mkdir -p code/UniAD_SIM/ckpts
curl -fL --retry 3 \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth \
  -o code/UniAD_SIM/ckpts/uniad_base_e2e.pth
```

STF 二进制只供对应 STF 数据预处理使用；已准备好的 Waymo 551 平台任务不依赖它。程序有合法来源并放妥后，可通过同目录的 `dump_camera_image_virtual.sh` 启动。不要将 `.part` 下载残片当作权重使用。

真实 API 配置仅保存在忽略的 `.env` 或本地私有配置中，源码只读取环境变量；示例见 [.env.example](.env.example)。不要把 API Key、云端 AK/SK 或私钥填入示例后提交。根忽略规则覆盖环境目录、私钥、常见私有配置、模型权重、上述预编译程序及会话记录。

遵循各上游项目和第三方文件已有许可；本整理没有替这些项目新增或变更许可。

可发布版本的源码归档边界记录在 `integrations/_shared/archived_sources.json`，只包含相对路径，不依赖本机整理报告。`docs/open-source/` 是维护者本地审计目录，已忽略且不是运行前提。仓库包含若干本地维护工具；迁移/审计命令需要对应的本地历史记录，日常运行从统一平台入口开始。
