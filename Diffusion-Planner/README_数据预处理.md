# Diffusion-Planner 预备数据生产说明

## 使用的代码

- **入口脚本**：`data_process.py`（主程序）
- **启动脚本**：`data_process.sh`（封装路径与参数，推荐用这个跑）

预处理会从 nuPlan 原始数据库（`.db`）读场景，生成模型训练用的 `.npz` 文件，并写出 `diffusion_planner_training.json`（npz 文件名列表）。

## 前置条件

1. **环境**：已安装 nuplan-devkit 和 Diffusion-Planner（conda 环境内 `pip install -e .` 两个项目）。
2. **原始数据**：
   - `data_path`：nuPlan 原始数据根目录（其下或子目录中有 `.db` 文件），例如 nuplan-v1.1 的 trainval 或 data 目录。
   - `map_path`：nuPlan 地图根目录（含各城市 `.gpkg` 等），例如 `nuplan-maps-v1.0` 解压后的目录或 `nuplan/dataset/maps`。
3. **场景列表**：当前实现会**固定读取工作目录下的 `nuplan_train.json`**（log 名称列表）。仓库里已带一份；若要做 mini/子集，需自行准备或改代码指向别的 json。

## 在 Linux 下运行

### 方式一：用 shell 脚本（推荐）

1. 进入项目根目录（**必须在 Diffusion-Planner 目录下执行**，因为会读 `./nuplan_train.json`、写 `./diffusion_planner_training.json`）：

   ```bash
   cd /home/xzl/diffusion_planner_test/Diffusion-Planner
   ```

2. 编辑 `data_process.sh`，把下面三个变量改成你本机的路径：

   - `NUPLAN_DATA_PATH`：nuPlan 原始数据根目录（包含 .db 的 trainval 或 data 的父路径）
   - `NUPLAN_MAP_PATH`：地图根目录
   - `TRAIN_SET_PATH`：预处理结果要保存的目录（将生成大量 .npz 和后续的 diffusion_planner_training.json 会引用这些文件）

3. 给脚本执行权限并运行：

   ```bash
   chmod +x data_process.sh
   ./data_process.sh
   ```

   脚本里默认会跑 `--total_scenarios 1000000`，可按需在 `data_process.sh` 里改小（例如 `--total_scenarios 10000`）做小规模测试。

### 方式二：直接调 Python

在 **Diffusion-Planner 目录**下执行（保证 `./nuplan_train.json` 存在）：

```bash
cd /home/xzl/diffusion_planner_test/Diffusion-Planner

python data_process.py \
  --data_path /path/to/nuplan-v1.1/trainval \
  --map_path /path/to/nuplan-maps-v1.0 \
  --save_path ./data/my_train_set \
  --total_scenarios 1000000
```

- `--data_path`：nuPlan 原始数据根目录  
- `--map_path`：地图根目录  
- `--save_path`：输出 .npz 的目录（建议用绝对路径或相对 Diffusion-Planner 的路径）  
- `--total_scenarios`：要处理的场景数量上限  

其他可选参数见 `data_process.py`（如 `--agent_num`、`--lane_num`、`--shuffle_scenarios` 等），一般不改即可。

## 输出结果

- **目录 `save_path`**：大量 `{map_name}_{token}.npz` 文件（每个场景一个）。
- **当前目录下**：`diffusion_planner_training.json`，内容为所有生成的 .npz 文件名列表，训练时用 `--train_set_list` 指向该 json、`--train_set` 指向 `save_path` 即可。

## 若使用本仓库自带的 nuplan 数据路径示例

若 nuPlan 数据和地图在 `nuplan-devkit/nuplan/dataset` 下，可把 `data_process.sh` 设为类似：

```bash
NUPLAN_DATA_PATH="/home/xzl/diffusion_planner_test/nuplan-devkit/nuplan/dataset/nuplan-v1.1"
NUPLAN_MAP_PATH="/home/xzl/diffusion_planner_test/nuplan-devkit/nuplan/dataset/maps"
TRAIN_SET_PATH="/home/xzl/diffusion_planner_test/Diffusion-Planner/data/my_train_set"
```

注意：`nuplan_train.json` 里的 log 名必须在你提供的 `data_path` 下能找到对应 .db，否则会过滤掉或报错；若只用 mini/部分数据，可先减小 `--total_scenarios` 或准备一份只含已有 log 的 json。
