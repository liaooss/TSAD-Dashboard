# TSAD Dashboard｜时间序列异常检测平台

基于 PyOD、Streamlit、Pandas 与 Plotly 的多变量监控指标异常分析演示平台。面向 CPU 使用率、内存占用、磁盘 I/O、请求量等时序数据，提供数据质量检查、异常检测、交互式可视化、算法对比和 CSV 导出。

Python **异常检测（离群点检测）工具箱**，做时序异常检测的核心算法库

Streamlit **快速 Web 数据应用框架，不用写 HTML/JS**
> **研究范围说明：** Isolation Forest、LOF 和 KNN 是通用异常检测器，本项目将其应用于滑动窗口特征；Spectral Residual 与 Matrix Profile 是本项目提供的基础实现。它们不应被描述为经过充分验证的生产级专用 TSAD 模型。用于实际运维前，需要结合业务标签、基线和误报成本进行评估。

## 功能模块

### 1. 监控数据上传与预处理
- 上传 CSV，支持 UTF-8/UTF-8 BOM，尝试兼容 GB18030 编码。
- 自动建议时间戳列，并识别原生数值列及可解析为数值的文本列。
- 展示行列数、缺失单元格、完全重复行、列类型、缺失值和唯一值数量。
- 支持保留重复时间戳、按时间戳聚合（数值指标取均值）、删除重复时间戳（保留第一条）。
- 模型输入阶段以中位数填补缺失值；可选择 Z-Score 或 Min-Max 逐指标归一化。
- 可选负值过滤（将所选指标中的负值视为缺失值），支持 1min、5min、10min、30min、1h 时间重采样。
- 可选粘贴 JSON 指标数组；可手动选择时间列、指标列与真实标签列。
- 对空文件、重复列名、无数值指标等情况给出提示。

### 2. 异常检测算法引擎
- **Isolation Forest（滑动窗口）**：PyOD 隔离森林模型。
- **Spectral Residual（频谱残差）**：基于频谱残差和局部偏差的轻量实现。
- **Matrix Profile（矩阵轮廓）**：多变量子序列 z-normalization 后的最近非平凡邻居距离。
- **LOF** 和 **KNN**：保留的附加 PyOD 基线算法。
- 可配置预期异常比例、窗口大小与窗口步长；提供统一的 [0, 1] 异常分数与可调阈值。
- 相同数据和参数的检测结果使用 Streamlit 内存缓存，减少重复计算。

Matrix Profile 当前采用精确两两距离计算，为控制运行时间最多处理 2500 个滑动子序列。较长数据建议先按时间聚合或下采样。

### 3. 时序曲线与异常可视化
- Plotly 支持多指标合并曲线或按指标拆分子图，并在异常点 hover 中展示时间、指标值和异常分数。
- 绘制异常分数及阈值线，支持框选/缩放查看局部时间段。
- 展示各指标对应的异常时间点数量。

### 4. 异常分析看板
- 展示检测样本数、异常点数量、实际异常比例和检测耗时。
- 合并连续异常点为时间区间，展示开始/结束时间、持续样本数与涉及指标。
- 提供 Spike、Drift、Contextual 规则型辅助诊断标签（用于提示，不替代根因分析）。
- 支持运行 Isolation Forest、Spectral Residual 和 Matrix Profile 三算法对比，查看异常点重合度。
- 如果选择了真实异常标签列，可计算 Precision、Recall 与 F1。
- 支持下载完整结果、异常记录、算法对比、参数快照及交互式 HTML 报告。

## 本地运行

推荐使用 Python 3.11。

```bash
# 进入项目根目录
cd TSAD-Dashboard

# 建立独立环境（任选一种）
python -m venv .venv

# Windows CMD
.venv\Scripts\activate

# 安装 PyOD 与 Dashboard 依赖
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dashboard.txt

# 启动应用
python -m streamlit run app.py
```

浏览器打开 Streamlit 输出的本地地址，通常是 `http://localhost:8501`。

如果你已经在 Conda 环境中安装了项目依赖，也可以直接运行：

```bash
conda activate tsad-dashboard
cd /d E:\Projects\TSAD-Dashboard
python -m streamlit run app.py
```

## 示例数据

打开应用后，点击“下载示例 CSV”。示例数据包含时间戳、CPU、内存、磁盘 I/O 和请求量，并注入了少量合成异常，适合用于功能演示，不代表真实生产数据。

## 部署到 Streamlit Community Cloud

1. 将代码推送到 GitHub 仓库，并确保根目录存在 `app.py`、`requirements.txt` 和 `.python-version`。
2. 登录 Streamlit Community Cloud，连接 GitHub。
3. 选择仓库与分支，主文件路径填写 `app.py`。
4. 点击部署并等待依赖安装完成。
5. 如果部署失败，优先检查构建日志中的 Python 版本和依赖冲突。

截图中 Deploy 弹窗的英文选项属于 Streamlit 平台自身界面，应用代码无法直接修改该弹窗的语言。本项目左侧“部署选项说明（中文）”提供了三种选项的中文说明：

- **Streamlit Community Cloud（社区云）**：适合课程项目、个人演示与公开在线应用。
- **Snowflake（企业数据云）**：适合使用 Snowflake 的企业数据场景，普通课程项目通常无需选择。
- **Other platforms（其他部署平台）**：适合自行管理的服务器或其他云平台，需要自行配置环境、网络和安全策略。

## 结果字段

- `anomaly_score`：算法输出的异常分数；不同算法的分数尺度不一定可直接横向比较。
- `is_anomaly`：是否被当前算法标记为异常。
- `anomaly_label`：中文标签“异常”或“正常”。
- 原始指标列：保留输入数据值；模型使用的缺失值填补只作用于内部特征矩阵。

## 项目结构（关键文件）

```text
TSAD-Dashboard/
├── app.py
├── requirements.txt
├── requirements-dashboard.txt
├── .python-version
├── .streamlit/
│   └── config.toml
└── README.md
```

仓库原有 PyOD 源码、文档和测试文件均予以保留。
