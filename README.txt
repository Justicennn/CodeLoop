CodeLoop

1. 项目简介
CodeLoop 是自研本地终端编程 Agent，可修改代码、诊断故障、构建/审查项目和连续交互。它按“决策 → 执行 → 观察”迭代：模型决定下一步，Runtime 执行工具并返回结果，据此继续，非固定流程或单次生成。需求可来自代码、PDF/DOCX、指定文本网页和本地图片；图片交由启用图像输入的兼容模型，不含 OCR、浏览器、搜索或爬虫。

2. Git 仓库
https://github.com/Justicennn/CodeLoop.git

3. 运行方式
Python 3.11，推荐 Conda。依赖由 pyproject.toml 统一管理。

conda create -n codeloop python=3.11 -y
conda activate codeloop
cd /d <CodeLoop源码目录>
python -m pip install .
set MODEL_API_KEY=<API_KEY>
set MODEL_BASE_URL=<API地址>
set MODEL_NAME=<模型名称>
set MODEL_SUPPORTS_IMAGE_INPUT=true

set 仅对当前 CMD 有效，重开终端需重设。启动：

cd /d <目标项目目录>
codeloop

当前目录默认为 Workspace Root。

4. 核心设计与特色
（1）自主执行：单 Agent 迭代调用工具，复杂任务可维护结构化计划，使流程随反馈调整，形成理解、实现与修正闭环。
（2）安全与控制：Runtime 约束路径、命令和权限，Workspace 阻止越界；敏感操作由任务级授权发起澄清、选择或审批，实现明确时自主、关键处用户控制。
（3）上下文与可靠性：按完整工具周期裁剪上下文，保持协议完整；跨任务仅留有界公开对话，据真实工具和命令结果形成验证证据，兼顾稳定与按需验证。
（4）分层与解耦：工具统一注册，新增工具不改 Agent 核心循环；模型与交互接口隔离，模型和交互方式可替换；运行时事件与终端展示分离，可观测性不侵入核心逻辑。

5. 原则与扩展性
聚焦终端和本地工作区；工具、模型、交互边界独立，可不改 Agent 核心循环而扩展。模型判断语义，Runtime 执行规则与安全约束。
