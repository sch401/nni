# Python 3.14 / PyTorch fork 说明

此 fork 包含本次 NNI 生命周期修复、Python 3.14 适配和直接依赖升级，属于非官方修改。代码基线为 `767ed7f22e1e588ce76cbbecb6c6a4a76a309805`。

## 范围与修复

- Windows 本地 PyTorch 超参数优化，实验并发设置为 4。
- dispatcher 消息串行处理、重复结束/失败及迟到指标处理。
- Evolution 并发计数、等待任务补发，TPE/PBT 的状态处理。
- NoMoreTrialJobs 参数覆盖、trial 参数分配、导出参数与指标配对。
- Windows 子进程树结束、取消状态、filelock 4 兼容和 registry 解锁。
- Hyperopt/GP 的 NumPy 2 兼容、零值结果保留，以及 SMAC 现代公开 API 迁移。
- Python 3.14 打包、WebSocket、Express 5、React 19 和 TypeScript 7 适配。

## 安装与构建

使用 Python 3.14 x64 新建环境。已有 Python 3.10 环境不能就地更换解释器。

```powershell
py -3.14 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -r dependencies/setup.txt
python -m pip install -r dependencies/required.txt -r dependencies/pytorch.txt
```

已有本地 whl 时可直接安装它。仓库只保存源码，未上传 whl、Node 二进制、构建目录或实验数据。不要使用 PyPI 的 `pip install nni` 来获取本 fork 的修改。

从源码构建的命令为：

```powershell
$env:NNI_RELEASE = '3.0.1'
python setup.py build_ts
python setup.py bdist_wheel -p win_amd64
```

TypeScript 构建会下载 Node 及 npm 依赖，需联网。已交付的本地 whl 采用构建后的 manager/Web UI 打包并完成验证；全新克隆仓库的完整构建流程尚未重新验证。

`dependencies/pytorch.txt` 是本地 PyTorch HPO 的安装清单；无需安装包含其他框架和示例依赖的 `dependencies/recommended.txt`。如果实际使用 SMAC、Anneal 等可选 tuner，另行安装对应 extra。

## 已有验证结果

- 最终本地 whl：Python 3.14.5、PyTorch 2.14.1+cpu，Evolution 4 并发，60 个实际训练 trial 全部成功。
- 60 条导出结果与参数逐条对应，错配 0；实验启动和停止完成。
- Python 回归检查 36 项通过，存在 2012 条数值/弃用警告。
- Node 回归检查 4 项通过，包括参数分配、迟到指标、取消状态和 Windows 子进程树终止。
- manager TypeScript 编译与 Web UI 生产构建通过；浏览器成功显示实验、指标和 trial 列表。生产构建存在包体积警告。
- 独立验证环境的 78 个已安装包通过依赖一致性检查。

可复现的 4 并发 PyTorch 检查脚本位于 `test/smoke/pytorch_local/smoke_4.py`。用已安装本 fork 的新环境运行：

```powershell
python test/smoke/pytorch_local/smoke_4.py
```

脚本会创建测试实验，检查 60 个任务及参数配对，并在结束时停止实验。每个 trial 使用 1 个 PyTorch CPU 线程，实验并发为 4；管理服务会额外创建管理进程。4 分钟超时是本机验证设置，慢速电脑可调整脚本中的 deadline。

## 剩余事项

验证结论限于上述路径，不能证明 NNI 所有功能均无 bug。CUDA、远程/云训练、NAS、模型压缩以及全部第三方 tuner 尚未完成端到端验证。真实产酸数据的完整建模任务尚未在此 fork 上重新训练。

BOHB 在新版 ConfigSpace 下的量化分布兼容仍需补齐。依赖升级针对直接声明的版本，旧云组件和部分前端工具仍存在陈旧传递依赖。前端构建使用 `legacy-peer-deps`，尚未形成完全无冲突的现代依赖树。

建议在新环境启动新实验；旧 checkpoint 在新依赖组合下的兼容性未经验证。
