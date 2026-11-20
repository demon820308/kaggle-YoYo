# 从注册 Kaggle 到部署与连接使用 OmniVoice 完整指南

本文档面向 **H3 Kaggle 控制台**（`127.0.0.1:8760`），完整说明以下全流程：
1. 注册 Kaggle 账号、完成手机验证与获取 `KGAT_` API Token
2. 本项目如何在 Kaggle 云端自动部署并运行 `k2-fsa/OmniVoice`（无需手动上传模型或手写 Notebook）
3. 本地项目配置、启动、连接与使用 OmniVoice（声音设计 / 声音克隆 / 自动声线 / 声音库复用 / SRT 导出）

---

## 1. 总体工作原理

本项目采用**「本地控制台 + Kaggle 云端一次性内核执行」**架构，你不需要提前在 Kaggle 网页上手动搭建 Notebook 或手动下载模型权重：

```text
本地浏览器 (http://127.0.0.1:8760)
   │  REST + SSE (web/index.html)
   ▼
本地 FastAPI 服务 (server/app.py)
   │  全局 GPU 锁 + 作业持久化 (server/jobs.py -> data/jobs.json)
   │  读取模板 kernels/omnivoice_unified.py.tpl 并注入参数
   ▼
Kaggle REST API (https://www.kaggle.com/api/v1)
   │  POST /kernels/push (带 Bearer KGAT_... Token)
   ▼
Kaggle 云端 GPU 容器 (Tesla T4 × 2, 使用 cuda:0, enableInternet=True)
   ├─ 1. pip install omnivoice soundfile (可选 omnivoice[tn])
   ├─ 2. 从 HuggingFace 加载模型 k2-fsa/OmniVoice (float16)
   ├─ 3. (克隆模式) 解码参考音频或直接加载 voice_prompt.pt
   ├─ 4. 逐句推理生成 s1.wav ~ sN.wav 并拼接整段音频
   └─ 5. 写出 result.json 与日志 -> 本地轮询完成后自动拉回 data/voice_outputs/<run_id>/
```

---

## 2. 注册 Kaggle 账号与获取 API Token

### 2.1 注册并完成手机号验证（必须）
1. 打开 [Kaggle 官网](https://www.kaggle.com/) 注册账号（支持邮箱或 Google 账号登录）。
2. 进入右上角头像 → **Settings** → 找到 **Phone verification**（手机号验证）。
3. 绑定并验证手机号。
   > **为什么必须验证手机？**  
   > Kaggle 规定未通过手机验证的账号**无法使用 GPU 加速器**，也**无法开启内核联网（Internet）**。OmniVoice 需要 T4 GPU 与联网安装 `omnivoice` 及拉取 `k2-fsa/OmniVoice` 权重，因此这一步不可或缺。

### 2.2 了解每周免费 GPU 配额
- Kaggle 免费账号每周提供 **30 小时（108,000 秒）** GPU 配额（对应 [server/config.py](file:///e:/demonyoyo/google-colab/server/config.py) 中的 `KAGGLE_GPU_WEEKLY_SEC = 108000`）。
- 本项目每次只推送单个作业，实测跑 3 句语音（含装包、加载模型与推理）大约消耗 40–70 秒左右 GPU 时间。

### 2.3 生成 `KGAT_` API Token
本项目使用 Kaggle 新版 Bearer Token（**必须以 `KGAT_` 开头**，见 [server/config.py](file:///e:/demonyoyo/google-colab/server/config.py#L166-L190)）：

1. 登录 Kaggle，点击右上角头像 → **Settings**。
2. 向下滚动到 **API** 区域，点击生成 API Token（获取以 `KGAT_` 开头的 Access Token 字符串）。
3. 妥善保存该 Token，不要提交到 Git 仓库或泄露给第三方。

> **Token 保存位置与优先级**（[server/config.py](file:///e:/demonyoyo/google-colab/server/config.py#L166-L174)）：
> 1. 优先读取环境变量 `KAGGLE_API_TOKEN`。
> 2. 否则读取本地文件 `data/kaggle_token`。
> 3. 通过控制台界面保存 Token 时，系统会自动调用 `icacls` 将 `data/kaggle_token` 的 Windows 访问权限限制为仅当前用户可读写，并在日志中自动脱敏为 `<KGAT_TOKEN>`。

---

## 3. OmniVoice 在 Kaggle 上的部署机制

本项目通过 [kernels/omnivoice_unified.py.tpl](file:///e:/demonyoyo/google-colab/kernels/omnivoice_unified.py.tpl) 实现了**自动化云端部署与单次模型加载推理**，无需人工去 Kaggle 网页手动操作。

### 3.1 内核自动配置规格
由 [server/kaggle_client.py](file:///e:/demonyoyo/google-colab/server/kaggle_client.py#L83-L120) 与 [server/config.py](file:///e:/demonyoyo/google-colab/server/config.py#L29-L60) 统一管理：

| 配置项 | 设计/自动模式 (`design` / `auto`) | 克隆模式 (`clone`) | 说明 |
|---|---|---|---|
| 内核 Slug | `{KAGGLE_OWNER}/yoyo-omnivoice` | `{KAGGLE_OWNER}/yoyo-omnivoice-clone` | 自动创建或更新版本 |
| 内核类型 | `python` / `script` | `python` / `script` | 私有内核（`isPrivate: True`） |
| 硬件机型 | `NvidiaTeslaT4` | `NvidiaTeslaT4` | 实测 `Tesla T4, 15360 MiB` |
| GPU / 联网 | `enableGpu=True`, `enableInternet=True` | `enableGpu=True`, `enableInternet=True` | 装包和拉模型必须联网 |
| 停顿默认 | `0.5s` | `0.4s` | 界面可自由调整 `0.0–5.0s` |
| 语速默认 | `0.85×` | `1.0×` | 界面可自由调整 `0.50–2.00×` |
| 整段输出名 | `test_full.wav` | `clone_full.wav` | 分句为 `s1.wav` ~ `sN.wav` |

### 3.2 新账号 / 多账号接入说明
控制台现在支持在网页上直接绑定、管理与切换多个 Kaggle 账号（无需再手动改代码）：
1. **用户名来源**：填写个人主页 URL `https://www.kaggle.com/<用户名>` 中的 `<用户名>`。
2. **默认环境镜像**：新增账号时默认**不传**固定 `OMNIVOICE_DOCKER_IMAGE`，直接由 Kaggle 分配当前默认 GPU Python 环境（新账号最稳妥）；如有需要可在高级选项中勾选使用预设 BYOD 镜像。

### 3.3 云端内核执行流程详解
每次点击「生成语音」，云端脚本会按以下顺序执行：
1. **探测 GPU**：运行 `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`。
2. **安装依赖**：执行 `pip install -q omnivoice soundfile`。
   - 若勾选了「开启数字/日期转读法（`normalize_text`）」，会尝试安装 `omnivoice[tn]`；若 `[tn]` 安装失败，脚本会自动记录 `tn_fallback: true` 并回退到 `omnivoice`。
3. **单次加载模型**：
   ```python
   model = OmniVoice.from_pretrained(
       "k2-fsa/OmniVoice",
       device_map="cuda:0",
       dtype=torch.float16,
   )
   ```
   输出采样率为 `24000 Hz`。
4. **声音克隆预处理（仅 `clone` 模式）**：
   - **若复用声音库**：直接解码内嵌的 `voice_prompt.pt` 并调用 `VoiceClonePrompt.load(...)`，跳过 `ffmpeg` 转码和语音识别，大幅缩短耗时。
   - **若上传新参考音频**：解码 base64 音频 → `ffmpeg` 转为单声道 `24000 Hz` 的 `ref.wav` → 调用 `model.create_voice_clone_prompt(...)`。若未填 `ref_text` 且内置 ASR 失败，会自动安装 `openai-whisper`（`small` 模型）转写 `ref.wav` 作为兜底，最后将生成的特征保存为 `voice_prompt.pt` 一并传回本地。
5. **逐句生成与拼接**：对输入文本逐句调用 `model.generate(...)`，保存 `s1.wav` ~ `sN.wav`，然后按 `pause_sec` 插入静音片段拼接成整段音频。
6. **写出结果**：生成 `result.json`（含逐句时长、推理耗时、模型加载耗时、`verdict` 状态码等）。

---

## 4. 本地项目安装、连接与启动

### 4.1 本地环境准备（Windows）
1. **Python 环境**：建议 Python 3.12（项目目录已包含 `.venv`）。如需新建虚拟环境：
   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   依赖清单（[requirements.txt](file:///e:/demonyoyo/google-colab/requirements.txt)）：
   - `fastapi>=0.115.0`
   - `uvicorn>=0.32.0`
   - `httpx>=0.27.0`
   - `python-multipart>=0.0.12`
   - `kagglesdk>=0.1.0`
   - `google-colab-cli>=0.8.0`
2. **可选安装 `ffprobe`（随 FFmpeg 提供）**：
   上传参考音频时，后端会调用 `ffprobe` 本地探测音频时长（判断是否在 3–10 秒最佳区间）。若系统未安装 `ffprobe`，不影响上传与云端克隆，仅无法在提交前显示参考音频时长。

### 4.2 启动控制台
- **方式一**：直接双击项目根目录的 [start.bat](file:///e:/demonyoyo/google-colab/start.bat)。
- **方式二**：在项目根目录终端运行：
  ```powershell
  .venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8760
  ```
启动后会自动在浏览器打开：[http://127.0.0.1:8760](http://127.0.0.1:8760)。

### 4.3 在控制台连接与切换多个 Kaggle 账号
1. **首次初始化连接（含自动检测与部署）**：
   - 尚未添加账号时，页面顶部会自动展示 **「初始化连接」** 卡片，左侧提供「注册账号 → 验证手机 → 创建 `KGAT_` API Key」三步详细说明。
   - 在右侧填入 **Kaggle 用户名**、**`KGAT_` API Key** 以及可选的 **备注名**，点击 **「验证并连接 Kaggle」**：
     - 系统会先只读检测该账号的 Kaggle 云端是否已存在 `yoyo-omnivoice` 内核（0 秒 GPU 消耗）。
     - **若已部署**：自动标记为「已部署 OmniVoice」，**跳过重复部署**，直接解锁语音生成。
     - **若尚未部署**（且勾选了「自动检测并部署」）：会自动推送并执行一次快速自检部署，并在页面上实时显示 5 步部署进度（验证配额 → 推送内核 → 等待调度 → 云端装包与加载 `k2-fsa/OmniVoice` → 校验并试听自检音频）。
2. **添加更多账号与快速切换**：
   - 点击顶栏右上角 **「账号设置」**，可在弹窗中添加第二个、第三个 Kaggle 账号（同样保存在本机 `data/kaggle_accounts.json`，仅限当前用户访问权限），同样支持「已部署跳过 / 未部署自动部署」与「检测线上」。
   - 添加完成后，可以直接在顶栏 **`Kaggle:` 下拉框** 或 **「账号设置」** 中一键切换当前使用的 Kaggle 账号；切换后会自动刷新该账号的本周 30 小时免费 GPU 额度（任务运行期间为防止状态错乱会暂锁切换）。
3. **验证通过后，顶部状态栏会实时显示**：
   - **当前 Kaggle 账号切换下拉框**（`owner`）
   - **本周 GPU 额度**（已用 / 总计 `108000` 秒、剩余时间、`reserved` 占用）
   - **GPU 锁状态**（`空闲` / 任务运行中）

> **只读连通性测试脚本（可选）**：  
> 在保存 Token 到 `data/kaggle_token` 或设置 `KAGGLE_API_TOKEN` 后，也可以执行以下命令测试 Token 与额度接口（不推内核、不耗 GPU）：
> ```powershell
> .venv\Scripts\python.exe spike\kaggle_spike.py
> ```

---

## 5. 在控制台中使用 OmniVoice

进入默认页签 **「语音生成 (Kaggle OmniVoice)」**，支持三种模式与完整后处理功能。

### 5.1 三种生成模式

#### 模式 A：声音设计（`design`，内核 `yoyo-omnivoice`）
通过文字描述控制音色，无需参考音频：
- **声线预设**：提供教学女声、男低音、童声故事、耳语助眠、四川话、东北话、陕西话、英式英语、印度口音等 11 个预设（定义于 [server/config.py](file:///e:/demonyoyo/google-colab/server/config.py#L96-L108)）。
- **声线属性构建器**：点击按钮自动拼接「性别 / 年龄 / 音调 / 风格（耳语）/ 12 种中文方言 / 10 种英文口音」。
- **内置校验护栏**：
  - 默认不允许 `instruct` 中英混写（如需要可勾选「允许中英混写」）。
  - 同一类目多选、中文方言配非中文文本、英文口音配非英文文本时，界面会提前给出提示。

#### 模式 B：声音克隆（`clone`，内核 `yoyo-omnivoice-clone`）
复刻指定说话人的音色：
1. **上传新参考音频**：
   - 支持 `.m4a` / `.wav` / `.mp3` / `.flac` / `.opus`。
   - 建议音频长度 **3–10 秒**、人声清晰无强背景音乐、体积尽量 **< 2 MB**（因为音频会随脚本 base64 内嵌推送）。
   - **参考文本 `ref_text`**：可留空（云端自动识别或 Whisper 兜底转写）；手动填写准确文本可显著提高首字稳定性与断句效果。
2. **从「我的声音库」直接复用 `.pt`**：
   - 任意一次新音频克隆成功后，点击结果卡片右上角 **「存为声音库」**，会将云端产出的 `voice_prompt.pt` 保存至 `data/voice_library/<voice_id>/`。
   - 下次在「声音来源」下拉框直接选择该声音即可，无需重新上传音频，云端也会跳过转码与 Whisper 步骤。

#### 模式 C：自动声线（`auto`，内核 `yoyo-omnivoice`）
不传 `instruct` 也不传参考音频，由 OmniVoice 模型根据输入文本自动匹配最合适的说话人声线。

---

### 5.2 推理参数与文本控制

#### 共享生成参数
- **目标语言（`language`）**：默认「自动识别（不传 language）」，内置 15 种常见语言，也支持选择自定义输入 OmniVoice 支持的 646 种语言名/代码（如 `yue`）。
- **时长控制方式**：
  - **语速调节（`speed`）**：`0.50× – 2.00×`（设计默认 `0.85`，克隆默认 `1.00`）。
  - **固定单句时长（`duration`）**：`0.3 – 60.0 秒`，开启后会覆盖语速设置。
- **推理步数（`num_step`）**：默认 `32` 步（标准质量），可选 `16` 步（快速推理），接口支持范围 `4 – 64`。
- **高级推理参数**：
  - **句间停顿（`pause_sec`）**：`0.0 – 5.0` 秒。
  - **引导强度（`guidance_scale`）**：默认 `2.0`。
  - **数字/日期转读法（`normalize_text`）**：文本含阿拉伯数字时界面会提示一键开启，避免逐字播报。
  - **人声降噪（`denoise`）**：默认开启。
  - **裁剪首尾长静音（`postprocess_output`）**：默认开启。

#### 文本编辑工具栏
文本框内**每行代表一句**，单次最多提交 **50 句**。工具栏支持：
- **长文本智能切句**：按中文/英文标点自动切分为每句 ≤ 55 字的逐行文本。
- **插入副语言标签**（13 种）：如 `[laughter]`（笑声）、`[sigh]`（叹气）、`[confirmation-en]`、`[question-ah]`、`[surprise-wa]`、`[dissatisfaction-hnn]` 等。
- **拼音纠音**：选中多音字后点击按钮，替换为大写拼音 + 声调（如 `打ZHE2`）。
- **CMU 音素**：插入英文音素控制发音（如 `[B EY1 S]`）。

---

### 5.3 任务提交、结果校验与导出

1. **防并发安全锁**：
   - 系统全局仅允许同时运行 1 个 Kaggle GPU 任务（视频或语音共用同一把锁）。
   - 若本机控制台意外重启，下次启动时检测到 `data/jobs.json` 中仍有 `queued / running / fetching / reconcile` 作业，会自动恢复**只读轮询**，绝不会重复推送消耗额度（见 [server/app.py](file:///e:/demonyoyo/google-colab/server/app.py#L54-L71)）。
2. **5 项自动校验（任务完成后自动执行）**：
   - `verdict_ok`：云端返回 `*OK` 且无失败句
   - `rms_ok`：整段拼接音频 `RMS > 0.01`（非静音）
   - `sample_rate_ok`：采样率为 `24000 Hz`
   - `duration_ok`：整段实测时长与「各分句时长之和 + 句间停顿」误差 ≤ `0.3s`
   - `quota_ok`：Kaggle `reserved_sec` 已归 0
3. **试听、导出与局部重生成**：
   - 可在线播放整段音频和逐句音频 `s1.wav` ~ `sN.wav`。
   - 若某几句发音不满意或失败，勾选对应句子后点击 **「带入选中/失败句重生成」**，会直接填入输入框方便微调重跑。
   - **下载 SRT**：根据各分句实测时长与 `pause_sec` 自动对齐时间轴导出 `.srt` 字幕。
   - **下载全部 (.zip)**：打包下载本次运行的所有 `.wav`、`result.json` 与 `kernel.log`。

---

## 6. 项目关键文件与本地数据目录

| 路径 | 作用 |
|---|---|
| [start.bat](file:///e:/demonyoyo/google-colab/start.bat) | Windows 一键启动脚本（打开 `127.0.0.1:8760`） |
| [requirements.txt](file:///e:/demonyoyo/google-colab/requirements.txt) | Python 依赖清单 |
| [server/config.py](file:///e:/demonyoyo/google-colab/server/config.py) | 多账号配置管理、内核 slug、声线词表与凭据脱敏 |
| [server/app.py](file:///e:/demonyoyo/google-colab/server/app.py) | FastAPI 服务入口与全部 HTTP/SSE 接口 |
| [server/kaggle_client.py](file:///e:/demonyoyo/google-colab/server/kaggle_client.py) | Kaggle API 封装（quota / push / status / output / download） |
| [server/jobs.py](file:///e:/demonyoyo/google-colab/server/jobs.py) | 全局 GPU 互斥锁、状态轮询与额度核对 |
| [server/voice_kaggle.py](file:///e:/demonyoyo/google-colab/server/voice_kaggle.py) | OmniVoice 参数校验、模板渲染、5 项 WAV 校验、声音库、SRT 导出 |
| [kernels/omnivoice_unified.py.tpl](file:///e:/demonyoyo/google-colab/kernels/omnivoice_unified.py.tpl) | 统一版 OmniVoice Kaggle 内核模板 |
| [web/index.html](file:///e:/demonyoyo/google-colab/web/index.html) | 前端单页控制台（含初始化引导与多账号切换） |
| `data/kaggle_accounts.json` | 本地保存的多账号用户名与 `KGAT_` Token（**切勿提交到公开 Git 仓库**） |
| `data/voice_outputs/<run_id>/` | 每次运行的音频、日志与 `result.json` |
| `data/voice_library/` | 声音库保存的 `voice_prompt.pt` 文件 |

---

## 7. 常见问题排查

1. **保存凭据报错 `API Key (Token) 必须以 KGAT_ 开头`**
   - 请确认使用的是 Kaggle Settings → API 页面中以 `KGAT_` 开头的 Access Token，而不是旧版 `kaggle.json` 里的 `{"username":...,"key":...}`。
2. **推送内核失败（403 / GPU 或 Internet 权限不足）**
   - 检查 Kaggle 账号是否已完成 **Phone verification（手机验证）**。
   - 检查填写的 **Kaggle 用户名**（个人主页 URL `kaggle.com/<用户名>`）是否与该 `KGAT_` API Key 所属账号完全一致。
   - 新增账号建议保持不勾选「使用预设 BYOD 镜像」，直接使用 Kaggle 默认 GPU 环境。
3. **提示「Kaggle 账号当前有任务在跑（time_reserved=...s），禁止并发提交」**
   - 同一 Kaggle 账号同一时间只能跑一个 GPU 任务。请打开 Kaggle 网页 → **Code → Your Work** 查看是否有手动开启的 Notebook Session 未关闭，或等待上一个作业结束、点击顶部「刷新额度」确认 `reserved` 归 0 后再提交。
4. **声音设计模式提示「不允许中英混写」**
   - OmniVoice 的声音设计指令分别在纯中文和纯英文指令上训练；如确需混写，勾选输入框旁边的 **「允许中英混写」** 即可提交。
5. **声音克隆首字不准或有吞字**
   - 建议参考音频控制在 **3–10 秒**，并在「参考音频文本 `ref_text`」中手动准确填写参考音频里说话的内容，不要完全依赖自动转写。
