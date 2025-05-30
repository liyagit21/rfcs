

<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`. 
    - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
    - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/master/RFC-0000-template.md#resolution).
    - If the RFC is idle here (no activity for 2 weeks), assign the label `stalled` to the PR.
- Once the discussion has settled, assign a new label based on the level of support:
    - `accepted` if a decision has been made in the RFC
    - `draft` if the author needs to rework the RFC’s proposal
    - `shelved` if there are no plans to move ahead with the current RFC’s proposal. We want neither to think about evaluating the proposal
nor about implementing the described feature until some time in the future.
- A state of `accepted` means that the core team has agreed in principle to the proposal, and it is ready for implementation. 
- The author (or any interested developer) should next open a tracking issue on Github corresponding to the RFC.
    - This tracking issue should contain the implementation next steps. Link to this tracking issue on the RFC (in the Resolution > Next Steps section)
- Once all relevant PRs are merged, the RFC’s status label can be finally updated to `closed`.

</details>





# Decouple CUDA Codes

**Authors:**
* @nickname
* @nickname 


## **Summary** （1人）王家喜
A short paragraph or bullet list that quickly explains what you're trying to do.
方案的总览


## **Highlights** （1人）袁孟雯
阐述CUDA代码分离工作的亮点
- 将 CUDA 相关实现从主工程中抽离，降低 PyTorch 核心框架对 CUDA 的直接耦合，提升整体工程可维护性。
- 更清晰、统一的目录层级结构，提升可读性与可维护性，使开发者能快速定位并理解后端逻辑，降低新开发者参与的学习门槛，为长期维护和社区贡献者提供更友好的结构。
- 重写构建系统以支持 CUDA 后端独立编译，降低编译复杂度，实现更快的增量构建和更少的构建依赖。
- 统一设备后端架构风格，为后续支持更多第三方后端提供模板，降低集成门槛和时间成本，提升 PyTorch 后端接入的一致性与可插拔性。

## **Motivation**（1人）祝贺
What motivates this proposal and why is it important?
How should users and developers think about this feature, how would it impact the way PyTorch is used?
Explain impact and value of this feature


## **Proposed Implementation**
This is the bulk of the RFC. Explain the design in enough detail for somebody familiar with PyTorch to understand, and for somebody familiar with the implementation to implement. 
This should get into specifics and corner-cases, and include examples of how the feature is used, and how it will interact with other features. Any new terminology should be defined here.
Consider:
*   using examples and diagrams to help illustrate your ideas.
*   including code examples, if you're proposing an interface or system contract.
*   linking to project briefs or wireframes that are relevant.


*   代码分离（1人）  张靖
<h3 id="fc3655d2"><font style="color:rgba(0, 0, 0, 0.9);">一、需要解耦的功能模块</font></h3>

<font style="color:rgba(0, 0, 0, 0.9);">涉及到需要解耦的功能模块包括：</font>

```markdown
├── aten                    # Tensor ops相关代码
├── c10/cuda                # 设备管理核心代码
├── caffe2                  # torch和caffe2合并的遗留文件夹，当前仅需保留CMakeLists.txt，用于编译和连接aten和c10的源文件
├── torch/cuda              # Python端接口
├── torch/backends          # PyTorch硬件后端Python接口
├── torch/csrc/cuda         # 主要是Python端和C++端的pybind
├── torch/csrc/distributed  # distributed底层实现代码
├── torch/csrc/jit          # jit底层实现代码
├── torch/csrc/profiler     # profiler底层实现代码
├── torch/csrc/dynamo       # dynamo底层实现代码
└── torch/csrc/inductor     # inductor底层实现代码
```

<h3 id="d65ce516"><font style="color:rgba(0, 0, 0, 0.9);">二、解耦方式</font></h3>

<font style="color:rgba(0, 0, 0, 0.9);">我们通过实践总结了以下四种解耦方式，这四种方式并不是独立的，而是相互照应、相互补充的。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/32361127/1748503065598-84d654cb-cecf-4e73-ac26-1e5e3bb30ac5.png)

<h4 id="83253c23"><font style="color:rgba(0, 0, 0, 0.9);">（一）文件间解耦</font></h4>

<font style="color:rgba(0, 0, 0, 0.9);">解耦</font><font style="color:rgba(0, 0, 0, 0.9);">标准如下：</font>

<h5 id="H6xQ7"><font style="color:rgba(0, 0, 0, 0.9);">1. 文件夹名称包含 </font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cuda</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cudnn</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">THC</font>`<font style="color:rgba(0, 0, 0, 0.9);"> 关键字。</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>


- `torch/backends/cuda`
- `torch/backends/cudnn`
- `torch/cuda`
- `aten/src/ATen/cuda`
- `aten/src/ATen/cudnn`
- `aten/src/ATen/native/cuda`
- `aten/src/ATen/native/cudnn`
- `aten/src/ATen/native/nested/cuda`
- `aten/src/ATen/native/quantized/cuda`
- `aten/src/ATen/native/quantized/cudnn`
- `aten/src/ATen/native/sparse/cuda`
- `aten/src/ATen/native/transformers/cuda`
- `aten/src/THC`
- `torch/csrc/cuda`
- `torch/csrc/distributed/c10d/cuda`

<h5 id="E2gdO"><font style="color:rgba(0, 0, 0, 0.9);">2. 文件名包含 </font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cuda</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cudnn</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">THC</font>`<font style="color:rgba(0, 0, 0, 0.9);"> 关键字。</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>


- `torch/csrc/distributed/rpc/tensorpipe_cuda.cpp`
- `torch/csrc/profiler/stubs/cuda.cpp`

<h5 id="bOOx4"><font style="color:rgba(0, 0, 0, 0.9);">3. 后缀名是 </font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">.cu</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">.cuh</font>`<font style="color:rgba(0, 0, 0, 0.9);">。</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>

- `torch/csrc/distributed/c10d/quantization/quantization_gpu.cu`

<h4 id="586d0dc4"><font style="color:rgba(0, 0, 0, 0.9);">（二）文件内解耦</font></h4>

有些cuda代码直接和torch代码耦合在一个文件内，通过环境变量、宏定义或者设备判断等隔离。

<h5 id="e819c231"><font style="color:rgba(0, 0, 0, 0.9);">1. 包含 CUDA 相关的环境变量判断</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>

```cpp
#if defined(__CUDA_ARCH__) 存在于下列文件
torch/csrc/aten/native/Distributions.h

#if defined(__CUDACC__) 存在于下列文件
torch/csrc/aten/native/sparse/Macros.h

#ifdef USE_CUDA 存在于下列文件或者文件夹
caffe2/CMakeLists.txt
torch/csrc/Storage.cpp
torch/csrc/dynamo/guards.cpp
torch/csrc/inductor/aoti_runner/pybind.cpp
torch/csrc/jit
```

<h5 id="f813fdd6"><font style="color:rgba(0, 0, 0, 0.9);">2. 文件内包含 CUDA 相关宏定义</font></h5>

+ `TORCH_CUDA_CU_API`
+ `TORCH_CUDA_CPP_API`
+ `TORCH_CUDA_CHECK`

<h5 id="2e31866b"><font style="color:rgba(0, 0, 0, 0.9);">3. 文件内包含 is_cuda、kCUDA、“cuda”等</font></h5>

示例：

```cpp
static CUDAHooksInterface* cuda_hooks = nullptr;
xxtensor.is_cuda()
xxtensor.device().type() == at::kCUDA
register_cuda_runner("cuda", &create_aoti_runner_cuda)
```

<h4 id="49e30103"><font style="color:rgba(0, 0, 0, 0.9);">（三）补充编译有依赖的文件</font></h4>

为了独立编译CUDA，CUDA编译需要依赖的文件也做了解耦。

<font style="color:rgba(0, 0, 0, 0.9);">需要补充的文件类型包括：</font>

<h5 id="FuDse">1. `<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">*.h</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">*.hpp</font>`<font style="color:rgba(0, 0, 0, 0.9);"> 头文件。</font></h5>
<style>
/* 定义用于文件路径的样式 */
.path-highlight {
    color: rgba(0, 0, 0, 0.9);
    background-color: rgba(0, 0, 0, 0.03);
    padding: 2px 4px;
    border-radius: 3px;
    font-family: monospace;
}

/* 定义用于标题的样式 */
.header-highlight {
    color: rgba(0, 0, 0, 0.9);
}
</style>

示例：

<span class="path-highlight">torch/csrc/autograd/functions/comm.h</span>

<h5 id="x8Noj"><span class="header-highlight">2. 配置文件。</span></h5>

示例：

<span class="path-highlight">aten/src/ATen/ATenConfig.cmake.in</span>  
<span class="path-highlight">aten/src/ATen/Config.h.in</span>  
<span class="path-highlight">aten/src/ATen/native/native_functions.yaml</span>  
<span class="path-highlight">aten/src/ATen/native/tags.yaml</span>  
<span class="path-highlight">aten/src/ATen/native/ts_native_functions.yaml</span>

<h5 id="x6ttV"><span class="header-highlight">3. 模板文件。</span></h5>

示例：

<span class="path-highlight">aten/src/ATen/templates</span>

<h5 id="CTl4m"><span class="header-highlight">4. 打桩文件。</span></h5>

示例：

<span class="path-highlight">torch/csrc/stub.c</span>

<h4 id="b867b20f"><font style="color:rgba(0, 0, 0, 0.9);">（四）根据编译文件功能模块划分</font></h4>

<font style="color:rgba(0, 0, 0, 0.9);">有助于查漏补缺、去除冗余代码。</font>

示例 1：

<font style="color:rgba(0, 0, 0, 0.9);">通过</font>build_variables<font style="color:rgba(0, 0, 0, 0.9);">.bzl中文件划分解耦 distributed 模块 CUDA 相关代码</font>

```cpp
# These files are the only ones that are supported on Windows.
libtorch_cuda_distributed_base_sources = [
    "torch/csrc/distributed/c10d/reducer_cuda.cpp",
]

# These files are only supported on Linux (and others) but not on Windows.
libtorch_cuda_distributed_extra_sources = [
    "torch/csrc/distributed/c10d/CudaDMAConnectivity.cpp",
    "torch/csrc/distributed/c10d/NCCLUtils.cpp",
    "torch/csrc/distributed/c10d/FlightRecorder.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupUCC.cpp",
    "torch/csrc/distributed/c10d/UCCTracing.cpp",
    "torch/csrc/distributed/c10d/UCCUtils.cpp",
    "torch/csrc/distributed/c10d/intra_node_comm.cpp",
    "torch/csrc/distributed/c10d/intra_node_comm.cu",
    "torch/csrc/distributed/c10d/CUDASymmetricMemory.cu",
    "torch/csrc/distributed/c10d/CUDASymmetricMemoryOps.cu",
    "torch/csrc/distributed/c10d/cuda/AsyncMM.cu",
    "torch/csrc/distributed/c10d/NanCheck.cu",
    "torch/csrc/distributed/rpc/tensorpipe_cuda.cpp",
    "torch/csrc/distributed/c10d/quantization/quantization_gpu.cu",
]

libtorch_cuda_distributed_sources = libtorch_cuda_distributed_base_sources + libtorch_cuda_distributed_extra_sources
```

示例 2：

根据aten\src\ATen\CMakeLists.txt中文件划分添加aten\src\ATen\native\miopen代码

```cpp
list(APPEND ATen_CUDA_CPP_SRCS
  ${cuda_cpp}
  ${native_cuda_cpp}
  ${native_cudnn_cpp}
  ${native_miopen_cpp}
  ${native_nested_cuda_cpp}
  ${native_quantized_cuda_cpp}
  ${native_quantized_cudnn_cpp}
  ${native_sparse_cuda_cpp}
  ${native_transformers_cuda_cpp}
)
```


*   目录重构（1人）  张靖
<h3 id="7e0821a0"><font style="color:rgba(0, 0, 0, 0.9);">三、目录重构</font></h3>

![](https://cdn.nlark.com/yuque/0/2025/png/32361127/1748588334657-62159db5-d4d0-4d15-adc6-2a833d54f966.png)

cuda解耦出来后，原始目录参考第一节，除了nvidia（cuda），我们调研了[AMD(gpu)](https://github.com/ROCm/pytorch)、[google(tpu)](https://github.com/pytorch/xla/tree/master)、[intel(xpu)](https://github.com/intel/intel-extension-for-pytorch)、[acend(npu)](https://gitee.com/ascend/pytorch)、[Cambricon(mlu)](https://github.com/Cambricon/torch_mlu/tree/r2.4_develop)等多个超算卡厂商适配pytorch的方式，总结了各厂商适配PyTorch的代码目录结构、相似和特异性改动点，着重考虑到了以下因素：

<h5 id="Ee0MW">1. Python/C++分层解耦</h5>

通过物理隔离Python层（core/、backends/）和C++层（csrc/），明确区分接口定义与底层实现，降低代码耦合度。这样有助于Python层专注于业务逻辑和用户接口，而C++层则处理底层实现和性能优化。

<h5 id="CkxwH"><font style="color:rgb(44, 44, 54);">2. 模块化独立插件</font></h5>

将distributed/、profiler/作为独立插件，与核心框架解耦，使得各个模块可以独立开发、测试和迭代，同时也便于第三方开发者根据需要选择性地集成或扩展某些功能。

<h5 id="HowFi"><font style="color:rgb(44, 44, 54);">3. 统一硬件适配框架</font></h5>

合并 `c10/cuda` 和 `caffe2` 为 `framework/`，形成统一的设备管理与资源调度层，降低了维护成本。

<h5 id="CtkWI"><font style="color:rgb(44, 44, 54);">4. 目录重命名</font></h5>

<font style="color:rgb(44, 44, 54);">新的目录命名尽量直观地反映了其包含的内容和功能，例如 core 表示核心接口层，csrc 表示C++源代码，python 表示Python与C++的绑定层等，便于开发人员快速理解和导航项目代码。</font>

最后，整理出的新的适配代码目录结构如下：

```markdown
├── backends/                     # 只保留CUDA、cuDNN相关python接口
├── core/                         # Python核心接口层
├── csrc/                         # C++核心代码仓库
│   ├── python/                   # Python与C++绑定层
│   ├── aten/                     # Tensor、ops运算相关
│   ├── framework/                # 框架基础架构
│   ├── dynamo/                   # 动态图转静态图工具
│   ├── inductor/                 # 硬件代码生成与编译器模块
│   └── jit/                      # JIT编译器模块
├── distributed/                  # 分布式计算模块（独立插件）
└── profiler/                     # 性能分析模块（独立插件）
```

<font style="color:rgb(44, 44, 54);"></font>













*   编译工程（1人）  黄雷


## 优缺点（1人）   付泽伟
## **Metrics **
What are the main metrics to measure the value of this feature? 


## **Drawbacks **
Are there any reasons why we should not do this? Here we aim to evaluate risk and check ourselves.

Please consider:
* is it a breaking change?
* Impact on UX
* implementation cost, both in terms of code size and complexity
* integration of this feature with other existing and planned features


## **Alternatives**   洪泓
What other designs have been considered? What is the impact of not doing this?
in-tree 
out-of-tree (cuda key, pu1 key)

## **Prior Art**（1人） 崔巍
Discuss prior art (both good and bad) in relation to this proposal:
* Does this feature exist in other libraries? What experience has their community had?
* What lessons can be learned from other implementations of this feature?
* Published papers or great posts that discuss this


## **How we teach this**
* What names and terminology work best for these concepts and why? How is this idea best presented?
* Would the acceptance of this proposal mean the PyTorch documentation must be re-organized or altered?
* How should this feature be taught to existing PyTorch users?


## **Unresolved questions**
* What parts of the design do you expect to resolve through the RFC process before this gets merged?
* What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
* What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?


## Resolution
We decided to do it. X% of the engineering team actively approved of this change.

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context
Some people were in favor of it, but some people didn’t want it for project X.


### Next Steps（1人）侯丽亚
Will implement it. 


#### Tracking issue
<github issue URL>


#### Exceptions
Not implementing on project X now. Will revisit the decision in 1 year.
