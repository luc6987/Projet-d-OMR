如何判断一个音的时值:
我现在需要调试最后一步组装符号 修改规则

符号定义:
s_i=[s_x1,s_x2] × [s_y1,s_y2] ∈ S 是符干
t_i=[t_x1,t_x2] × [t_y1,t_y2] ∈ T 是符头
两个符号的距离定义为: d = sup_{a ∈ s_i, b ∈ t_i} (a-b)^2

规则1: 符头-符干连接
对于每一个符头:
- 如果和一个符干有重合（bbox overlap），那么连接符干到符头
- 如果没有重合，找最近的符干（距离d最小）
- 特殊情况：对于实心符头（notehead-full），只接受重合的符干，不接受非重合的符干
  - 如果实心符头没有重合的符干，则进入规则1.1（虚拟符干创建）

规则1.1: 实心符头虚拟符干创建
对于没有重合符干的实心符头（notehead-full）:
1. 向正上方和正下方搜索beam和flag
   - 搜索范围：不超过本五线谱和相邻五线谱间距的一半
   - 水平对齐：在符头宽度×2范围内
   - 垂直距离：不超过搜索范围
   - 非重叠：beam/flag与符头不重叠（垂直距离 > 符头高度/2）
2. 如果找到beam/flag：
   - 创建虚拟符干，延伸到所有找到的beam/flag
   - 连接虚拟符干和所有找到的beam/flag
3. 如果没找到beam/flag：
   - 仍然创建虚拟符干（默认长度：3.5个线间距）
   - 符干方向修正：根据符头在五线谱中的位置决定符干延伸方向
     * 如果符头在五线谱中间的线下方（包括中间线）：符干向上延伸（符尾朝上）
     * 如果符头在五线谱中间的线上方：符干向下延伸（符尾朝下）
4. 虚拟符干属性：
   - class_name: "stem"
   - confidence: 0.5（表示虚拟）
   - bbox: 基于符头位置和找到的beam/flag计算，或根据符头位置决定方向

规则2: 符干-flag/beam连接
对于每一个符干（包括虚拟符干）:
- 如果有和flag/beam重合（bbox overlap），那么把flag/beam链接到符干上
- 如果没有重合，找最近的flag/beam（距离d最小）
- 距离限制：只考虑在符干自身宽度范围内的flag/beam
  - 如果符干附近（一个自身宽度）没有beam或flag，就不添加任何链接
- 重复查找：不要找到一个就停下，应该反复直到没有更多的beam/flag
  - 返回所有符合条件的flag和beam列表，而不是只返回一个

规则3: 符点查找
对于每个符头:
- 向正右方（x方向）寻找符点（duration-dot）
- 如果找到符点，时长乘以1.5

规则4: 三连音检测
如果一个beam链接了三个符干:
- 在区域a中搜索"3"符号（tuplet标记）
- 区域a定义：
  - 设beam为 b_i = [b_x1, b_x2] × [b_y1, b_y2]
  - 区域a = [b_x1, b_x2] × [l_y1, l_y2]
  - 其中l_y1是除该符干所对应符头所在的五线谱的下方五线谱的最上面的一根线
  - l_y2是除该符干所对应符头所在的五线谱的上方五线谱的最下面的一根线
- 如果找到了"3"符号，则标记所有三个音符为三连音（is_tuplet=True, tuplet_type="triplet"）

输出规则:
- 所有测试输出保存到 Output/test 目录
- 可视化文件也保存到 Output/test 目录

这套规则旨在将离散的五线谱线（Staff Line）和符号（Symbols）构建成具有层级关系的**乐谱结构树（Score Tree）**。

### 定义与预处理

  * **输入集合**:

      * $L = \{l_1, l_2, ...\}$: 已提取的五线谱线对象集合。每个 $l_i$ 包含属性 $y_{center}$ (垂直中心) 和 $h_{staff}$ (五线谱总高度)。
      * $S$: 所有检测到的符号集合 (Bounding Boxes)。
      * 符号定义: $s = [x_1, x_2] \times [y_1, y_2]$。

  * **辅助函数**:

      * $Overlap_Y(A, B)$: 两个对象在 Y 轴上的投影重叠比例。
      * $Contains_Y(Container, Item)$: $Container.y_1 \le Item.y_{center} \le Container.y_2$。

-----

### 第一阶段：乐行与谱表划分 (System & Staff Grouping)

目标：将所有五线谱线 $L$ 划分为若干个 **System (乐行)**，并确定每个 System 内的 **Part (声部)** 顺序。

#### 规则 1: 五线谱聚类 (System Clustering)

我们要决定哪些五线谱属于同一时间段（即同一个 System）。

1.  **基于连谱号 (Brace/Bracket) 的聚类 [优先级最高]**
      * 遍历所有 $s \in S$ 其中 class 为 `multi-staff_brace`, `multi-staff_bracket`, `staff_grouping`。
      * 对于每个符号 $g$:
          * 找出所有被 $g$ 垂直覆盖的五线谱线 $l_i$ ($Overlap_Y(g, l_i) > 0.5$)。
          * 将这些 $l_i$ 标记为同一个 **System Group**。
2.  **基于连小节线 (Measure Separator) 的聚类 [优先级次之]**
      * 遍历所有 class 为 `measure_separator` 的符号 (连接上下谱表的竖线)。
      * 如果一个 `measure_separator` 的顶端接触 $l_a$，底端接触 $l_b$，则 $l_a$ 和 $l_b$ 以及它们之间的所有谱线属于同一个 **System Group**。
3.  **未归类的处理**
      * 对于未被上述规则归类的剩余谱线，每个谱线独立成为一个 **System Group**。

#### 规则 2: 声部索引 (Part Indexing)

确定了 System 之后，需要给每个谱线分配逻辑 ID。

  * 对于每一个 System $Sys_k$:
      * 将该 System 内的所有谱线按 Y 坐标从小到大（从上到下）排序：$l_{k,1}, l_{k,2}, ..., l_{k,m}$。
      * 分配 **Part ID**:
          * $l_{k,1} \rightarrow Part\_1$
          * $l_{k,2} \rightarrow Part\_2$
          * ...
      * **一致性检查**: 如果 $Sys_k$ 有 2 行谱，而 $Sys_{k+1}$ 有 3 行谱，标记警告 (Warning)，默认前两行对应 Part 1/2，第三行标记为 Part 3 (或根据前面的乐器名文本 `letter_*` 进行修正)。

-----

### 第二阶段：小节划分 (Measure Slicing)

目标：在 System 内部横向切分时间。

#### 规则 3: 全局小节线对齐

小节线在同一个 System 的不同声部间通常是对齐的。

1.  **收集小节线**:
      * 在当前 $Sys_k$ 的区域内，收集所有 `thin_barline`, `thick_barline`, `repeat` (复纵线)。
2.  **X轴投影融合**:
      * 由于检测误差，同一时刻上下两行的小节线 X 坐标可能不完全一致（比如差 5px）。
      * 将所有小节线的 X 坐标投影到一维轴上。
      * 如果两个小节线的 $|x_i - x_j| < threshold$ (例如符头宽度的一半)，则合并为一个 **Global Barline**。
      * 取平均 X 坐标作为该 System 的切分点。
3.  **小节对象创建**:
      * 根据 Global Barline 的 X 坐标区间 $(x_{start}, x_{end})$，创建 **Measure Object**。
      * 将该 System 内所有处于该 X 区间的音符、休止符，归属到该 Measure Object 中。

-----

### 第三阶段：属性识别 (Attributes)

目标：确定每个小节的“环境参数”（谱号、调号、拍号）。这些属性具有**状态保持性 (State Persistence)**。

#### 规则 4: 谱号判定 (Clef Detection)

对于每个 $Measure_{i}$ 中的每个 $Staff_{j}$:

1.  **搜索**: 在该小节的 $Staff_j$ 区域内搜索 `g-clef`, `f-clef`, `c-clef`。
2.  **判定**:
      * 如果找到: 更新当前 Staff 的 `Current_Clef` 状态。
      * 如果没找到:
          * 如果是全曲第一个小节 ($Measure_1$): 必须强制指派 (默认 Part 1=Treble, Part 2=Bass，并标记 Low Confidence)。
          * 如果不是第一小节: 继承上一个小节 ($Measure_{i-1}$) 的 `Current_Clef`。

#### 规则 5: 调号判定 (Key Signature)

调号通常出现在谱号之后，拍号之前。

1.  **识别模式**:
      * **模式 A (整体标签)**: 如果检测到 `key_signature` 标签，直接使用。
      * **模式 B (离散符号聚类)**:
          * 在谱号右侧、拍号/第一个音符左侧的区域。
          * 统计 `sharp` (\#) 或 `flat` (b) 或 `natural` 的数量。
          * 聚类逻辑: 如果一组升降号的 X 轴距离非常近 (within 1.5 note width)，视为同一组调号。
2.  **语义解析**:
      * 1个 `#` -\> G Major / E Minor
      * 3个 `b` -\> Eb Major / C Minor
      * 没有任何升降号 -\> C Major / A Minor
3.  **状态更新**: 同样遵循“有则更新，无则继承”原则。

#### 规则 6: 拍号判定 (Time Signature)

1.  **搜索**:
      * 寻找 `time_signature` 标签。
      * 寻找 `letter_c` (Common Time = 4/4)。
      * 寻找组合数字 (Vertical Digit Stack): 比如 `numeral_3` 在 `numeral_4` 正上方。
2.  **状态更新**:
      * 有则更新，无则继承。
      * **异常检测**: 如果一个小节内的音符实际时值总和与当前拍号严重不符（且不是弱起小节），在 Output log 中标记 `TimeSignature Mismatch`。

-----

### 第四阶段：全局索引与组装 (Global Assembly)

#### 规则 7: 全局小节计数 (Global Measure Indexing)

MusicXML 需要连续的小节编号。

  * 初始化 `Global_Index = 1`。
  * 按 System 顺序 ($Sys_1 \rightarrow Sys_N$) 遍历:
      * 按水平顺序遍历该 System 的 Measure ($M_1 \rightarrow M_k$):
          * 如果不属于 **多小节休止 (Multi-measure Rest)**:
              * $M.number = Global\_Index$
              * $Global\_Index += 1$
          * 如果检测到 `multi-measure_rest` (上方有数字 N):
              * $M.number = Global\_Index$
              * $Global\_Index += N$ (跳过 N 个小节号)

#### 规则 8: 弱起小节处理 (Anacrusis / Pickup Measure)

  * **条件**: 全曲第一个小节 ($Measure_1$)。
  * **检查**: 计算该小节内音符总时值 $D_{sum}$。
  * **逻辑**: 如果 $D_{sum} < TimeSignature_{expected}$ (例如 4/4 拍里只有 1 拍):
      * 标记该小节为 `implicit="yes"` (MusicXML 属性)。
      * 不报错，视为正常的弱起小节。

这套规则专注于解决乐谱识别中最棘手的问题之一：**三连音 (Triplets) 与多连音 (Tuplets) 的判定**。

三连音的难点在于它有多种形态：有括号的、没括号只有数字的、连在符杠(Beam)上的。我们需要建立一套从“强约束”到“弱约束”的级联判定机制。

---

### 定义与预处理

* **输入对象**:
    * $N_{sorted}$: 已按 X 轴排序的音符对象列表 $\{n_1, n_2, ...\}$。
    * $S_{tag}$: 相关的标记符号集合，包含 `numeral_3` (数字3), `tuple_bracket/line` (连音线/括号)。
    * $B$: 符杠 (Beam) 集合。

* **搜索区域定义 ($Region_{search}$)**:
    * 对于数字或符号，其有效影响范围通常在垂直方向上跨越五线谱的高度，水平方向上覆盖其 Bounding Box 的宽度。

---

### 规则表：三连音判定逻辑

#### 规则 1: 基于括号的判定 (Bracket-Based) [优先级最高]
这是最明确的指示，通常出现在无符杠的音符组（如三个四分音符）上方。

1.  **遍历**: 所有 `class` 为 `tuple_bracket/line` 的符号 $b_{bracket}$。
2.  **空间投影**:
    * 获取 $b_{bracket}$ 的 X 轴区间 $[x_{start}, x_{end}]$。
    * 在 $N_{sorted}$ 中寻找所有 **符头中心点 X 坐标** 落在 $[x_{start}, x_{end}]$ 范围内的音符，记为候选组 $G_{cand}$。
3.  **筛选**:
    * 检查 $G_{cand}$ 中的音符与 $b_{bracket}$ 的垂直距离。如果距离过远（超过一个五线谱高度），剔除该音符（防止把另一行谱的音符算进去）。
4.  **数字关联**:
    * 在 $b_{bracket}$ 的中心点附近搜索 `numeral_3`。
    * **判定**:
      * 如果找到 `numeral_3` 且 $G_{cand}$ 有 3 个音符 -> 确认三连音。
      * **放松条件**: 如果没找到数字，但 $G_{cand}$ **恰好有 3 个音符** -> 也确认为三连音（推断省略了数字）。
    * 确认 $G_{cand}$ 为三连音。

#### 规则 2: 基于符杠的判定 (Beam-Based) [优先级次之]
这是最常见的情况，八分音符或十六分音符的三连音通常由符杠连接。

1.  **遍历**: 所有 `beam` 对象。
2.  **筛选**: 找出连接了 **3个或3的倍数** 个符干的 Beam 组。记为 $G_{beam}$。
3.  **数字搜索**:
    * 定义搜索区域 $R$: 以 Beam 的中心点为圆心，半径为 2倍行间距的区域。
    * 在 $R$ 中搜索 `numeral_3`。
    * **碰撞检查**: 检查 `numeral_3` 的 BBox 是否与 Beam 有重叠，或者位于 Beam 的正上方/正下方。
4.  **指法排除 (Finger Number Exclusion)**:
    * 如果找到的 `numeral_3` **严格垂直对齐** 于某一个单一符头的正上方（且距离极近），这可能是指法（指法通常标在符头侧，三连音标在符杠侧）。
    * **三连音特征**: 数字通常位于这组音符的**几何中心** X 轴位置。
5.  **判定**:
    * 如果满足上述条件，标记该 Beam 下的所有音符为三连音。

#### 规则 3: 孤立数字判定 (Loose Number) [优先级最低]
处理没有括号、没有符杠，只有一个 "3" 悬浮在三个音符上方的情况（常见于密集排版）。

1.  **遍历**: 剩余未被使用的 `numeral_3` 符号（排除时间签名的一部分）。
2.  **基于距离的搜索 (Distance-Based Search)**:
    * 以 `numeral_3` 的中心为原点，搜索周围所有音符（搜索半径通常为 6倍行间距）。
    * 计算每个音符到数字 "3" 的 **欧几里得距离 (Euclidean Distance)**。
    * 选取 **距离最近的 3 个音符**。
3.  **判定**:
    * **条件 A (完整性)**: 这 3 个最近的音符必须全部未被之前的规则处理（Unprocessed）。
    * 如果满足条件，推断为三连音。
    * *注意：此规则不再校验间距均匀性或时值一致性，而是完全依赖几何邻近度。*
    * **指法排除 (Finger Number Exclusion)**:
      * 如果数字 "3" 正好位于某个音符的**正上方/正下方**（垂直对齐且距离近），则被视为指法标记，**不触发**三连音判定。
      * 这是一个关键的防错机制，防止指法被误判为三连音。

#### 规则 4: 属性赋值与 MusicXML 生成
一旦一组音符 $G = \{n_1, n_2, n_3\}$ 被判定为三连音，执行以下修改：

1.  **时值修改**:
    * 对于 $n \in G$:
    * `time_modification_actual_notes` = 3
    * `time_modification_normal_notes` = 2
    * `duration_xml` = `base_duration` $\times \frac{2}{3}$

2.  **XML 标签生成**:
    * $n_1$ (第一个音): 添加 `<tuplet type="start" bracket="yes/no"/>`
    * $n_2$ (中间音): 不添加 tuplet 标签（或根据 MusicXML 版本需求处理）。
    * $n_3$ (最后一个音): 添加 `<tuplet type="stop"/>`

3.  **括号显示逻辑**:
    * 如果是由 **规则 1** (Bracket) 触发: `bracket="yes"`。
    * 如果是由 **规则 2** (Beam) 触发: 通常 `bracket="no"` (因为有 Beam 了，不需要括号，数字通常显示在 Beam 侧)。
    * 如果是由 **规则 3** (Loose) 触发: `bracket="yes"` (为了阅读清晰，通常生成时强制加上括号)。

#### 规则 5: 数学校验 (Sanity Check) [兜底]

在小节结束时进行校验：
* **计算**: `Current_Measure_Duration` = $\sum n.duration$。
* **检测**: 如果 `Current_Measure_Duration` > `Time_Signature_Duration` + Tolerance (即小节溢出)。
* **搜索与修正**:
    * 遍历小节内所有未处理的连续 3 个音符组 $\{n_i, n_{i+1}, n_{i+2}\}$。
    * **条件 A (时值一致)**: 这 3 个音符必须具有相同的 `Base_Duration`。
    * **条件 B (修复溢出)**: 假设这 3 个音符是三连音，计算它们减少的总时值 $Reduction = Base\_Duration$。
    * **验证**: 如果 `Current_Measure_Duration - Reduction` 接近 `Time_Signature_Duration`。
    * **条件 C (无数字冲突)**: 检查这组音符区域内是否确实**没有**检测到 `numeral_3` (防止与已有规则冲突)。
    * 如果所有条件满足，**强制转换**为三连音并标记 `Confidence="Low"` (Implicit Triplet)。

---

### 可视化调试建议 (Output/test)

在生成的调试图片中：

1.  **Tuplet Grouping**:
    * 用 **紫色矩形框** 框住被判定为一组的三个音符。
    * 在框上方标注文本: `Triplet (Rule 1: Bracket)` 或 `Triplet (Rule 2: Beam)`.
2.  **Trigger Symbol**:
    * 用高亮颜色（如黄色）圈出触发该规则的 `numeral_3` 或 `bracket`，并画线指向对应的音符组。
3.  **Error Highlight**:
    * 如果 Rule 5 触发了强制修正，用 **红色虚线框** 框住，并标注 `Implicit Triplet detected`.

---

### 第五阶段：音高判定 (Pitch Determination)

#### 规则 9: 音高计算 (Pitch Calculation)

目标：根据符头在五线谱中的垂直位置计算音高名称（如 C4, F#5）。

1.  **参考基准 (Reference Standard)**:
    *   以五线谱的 **最上方一条线 (Top Line)** 作为基准线 (Step 0)。
    *   定义各谱号在 Top Line 的参考音高:
        *   **Treble Clef (高音谱号)**: Top Line = **F5**
        *   **Bass Clef (低音谱号)**: Top Line = **A3**
        *   **Alto Clef (中音谱号)**: Top Line = **G4**

2.  **步数计算 (Step Calculation)**:
    *   计算符头中心 $y_{note}$ 与 Top Line $y_{top}$ 的垂直距离 $\Delta y = y_{note} - y_{top}$。
    *   计算半个线间距 (Half-Spacing) $h = \frac{\text{Average Line Spacing}}{2}$。
    *   计算下移步数 (Steps Down): $Steps = \frac{\Delta y}{h}$。
    *   **取整**: 将 $Steps$ 四舍五入到最近的 0.5 整数倍 (Round to nearest 0.5)。
        *   整数步表示在线上或间上。

3.  **音高推导 (Derivation)**:
    *   根据参考音高向下移动 $Steps$ 个自然音级 (Diatonic Steps)。
    *   示例 (Treble Clef):
        *   Steps = 0 (Top Line) -> F5
        *   Steps = 1 (Space below Top Line) -> E5
        *   Steps = 2 (Second Line from top) -> D5
        *   ...

4.  **变音记号 (Accidentals)**:
    *   首先计算**视觉自然音高** (Visual Natural Pitch)。
    *   如果符头关联了变音记号 (Sharp/Flat/Natural) 或受到调号 (Key Signature) 影响，修改音高名称。
    *   *注：当前实现优先使用关联的局部变音记号，其次应用调号。*

