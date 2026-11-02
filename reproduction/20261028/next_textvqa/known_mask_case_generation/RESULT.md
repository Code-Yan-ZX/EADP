# 已知 changed-mask 个案：source 与 Eq6 的有界生成对照

固定位置3632、原面板image11的一次已知个案，两臂返回的 **129个token ID与文本完全相同**。129包含起始BOS1和128个新token；模型配置EOS为2，两臂均未出现EOS，已达到本次`max_new_tokens=128`范围。因此结论限于该horizon，不能称自然终止回答，也不能排除logits差异、更长生成分歧或其他图的差异。

独立CPU终端审查实际执行一次、exit0，未运行原verify、评分器或模型，未读取GT或旧答案。逐项核验了root终端证明、plan／allocation／claim／release／独立报告／实际输出文件的SHA链，12份source、9份冻结证据和20份模型stat。

- 原生spatial路径先base crop，再将4个local crop按2×2布局交错成48×48 raster。独立坐标遍历重建顺序后，两臂各637行实际`inputs_embeds`与lossless原生FP16 projected缓存逐bit一致；完整嵌入shape均为`[1,884,4096]`。
- 两臂quota均为`[146,162,106,130,93]`，总637。Eq6 mask在crop3移除token330、加入115；顺序中83行视觉位置改变，位置为465–547。非视觉前35行、后212行逐bit相同。
- 每臂各一次原生prepare／generate，CLIP和projector forward均0；重用同一cached features对象。原生prepare／generate源码保持冻结，观察发生在generate返回后。
- 根PTY89061实际exit0；controller2774786/ticks1333118631与native2774847/ticks1333118719均已退出，独立复查时原`/proc`路径与GPU PID均不存在。

原生feature／selection提取锚点是beta1／CLI128。这里beta2 source mask来自同一精确特征上的原source helper replay，不能写成另一次beta2端到端提取锚点。这个对照验证的是已声明的cached-mask干预，不是新的完整benchmark或无偏采样。

关键证据：

| 工件 | SHA256 |
|---|---|
| 冻结plan | `34b815cc26db836ad6bc9f7ea69eb08956680d350a86b8e7a697c904545a84c3` |
| 根实际终端证明 | `77dca7df064b943cd61b309943800827c9dd1622362e7a2c126a69968c86a068` |
| 原完整独立报告 | `637a1adf82a3757dbe8b14f6e927dc5c5d4b4a608facf0ebce14cbe638f8ef7a` |
| 新peer终端审查 | `a5fe64f95e337624bb48c1f8115f5ac1293c38d1905689aaeb88046b2fe5a1f6` |

完整输入数组、模型配置BOS／EOS、文件SHA和限制见[peer审查](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/known_mask_case_generation/peer_terminal_review.json)。新审查只核对记录文本相等及输出ID，不重复token重解码；原完整终端报告已核两臂重解码。

本个案没有GT、accuracy或benchmark主行，不改变七个剪枝目标仍未复现的状态，也不自动支持全量Eq6变体或再次延长生成。
