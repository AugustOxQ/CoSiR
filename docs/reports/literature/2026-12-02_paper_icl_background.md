# Literature check: in-context learning background and the novelty boundary for the CVPR paper

**Written:** 2026-10-13 16:15 (Amsterdam). The file name uses CoSiR's sequence date (2026-12-02).
**Model and depth:** Opus, deep.
**Questions:** (1–3) Which in-context learning (ICL) work should §2.3 cite: in LLMs, in multimodal models, in visual ICL, and in analyses that show MLLMs make little use of demonstrations? (4) Has any work let test-time demonstrations define the notion of similarity for retrieval, and which of our five properties does each close hit lack?
**Builds on:** `docs/paper/cvpr2027_aspect_episodes/notes/related_work_digest.md` (§1.4, §1.6, §3, §5), `2026-10-24_aspect_task_novelty_check.md` and `2026-11-25_held_claim_check.md`. Rows that those notes already verified (BGE-EN-ICL, RICE, Promptagator, Qwen3-VL-Embedding, VLM2Vec/GME/Lens, GeneCIS, CLAY, TPIPS, VisDiff, MARS) are not repeated here.
**Method:** 22 web searches and 20 page reads. Three arXiv API queries searched titles and abstracts directly. DBLP blocked us (Anubis bot check), and the CVF page for SegGPT returned 403. For each paper we opened an arXiv abstract page or API record, a NeurIPS proceedings or virtual page, or the ACL Anthology page.

## 1. Answer first

- **Novelty (Q4): to our knowledge, no paper has all five properties.** Since the October notes we found two more text-only in-context retrievers (RARe, COLM 2025; few-shot pairwise rank prompting, EMNLP 2024) and one more text-named conditional image embedder (DIOR, preprint). We found no multimodal embedder or reranker whose criterion comes only from demonstrations. The digest's novelty sentence still holds; §4 gives an updated version that cites these papers.
- **Q3: the field has shown that MLLMs make limited use of demonstrations.** This is the strongest support for our MLLM result. VL-ICL Bench (ICLR 2025) reports small gains over zero-shot, gains that do not grow with more shots, and on operator induction "a very common mistake is to use a different operator than what would be induced from the support examples". This matches our finding that the MLLM does not choose the shown aspect. Several more papers report that demonstration text drives multimodal ICL and demonstration images do little: Chen et al. (WACV 2025), Baldassini et al. (CVPR workshop 2024), Chen et al. (COLM 2025), and dos Santos et al. (preprint). Huang et al. (preprint) report copying of demonstrated answers and drops with more shots.
- **Q1:** cite Brown et al. (NeurIPS 2020) for ICL, Dong et al. (EMNLP 2024) for the survey, and Min et al. (EMNLP 2022) for "demonstrations mostly show format and label space". Wei et al. (preprint, 2023) contradicts Min et al. for large models, so cite both or neither.
- **Q2:** for interleaved MLLM ICL, cite Flamingo (NeurIPS 2022), plus MMICL (ICLR 2024) or Emu2 (CVPR 2024). For visual ICL, cite visual prompting by inpainting (NeurIPS 2022), Painter (CVPR 2023), SegGPT (ICCV 2023), and Zhang et al. (NeurIPS 2023) for demonstration choice.
- **Positioning:** in all of this prior work, demonstrations define a *task output*: a label, an answer, a mask or a relevance judgment. In ours they define *which notion of similarity* ranks candidates across modalities. The examples are value-disjoint from the query and paired with a contrast aspect. We can state this difference, but not "demonstrations define the task" as new.

## 2. Findings by question

### Q1. ICL in LLMs

- **Brown et al. 2020** (NeurIPS 2020) introduced GPT-3. It does tasks from a few demonstrations in the prompt, with no gradient updates. This is the standard founding citation.
- **Dong et al. 2024** (EMNLP 2024, pp. 1107–1128) is the standard survey. It covers the definition, training, prompt and demonstration design, and analysis. The ACL Anthology lists "Lei Li" twice among the 13 authors; keep both names, as the Anthology does.
- **Min et al. 2022** (EMNLP 2022) found across 12 models that random labels in demonstrations "barely hurt" performance. The value of demonstrations lies in showing "the label space, the distribution of the input text, and the overall format".
- **Contradiction: Wei et al. 2023** (arXiv preprint) found that this depends on scale. Small models ignore flipped labels, but large models can override their semantic priors and learn even unrelated label mappings.

*Positioning:* our demonstrations carry no labels at all. The aspect must be inferred from what the pairs share. A reader may therefore ask whether an MLLM reads only the format. The MLLM's near-zero condition gain is consistent with that reading, and the matched control measures it directly.

### Q2. Multimodal ICL and visual ICL

- **Flamingo** (Alayrac et al., NeurIPS 2022): a model that reads interleaved image–text input and learns new tasks from a few example pairs in the prompt.
- **OpenFlamingo** (Awadalla et al., 2023): a technical report and open replication. **MMICL** (Zhao et al., ICLR 2024): trains for multi-image, multi-modal ICL. **Emu2** (Sun et al., CVPR 2024): "Generative multimodal models are in-context learners".
- **Visual ICL:**
  - Bar et al. (NeurIPS 2022) cast visual prompting as inpainting a grid of example input–output images.
  - Painter (Wang et al., CVPR 2023) is a generalist model that takes image pairs as prompts.
  - SegGPT (Wang et al., ICCV 2023) trains in-context segmentation as coloring with random color maps.
  - Zhang et al. (NeurIPS 2023) show that visual ICL is highly sensitive to which demonstrations are chosen, and they propose retrieving demonstrations.

*Positioning:* in visual ICL, each demonstration pairs an input with its output *on the same item* (image → mask), and the model produces an output. In our task, each pair joins *two different items* across modalities, and the model must rank, not generate.

### Q3. Analyses: limited use of demonstrations by MLLMs

The findings below are in their own settings, so §2.3 and §7 can say "consistent with" honestly.

| Paper | Setting | Finding (the paper's words where quoted) |
|---|---|---|
| Zong et al., VL-ICL Bench (ICLR 2025) | About 20 MLLMs, including OpenFlamingo, IDEFICS 1/2, Qwen-VL-Chat, LLaVA-Next, LLaVA-OneVision 0.5B/7B/72B and GPT-4V. Tasks: VQA and captioning, plus 10 new tasks such as operator induction, fast concept binding and fast matching. | On VQA and captioning, "most … models show only limited improvement with ICL … compared to the zero-shot case". With LLM scoring, the benefit almost vanishes. "VLLMs often struggle to make use of a larger number of ICL examples." On operator induction, models often "use a different operator than what would be induced from the support examples". On Fast Matching, the 0.5B and 7B LLaVA-OneVision models fall below chance as shots increase; the 72B model improves. |
| Chen et al. 2023/2025 (WACV 2025) | Open MLLMs on VQA and captioning | Multimodal ICL is "primarily driven by textual information"; demonstration images have little effect. |
| Baldassini et al. (CVPR 2024 Workshop on Prompting in Vision) | Interleaved MLLMs | ICL depends mainly on text, with little influence from images. RICES demonstration selection does no better than majority voting over the demonstrations. |
| Chen et al., TrueMICL (COLM 2025) | MLLMs on a new dataset that needs the demonstration images | Models "neglect visual cues and over-rely on textual patterns", "leading to mere text imitation rather than genuine multimodal adaptation". |
| Huang et al. (preprint, 2025) | Open VLMs from 3B to 72B and Gemini 2.0; demonstrations drawn from a different distribution than the query | "Performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them." Adding rationales changes little. |
| dos Santos et al. (preprint, 2025) | 7 models, 4 architectures, 3 captioning benchmarks | Models "primarily focus on textual cues and fail to leverage visual information". Instruction tuning "can reduce reliance on in-context demonstrations". |
| Wüst et al., Bongard in Wonderland (ICML 2025 per the listing; **not verified**) | VLMs on Bongard problems: infer a rule from positive and negative image sets | VLMs often fail, even on simple concepts. |

*Positioning:* these papers study generative tasks (answers, captions, labels), not ranking, and most use a single modality per demonstration slot. Our result extends the pattern to a ranking task: the MLLM's condition gain is 0.12 [−0.17, +0.39], so it does not choose the demonstrated aspect. That is "consistent with" these findings, not a replication. We did not test *why*, whether by text reliance or by applying a different rule. The VL-ICL operator-induction finding is the closest match in kind.

### Q4. Novelty boundary: demonstrations that define similarity or relevance

We did not find a paper with all five properties: (a) aspect never named; (b) examples value-disjoint from the query; (c) cross-item example pairs; (d) a contrast aspect and a swap test; (e) cross-modal image↔text. The new close hits, and what each lacks:

| Work | What the examples do | Lacks |
|---|---|---|
| RARe (Tejaswi et al., COLM 2025) | Prepends query–document pairs whose queries are similar to the test query; fine-tunes the retriever on this input. It reports that naive prepending "does not work out of the box" for retrievers. | (b): examples are chosen to *resemble* the query. (d), (e): text only. (a) only partly: the query text carries the intent. |
| Few-shot pairwise rank prompting (Sinhababu et al., EMNLP 2024; ACL Anthology lists it as Findings) | Preference examples from similar training queries guide an LLM's pairwise reranking. | (b), (d), (e); text. |
| BGE-EN-ICL, RICE, Promptagator (already in the digest) | Task-level example pairs in the encoder or LLM context | (d), (e); text. |
| DIOR (Kawarada et al., preprint, 2512.21860) | Training-free conditional image embedding from an LVLM's one-word answer | (a): the condition is named in text, with no demonstrations. (e): image only. |
| Visual ICL (Bar; Painter; SegGPT) | Input–output pairs of the same item define a dense task | (c): same-item pairs. (b) does not apply. (d). (e): output is an image, not a ranking. |
| MLLM ICL (Flamingo, MMICL, VL-ICL's matching tasks) | Interleaved image–text demonstrations define an answer mapping | No ranking. No value-disjoint design. (d). |
| Bongard-style set induction (digest; Wüst et al.) | Positive and negative image sets define a concept | (b): the positives share the value. (c), (e): image only, and classification rather than retrieval. It has a negative set but no swap. |
| Look-alike direction (arXiv 2511.19200, abstract only) | An embedding direction estimated from "real" and "lookalike" exemplars, applied to image–text retrieval | (a): the criterion is named. The direction is estimated once, not taken from each query's demonstrations. (b), (d). |

Three arXiv API queries and eight web searches for multimodal in-context embedding or reranking found nothing closer. These covered 2023 to 2026-10, with arXiv entries to September and October 2026. The last arXiv sweep before the abstract (digest §3, HC 3.1) still applies.

## 3. Citation table

| Key | Authors | Title | Venue, year | arXiv / DOI | URL opened | Status | Role |
|---|---|---|---|---|---|---|---|
| brown2020language | Brown, Mann, Ryder et al. (31) | Language Models are Few-Shot Learners | NeurIPS 2020 | 2005.14165 | https://papers.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html | verified | §2.3 background |
| dong2024survey | Dong, Li, Dai, Zheng, Ma, Li, Xia, Xu, Wu, Chang, Sun, Li, Sui | A Survey on In-context Learning | EMNLP 2024 | 2301.00234; 10.18653/v1/2024.emnlp-main.64 | https://aclanthology.org/2024.emnlp-main.64/ | verified | §2.3 background |
| min2022rethinking | Min, Lyu, Holtzman, Artetxe, Lewis, Hajishirzi, Zettlemoyer | Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? | EMNLP 2022 | 2202.12837 | https://arxiv.org/abs/2202.12837 | verified (venue from the arXiv comment) | §2.3 background |
| wei2023larger | Wei, Wei, Tay, Tran et al. (11) | Larger language models do in-context learning differently | preprint 2023 | 2303.03846 | arXiv API record | verified (preprint) | §2.3 background (contradicts Min) |
| alayrac2022flamingo | Alayrac, Donahue, Luc, Miech et al. | Flamingo: a Visual Language Model for Few-Shot Learning | NeurIPS 2022 | 2204.14198 | https://arxiv.org/abs/2204.14198 | verified (venue from the arXiv page) | §2.3 background |
| awadalla2023openflamingo | Awadalla, Gao, Gardner, Hessel et al. (16) | OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models | preprint 2023 | 2308.01390 | arXiv API record | verified (preprint) | §2.3 background (optional) |
| zhao2024mmicl | Zhao, Cai, Si, Ma et al. (10) | MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning | ICLR 2024 | 2309.07915 | arXiv API record | verified (venue from the arXiv comment) | §2.3 background |
| sun2024generative | Sun, Cui, Zhang, Zhang et al. (11) | Generative Multimodal Models are In-Context Learners | CVPR 2024 | 2312.13286 | arXiv API record | verified (venue from the arXiv comment) | §2.3 background |
| bar2022visual | Bar, Gandelsman, Darrell, Globerson, Efros | Visual Prompting via Image Inpainting | NeurIPS 2022 | 2209.00647 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html | verified | §2.3 background; novelty boundary |
| wang2023images | Wang, Wang, Cao, Shen, Huang | Images Speak in Images: A Generalist Painter for In-Context Visual Learning | CVPR 2023 | 2212.02499 | arXiv API record | verified (venue from the arXiv comment) | §2.3 background; novelty boundary |
| wang2023seggpt | Wang, Zhang, Cao, Wang, Shen, Huang | SegGPT: Towards Segmenting Everything in Context (arXiv title: "SegGPT: Segmenting Everything In Context") | ICCV 2023, pp. 1130–1140 | 2304.03284 | arXiv API record; the CVF page returned 403 | arXiv verified; **ICCV venue and pages not verified** (seen only in a search snippet) | §2.3 background |
| zhang2023makes | Zhang, Zhou, Liu | What Makes Good Examples for Visual In-Context Learning? | NeurIPS 2023 | 2301.13670 | https://neurips.cc/virtual/2023/poster/70423 | verified | §2.3 background |
| zong2025vlicl | Zong, Bohdal, Hospedales | VL-ICL Bench: The Devil in the Details of Multimodal In-Context Learning | ICLR 2025 | 2403.13164 | https://arxiv.org/html/2403.13164v4 | verified (quotes taken through the fetch tool; re-check the wording before quoting) | §7 consistent-with |
| chen2025can | Chen, Han, He, Liu et al. (9) | Can Multimodal Large Language Models Truly Perform Multimodal In-Context Learning? | WACV 2025 | 2311.18021 | arXiv API record | verified (venue from the arXiv comment) | §7 consistent-with |
| baldassini2024makes | Baldassini, Shukor, Cord, Soulier, Piwowarski | What Makes Multimodal In-Context Learning Work? | CVPR 2024 Workshops (Prompting in Vision) | 2404.15736 | arXiv API record | verified (workshop paper) | §7 consistent-with |
| chen2025true | Chen, Liu, Han et al. (8) | True Multimodal In-Context Learning Needs Attention to the Visual Context | COLM 2025 | 2507.15807 | arXiv API record | verified (venue from the arXiv comment) | §7 consistent-with |
| huang2025mimicking | Huang, Zhu, Zhu et al. (7) | Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models | preprint 2025 | 2506.07936 | arXiv API record | verified (preprint; no venue found) | §7 consistent-with |
| dossantos2025what | dos Santos, Colombini, Avila | What do vision-language models see in the context? Investigating multimodal in-context learning | preprint 2025 | 2510.24331 | arXiv API record | verified (preprint) | §7 consistent-with (optional) |
| wust2025bongard | Wüst et al. | Bongard in Wonderland: Visual Puzzles that Still Make AI Go Mad? | ICML 2025 (per a listing) | 2410.19546 | search snippet only | **not verified** (no primary page opened) | §7 consistent-with (optional) |
| tejaswi2025rare | Tejaswi, Lee, Sanghavi et al. (4) | RARe: Retrieval Augmented Retrieval with In-Context Examples | COLM 2025 | 2410.20088 | arXiv API record | verified (venue from the arXiv comment) | novelty boundary |
| sinhababu2024few | Sinhababu, Parry, Ganguly et al. (5) | Few-shot Prompting for Pairwise Ranking: An Effective Non-Parametric Retrieval Model | EMNLP 2024 (Findings, per the ACL Anthology URL in a snippet) | 2409.17745 | arXiv API record | verified on arXiv; **"Findings" not verified** | novelty boundary |
| kawarada2025training | Kawarada, Yamada, Tejero-de-Pablos et al. (4) | Training-free Conditional Image Embedding Framework Leveraging Large Vision Language Models | preprint 2025 | 2512.21860 | arXiv API record | verified (preprint) | novelty boundary |

## 4. Wording for §2.3

> Large language models learn tasks from a few demonstrations in their context [brown2020language; dong2024survey], although demonstrations often convey format and label space more than the input–label mapping [min2022rethinking; wei2023larger]. Interleaved vision–language models extend this to image–text demonstrations [alayrac2022flamingo; zhao2024mmicl; sun2024generative], and visual in-context learning lets input–output image pairs define a dense prediction task [bar2022visual; wang2023images; wang2023seggpt; zhang2023makes]. Benchmarks and analyses find that MLLMs gain little from demonstrations, lean on their text, and often apply a rule other than the one shown [zong2025vlicl; chen2025can; baldassini2024makes; chen2025true; huang2025mimicking]; our in-context MLLM comparator behaves consistently with these findings on a ranking task.

**Updated novelty sentence:** this is the digest's NC §4 statement, extended with the new rows.

> To our knowledge, prior work lets demonstrations define a task output in language [brown2020language], in images [bar2022visual; wang2023images] or in text retrieval and reranking [li2024making; jedidi2026effective; tejaswi2025rare; sinhababu2024few], or conditions image similarity on a named criterion [kawarada2025training; CLAY]; none lets value-disjoint, cross-item image–caption demonstrations with a contrast aspect fix the notion of cross-modal similarity, tested in both directions with a paired swap.

The keys li2024making (BGE-EN-ICL) and jedidi2026effective (RICE) follow the digest's rows. Confirm them against `refs.bib`.

## 5. Claims we must not make (new ones only)

- Do not say that "demonstrations define the task" or "in-context retrieval" is new. RARe and few-shot pairwise rank prompting join BGE-EN-ICL, RICE and Promptagator here.
- Do not say that our MLLM result shows *why* MLLMs fail, for example by text reliance or by attention. We did not test the mechanism, so write "consistent with [zong2025vlicl; …]".
- Do not say we are first to show that MLLMs fail to infer a rule from demonstrations. VL-ICL's operator induction and Bongard-style benchmarks already show this.
- Do not cite Min et al. alone for "labels do not matter". Wei et al. show that the effect depends on scale.
- Do not cite SegGPT under its arXiv title as ICCV. The ICCV title has "Towards". Do not cite Brown et al. as arXiv only; it is NeurIPS 2020.
- RARe reports that naive in-context prepending does not work for retrievers without fine-tuning. If we report a RICE-style comparator, do not present a weak result as a finding of ours alone.

## 6. Open points

1. **SegGPT's ICCV page and Bongard in Wonderland's ICML page were not opened** (403 and budget). Settle them with the CVF ICCV 2023 open-access page and the PMLR v267 page.
2. **VL-ICL's "Fast Matching" and "Fast Attribute Matching" tasks** were seen by name only. They may be the closest MLLM-side protocol to ours. Read their task definitions in arXiv 2403.13164v4, §3, before §7 relies on them.
3. **Exact settings of Baldassini et al. and Chen et al. 2023** (models, shot counts) were read at the abstract level only. Open their result sections before quoting numbers.
4. **Huang et al. (2506.07936) and dos Santos et al. (2510.24331)** have no venue that we found. Check OpenReview and recent proceedings before the camera-ready.
5. **Re-run the arXiv sweep the week before the abstract** (digest §3). Use the query "in-context" AND (embedding OR rerank*) AND (image OR multimodal) AND (demonstration* OR example*), sorted by date.

Sources opened: arxiv.org/abs/2202.12837, arxiv.org/abs/2301.00234, arxiv.org/abs/2005.14165, arxiv.org/abs/2204.14198, export.arxiv.org API records (2209.00647, 2212.02499, 2304.03284, 2301.13670, 2308.01390, 2309.07915, 2312.13286, 2403.13164, 2404.15736, 2311.18021, 2303.03846, 2507.15807, 2510.24331, 2506.07936, 2505.16193, 2410.20088, 2512.21860, 2607.27959, 2602.21854, 2610.10508, 2409.17745, 2409.15700, 2209.11755), arxiv.org/html/2403.13164v4, papers.neurips.cc (Brown 2020; Bar 2022), neurips.cc/virtual/2023/poster/70423, aclanthology.org/2024.emnlp-main.64, and three arXiv API abstract searches.

Note on excluded items: arXiv 2608.23570 ("Taming Visual Neglect") showed a mismatch between its ID and its publication date in the API, so we did not cite it. arXiv 2505.16193 (ICL for MLLM sentiment) is an application paper with no bearing on the boundary.
