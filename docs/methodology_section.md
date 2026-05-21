# METHODOLOGY

This section describes the methodology followed in this project, including the dataset, technologies, and proposed pipeline. The main stages include data collection, preprocessing, aspect classification, evidence retrieval, source attribution, model fine-tuning, and evaluation. The purpose of this section is to explain the process clearly enough to make the study reproducible, while also documenting the experimental decisions made during development: what alternatives were tried, which approaches were rejected, and why the final methods were selected.

## 3.1. Overview of the Dataset/Model

This project focuses on Turkish e-commerce reviews and aims to generate short, aspect-specific summaries from large sets of product comments. The problem was defined as aspect-based multi-review summarization: instead of producing a single general summary for a product, the system identifies opinion evidence for predefined aspects and produces one concise Turkish sentence for each aspect when enough evidence exists.

The final aspect set contains five categories: `kargo`, `paketleme`, `fiyat_performans`, `kalite`, and `genel_memnuniyet`. At the beginning of the project, the taxonomy contained four aspects: shipping, packaging, price-performance, and quality. During data inspection, however, many general satisfaction statements were being absorbed into `kalite`. For example, comments such as "tavsiye ederim", "memnun kaldim", or "tekrar alirim" do not necessarily describe material quality or durability, but they clearly indicate general satisfaction. Because these sentences were semantically different from true quality statements, a fifth aspect, `genel_memnuniyet`, was added. This change made the class definitions cleaner and reduced confusion between product quality and general user satisfaction.

The aspect definitions used in the final system are shown below.

| Aspect | Meaning |
|---|---|
| `kargo` | Delivery process, delivery speed, courier experience, and shipping performance |
| `paketleme` | Packaging quality, box condition, protection of the product, and whether the product arrived damaged |
| `fiyat_performans` | Whether the product is worth its price and whether users perceive good value |
| `kalite` | Material quality, durability, build quality, freshness, workmanship, and product-level quality |
| `genel_memnuniyet` | Overall satisfaction, recommendation intent, and willingness to purchase again |

The dataset was collected from Turkish e-commerce platforms. Amazon Turkey was used both for dataset construction and for the live demo pipeline. Trendyol was also used during dataset expansion to increase platform variety and expose the model to different review styles, seller structures, and product categories. Amazon was especially useful for the demo because each product has a clear ASIN, product URL, product title, review pages, ratings, and review dates.

The data collection process was carried out gradually. First, a pilot dataset was collected to verify whether the complete pipeline could work end to end. The pilot targeted approximately 5 categories, 10 products per category, and around 80 reviews per product, producing roughly 6,800 reviews. This early dataset was not intended to be the final training corpus; it was used to debug scraping, preprocessing, sentence splitting, labeling, and retrieval. After the pipeline was validated, the dataset was expanded to more than 1,000 products and approximately 80,000 reviews. In the later stage, the collection reached more than 7,000 products and approximately 420,000 reviews.

The project stores review-level data first, then converts it into sentence-level data. This is necessary because a single review can mention multiple aspects. For example, "Urun kaliteli, fiyatina gore iyi ama kargo gec geldi" includes quality, price-performance, and shipping evidence in one review. Therefore, document-level processing would mix different opinions, while sentence-level processing allows each claim to be classified and retrieved separately.

The raw review schema includes the following fields.

| Field | Description |
|---|---|
| `review_id` | Platform review identifier when available |
| `asin` | Amazon product identifier |
| `title` | Review title |
| `rating` | Parsed star rating |
| `body` | Main review text |
| `review_date` | Review date converted to ISO format |
| `product_title` | Product title from the e-commerce page |
| `product_description` | Product bullet points or available product description |
| `product_url` | Product page URL |
| `source` | Source platform, such as `amazon_tr` |
| `review_summary` | Reserved field for possible platform-provided summaries |
| `collected_at` | Collection timestamp |

The model design follows a teacher-student methodology. A strong teacher model is used to generate controlled training targets, and a smaller student model is fine-tuned to reproduce this behavior more efficiently. GPT-5.4-mini was selected as the teacher model after comparing it with GPT-5.4 for this specific summarization task. Although GPT-5.4 is the larger model, the project observations showed that GPT-5.4-mini produced more suitable, concise, and controlled review summaries for the dataset generation workflow. It also fit the project’s need for large-scale target generation through structured JSON outputs.

For the student model, several base-model candidates were considered, including Qwen 2.5 7B, Llama 3.1 8B, and Mistral 7B. The final student model was `meta-llama/Llama-3.1-8B-Instruct`. It was chosen because it produced better Turkish sentence quality and more coherent task-following behavior during qualitative testing. In this project, Turkish fluency and the ability to follow strict formatting rules were more important than only selecting the smallest or fastest model. Llama 3.1 8B gave the best balance between output quality, trainability, and deployment feasibility.

The supervised fine-tuning dataset was built at product-aspect level. Each example contains an `input_text` and a `target_text`. The input includes the task instruction, product ID, product title, target aspect, aspect meaning, and selected evidence sentences. The target is a one-sentence teacher summary or `null` if evidence is weak, missing, or off-topic.

| Dataset Component | Description |
|---|---|
| `input_text` | Instruction, product information, target aspect, aspect meaning, and selected evidence |
| `target_text` | Teacher-generated Turkish summary sentence or `null` |

The data split was performed by `product_id`, not by individual rows. This is important because each product can generate multiple aspect examples. If rows were randomly split, evidence from the same product could appear in both training and validation/test sets, causing leakage. Product-level splitting prevents this issue.

| Split | Number of Examples | Ratio |
|---|---:|---:|
| Train | 6,150 | 80 percent |
| Validation | 765 | 10 percent |
| Test | 775 | 10 percent |

This dataset and model design supports the central goal of the project: given a product and aspect-specific evidence sentences, the student model should generate one short Turkish sentence that is faithful to the evidence, semantically limited to the target aspect, and able to return `null` when the evidence is insufficient.

## 3.2. Tools and Technology

The project combines web scraping, Turkish NLP preprocessing, transformer-based classification, semantic retrieval, teacher-model target generation, LoRA fine-tuning, model evaluation, and web deployment. The tools were not chosen only because they were available; most of them were selected after comparing alternatives and identifying which options worked best for Turkish e-commerce review data.

The main technologies used in the final system are summarized below.

| Category | Tool or Technology | Purpose |
|---|---|---|
| Web scraping | Selenium WebDriver | Collect product reviews from Amazon Turkey pages |
| Data processing | pandas, NumPy | Clean, transform, save, and inspect tabular datasets |
| Punctuation restoration | `uygarkurt/electra-restore-punctuation-turkish` | Restore punctuation in noisy Turkish reviews |
| Sentence splitting | Zemberek `TurkishSentenceExtractor` | Split cleaned Turkish reviews into sentence-level units |
| LLM annotation | Groq Llama 3.3 70B | Generate sentence-level aspect labels for classifier training |
| Aspect classifier | BERTurk, `dbmdz/bert-base-turkish-cased` | Multi-label aspect classification |
| Embedding model | `trmteb/turkish-embedding-model` | Encode sentences and aspect queries for semantic retrieval |
| Retrieval method | Cosine similarity, near-duplicate removal, MMR | Select relevant and diverse aspect evidence |
| Teacher model | GPT-5.4-mini | Generate controlled target summaries |
| Student model | Llama 3.1 8B Instruct | Fine-tuned aspect summarization model |
| Fine-tuning | LoRA, PEFT, TRL `SFTTrainer` | Parameter-efficient supervised fine-tuning |
| Quantization | BitsAndBytes 4-bit NF4 | Reduce GPU memory usage during training |
| Evaluation | ROUGE, BERTScore | Measure lexical and semantic similarity |
| Training environment | Kaggle T4x2 GPU | Train LoRA adapters under available GPU constraints |
| Student serving | RunPod, FastAPI, PyTorch, Transformers | Host the fine-tuned student model |
| Demo backend | FastAPI | Run the end-to-end pipeline and expose API endpoints |
| Demo frontend | React, HTML, CSS, Server-Sent Events | Show progress, summaries, and supporting evidence |

For scraping, Selenium was selected because Amazon review pages are dynamic and require browser-like interaction. The scraper visits the product page to collect metadata, opens review pages, waits for review cards to become stable, parses ratings and dates, and filters out foreign review cards where possible. Simpler static-request scraping would be more fragile for this use case because Amazon pages can rely on dynamic loading, changing DOM structures, login state, and pagination behavior. Selenium also made it possible to manually log in when required and continue scraping through the browser session.

For preprocessing, the project used an ELECTRA-based Turkish punctuation restoration model. This choice was necessary because many Turkish reviews are written with missing punctuation. Without punctuation restoration, sentence splitting becomes unreliable and long review fragments can contain several aspects together. The punctuation model is used before sentence splitting, and the implementation processes long texts in chunks to avoid token overflow. A confidence threshold of 0.55 is used so that low-confidence punctuation predictions are ignored. The pipeline also includes a comma-to-period heuristic for long comma-separated segments because users often connect independent claims with commas rather than full stops.

For sentence splitting, three approaches were evaluated: regex splitting, Stanza, and Zemberek. Regex splitting was rejected because it can incorrectly split abbreviations, decimal values, thousands separators, URLs, and noisy product-review text. For example, values such as `3.5`, `12.000`, or domains such as `amazon.com.tr` can cause incorrect boundaries. Stanza was also considered, but it was heavier and less stable for the project’s noisy review data and large-scale processing needs. Zemberek was selected because it is Turkish-specific, lightweight, and gave a better practical balance between speed and sentence-boundary quality.

For aspect labeling, the first attempted method was weak labeling using keywords and handcrafted patterns. This was rejected because it produced too many false positives and did not handle semantic overlap well. The biggest problem was the distinction between `kalite` and `genel_memnuniyet`. A keyword-based system could easily label broad satisfaction words incorrectly as quality, or miss sentences that expressed an aspect without using obvious keywords. As rules increased, the method became harder to maintain and less reliable. Therefore, the project moved to LLM-assisted labeling.

The final labeling dataset was generated using Groq-hosted Llama 3.3 70B. This model was used as an annotation assistant to label approximately 9,400 sentences with strict JSON output. The labeling setup allowed multiple labels per sentence and also allowed empty labels when no aspect was present. This approach was more flexible than weak labeling because the LLM could use sentence meaning rather than only surface keywords.

However, the LLM was not used as the final production labeler. Calling a large model for every sentence in a live pipeline would be slow and expensive. Instead, the LLM-generated labels were used to train a smaller BERTurk classifier. This is an important design decision: the project uses an expensive model offline to create training labels, then distills that labeling behavior into a faster local classifier for production.

Two candidate aspect classifiers were compared: an ELECTRA-based model and BERTurk. ELECTRA achieved a macro-F1 around 0.799 but showed weaker empty-label recall and tended to over-label sentences. BERTurk achieved a micro-F1 of 0.841 and macro-F1 of 0.797, and it behaved more reliably in the pipeline. Although ELECTRA was slightly better for the `paketleme` aspect, BERTurk performed better or more balanced for `kargo`, `kalite`, and `genel_memnuniyet`, and was selected as the production classifier.

| Label | ELECTRA F1 | BERTurk F1 | Selected Behavior |
|---|---:|---:|---|
| `kargo` | 0.685 | 0.738 | BERTurk was better |
| `paketleme` | 0.774 | 0.734 | ELECTRA was better |
| `fiyat_performans` | 0.838 | 0.840 | Similar |
| `kalite` | 0.866 | 0.895 | BERTurk was better |
| `genel_memnuniyet` | 0.746 | 0.777 | BERTurk was better |

The BERTurk model outputs sigmoid probabilities for each aspect. Different thresholds are used for each label because the classes have different distributions and different false-positive risks.

| Aspect | Threshold |
|---|---:|
| `kargo` | 0.85 |
| `paketleme` | 0.90 |
| `fiyat_performans` | 0.95 |
| `kalite` | 0.40 |
| `genel_memnuniyet` | 0.70 |

For semantic retrieval, the project uses `trmteb/turkish-embedding-model`. The goal is not to summarize every sentence classified under an aspect, but to select the most relevant and diverse evidence. The embedding model encodes both candidate sentences and short aspect queries. Cosine similarity is then used to measure relevance.

| Aspect | Semantic Query |
|---|---|
| `kargo` | `kargo hizi teslimat suresi paket geldi` |
| `paketleme` | `paket ambalaj kutu hasar koruma` |
| `fiyat_performans` | `fiyat performans deger para eder mi` |
| `kalite` | `urun kalitesi malzeme saglamlik dayaniklilik` |
| `genel_memnuniyet` | `genel memnuniyet tavsiye eder tekrar alir` |

The teacher model comparison also influenced the final methodology. GPT-5.4 and GPT-5.4-mini were both evaluated as possible teacher models. GPT-5.4-mini was selected because, for this specific task, it produced more suitable concise summaries and worked well with the strict JSON schema. The teacher prompt forces one short Turkish sentence per aspect, forbids unsupported frequency claims such as "cogu" or "birçok", prevents numerical claims, and requires `null` when evidence is weak. This strict prompting was necessary because the generated summaries become the targets for student fine-tuning.

For the student model, Qwen 2.5 7B, Llama 3.1 8B, and Mistral 7B were compared qualitatively. Llama 3.1 8B was selected because it produced more correct and natural Turkish sentences and followed the summarization instructions better. The project also investigated future larger-model options, including Gemma 4 31B Dense, but this was not adopted in the final methodology because of ecosystem and training issues such as PEFT compatibility problems, attention behavior with `use_cache=False`, and fp16 overflow concerns. Therefore, Llama 3.1 8B was the more reliable final choice for the capstone implementation.

Fine-tuning was performed with LoRA because full fine-tuning of an 8B model would require more GPU memory than the available environment could provide. The training was performed in a Kaggle T4x2 GPU environment using 4-bit NF4 quantization with double quantization. This allowed the project to train efficiently while keeping memory usage manageable.

| Hyperparameter | Value |
|---|---:|
| Epochs | 3 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Learning rate | 1e-4 |
| Scheduler | Cosine |
| Warmup steps | 50 |
| Weight decay | 0.01 |
| Max sequence length | 672 |
| Precision | fp16 |
| Optimizer | Paged AdamW 8-bit |
| Max gradient norm | 0.3 |
| Gradient checkpointing | Enabled |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Loss calculation | Completion-only loss |

Three training configurations were tested. V1 used message-format examples and applied LoRA only to attention modules. V3 changed the data format to prompt-completion format and used completion-only loss. V4 kept the completion format and expanded LoRA to MLP layers as well.

| Version | Format | LoRA Target Modules | Why It Was Tried |
|---|---|---|---|
| V1 | Messages | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Initial baseline using standard chat-style examples |
| V3 | Completion | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Tested whether training only on the answer improves learning |
| V4 | Completion | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Tested whether adapting MLP layers improves generation quality |

The switch from messages to completion format produced the largest improvement because the model stopped spending loss on the prompt and focused on learning the assistant answer. Expanding LoRA to MLP layers produced a smaller but consistent improvement. The final selected version was V4.

The evaluation used ROUGE and BERTScore. ROUGE was included because it measures lexical overlap with the teacher reference. BERTScore was included because semantically correct Turkish summaries may use different wording than the teacher output. The best model was V4, with BERTScore F1 of 0.8521.

| Model | Configuration | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore F1 |
|---|---|---:|---:|---:|---:|---:|
| Base | Untrained | 0.2944 | 0.1304 | 0.2549 | 0.2552 | 0.7541 |
| V1 | Messages + attention LoRA | 0.5520 | 0.3042 | 0.5116 | 0.5117 | 0.8405 |
| V3 | Completion + attention LoRA | 0.5777 | 0.3303 | 0.5371 | 0.5384 | 0.8495 |
| V4 | Completion + full LoRA | 0.5789 | 0.3387 | 0.5445 | 0.5442 | 0.8521 |

The deployment stack uses RunPod for serving the fine-tuned student model. A lightweight FastAPI generation server loads the model, applies the tokenizer chat template, generates the response, and returns it to the main demo application. The main demo backend is also built with FastAPI and exposes endpoints for health checking, running the pipeline, streaming progress, and returning results. The frontend is a React application that displays pipeline progress, teacher and student summaries, and supporting evidence.

## 3.3. Proposed Approach

The proposed approach is an end-to-end aspect-aware, evidence-grounded summarization pipeline. The main design principle is that summaries should not be generated directly from a large unfiltered set of reviews. Instead, the system first cleans and splits reviews, identifies aspect-related sentences, retrieves compact evidence, and then asks the model to summarize only that evidence. This reduces hallucination, avoids aspect mixing, and makes the output traceable to source reviews.

The complete pipeline consists of the following stages:

1. Review collection
2. Text cleaning and punctuation restoration
3. Sentence splitting
4. Aspect classification
5. Evidence retrieval and MMR selection
6. Teacher summary generation
7. SFT dataset construction
8. Student model fine-tuning
9. Evaluation
10. Evidence attribution
11. Demo deployment

In the review collection stage, the scraper accepts an Amazon ASIN or product URL. It extracts the ASIN, visits the product page to collect metadata, then navigates through review pages sorted by recency. It extracts local review cards and avoids foreign or translated reviews when possible. The live implementation targets up to 80 reviews per product by default. If the scraper is unavailable or too slow in the demo, a CSV upload option is provided so the rest of the pipeline can still run.

In the preprocessing stage, punctuation restoration is applied before cleaning and sentence splitting. This order was chosen because sentence splitting depends heavily on punctuation. The punctuation restoration model adds periods, commas, question marks, and other punctuation marks based on token-level predictions. The pipeline then normalizes Unicode, masks URLs and emails, removes invisible characters, normalizes whitespace, reduces repeated punctuation, preserves numeric dots, and fixes punctuation spacing.

After preprocessing, Zemberek is used to split each review into sentences. The resulting sentence table preserves metadata such as review ID, rating, product title, product URL, review date, source, and sentence index. This metadata is important because later evidence sentences must be linked back to original reviews for source attribution.

In the aspect classification stage, each cleaned sentence is passed to the BERTurk multi-label classifier. The output includes predicted labels, one binary prediction column per aspect, and one probability column per aspect. Because the system is multi-label, a sentence can be assigned to more than one aspect. This is necessary for realistic review text, where users often discuss delivery, packaging, price, and quality in the same sentence or adjacent clauses.

In the evidence retrieval stage, the pipeline first removes noisy sentences. Sentences with fewer than four tokens or sentences containing only punctuation are removed. Then, for each product and aspect, the system keeps only sentences predicted as belonging to that aspect. These candidate sentences are embedded and compared with the corresponding aspect query using cosine similarity. Sentences with relevance lower than 0.1 are removed.

Next, near-duplicate filtering is applied. E-commerce reviews often repeat the same simple statements, especially for shipping and packaging. If two candidate sentences have embedding similarity of 0.85 or higher, the lower-relevance sentence is removed. This step prevents the evidence set from being filled with repeated versions of the same opinion.

Finally, Maximal Marginal Relevance is used to select the final evidence sentences. MMR was chosen because selecting only the top relevance scores can create redundant evidence. MMR balances relevance and diversity. The pipeline takes the top 100 relevance-ranked candidates and selects up to 30 final sentences with lambda 0.7. This keeps the selected evidence close to the target aspect while still covering different user claims.

| Evidence Selection Parameter | Value |
|---|---:|
| Top candidate pool | 100 |
| Final MMR sentences | 30 |
| MMR lambda | 0.7 |
| Duplicate threshold | 0.85 |
| Minimum tokens | 4 |
| Minimum relevance | 0.1 |

The MMR output contains product ID, product title, aspect, selected sentence, relevance score, MMR rank, rating, review ID, sentence index, and source. This table is the bridge between raw reviews and model summarization.

For teacher generation, selected evidence is organized in a flattened product-aspect format. Each teacher request focuses on a single product and a single target aspect, using the same task structure later used for student fine-tuning. The prompt contains the product ID, product title, target aspect, aspect definition, and only the evidence sentences retrieved for that aspect. The teacher returns either one short Turkish sentence or `null`. The teacher prompt was designed with strict constraints because teacher outputs become training targets. It forbids unsupported numerical claims, frequency words, copying review text directly, and artificial balancing of positive and negative evidence. If evidence is weak or irrelevant, the teacher must return `null`.

The student dataset is created directly from these flattened teacher outputs. Therefore, the teacher-generation format and the student-training format are aligned: one product, one aspect, one evidence block, and one target answer. This makes the learning task narrower and helps prevent aspect leakage. Each student input contains one target aspect, its definition, and only the evidence for that aspect.

The student model is then fine-tuned using supervised fine-tuning. The final model uses completion-format training with completion-only loss. This was chosen after comparing V1, V3, and V4. V1 showed that basic LoRA training improved the base model, but V3 showed that completion-only training was more suitable because the loss focused only on the expected output. V4 further improved the model by applying LoRA to MLP modules in addition to attention modules.

After training, the model is evaluated on the held-out validation set. The base model, V1, V3, and V4 are compared using ROUGE and BERTScore. The results show that fine-tuning substantially improves over the base model. The biggest improvement comes from switching to completion-format training, and the best overall result comes from V4.

The pipeline also includes source attribution after summary generation. For each generated student summary, the system embeds the summary and compares it with MMR-selected evidence sentences from the same product and aspect. It returns the top matching evidence sentences with similarity score, evidence rank, MMR rank, relevance score, rating, review ID, and full review text when available. This stage makes the system explainable because a user can inspect which review sentences support each summary.

The final demo integrates the complete methodology into a usable application. The FastAPI backend loads the preprocessing model, aspect classifier, embedding model, prompt builder, teacher/student summarizer, evidence retriever, and scraper. The frontend allows the user to enter an ASIN, upload a CSV, optionally add a comment, watch the pipeline progress, and compare student and teacher summaries with source evidence.

| Demo Stage | Output |
|---|---|
| Scraping or CSV loading | Raw review table |
| Preprocessing | Cleaned reviews and sentence table |
| Aspect labeling | Aspect probabilities and binary predictions |
| Evidence selection | MMR-ranked aspect evidence |
| Summarization | Student and teacher aspect summaries |
| Evidence retrieval | Source-attributed supporting review sentences |

This approach addresses the main difficulties of Turkish e-commerce review summarization. Punctuation restoration and Zemberek sentence splitting handle noisy user text. Multi-label aspect classification handles sentences that contain multiple product claims. Embedding retrieval and MMR reduce irrelevant and repetitive evidence. Teacher-generated targets make it possible to train a smaller student model without manually writing thousands of summaries. LoRA fine-tuning makes training feasible under limited GPU resources. Finally, source attribution makes the output more transparent by linking generated summaries back to review evidence.

The project also identified remaining limitations. The quality of the final summary depends on upstream stages: scraping quality, sentence splitting, aspect labeling, and evidence selection. If weak evidence enters the prompt, the model may produce weaker summaries. The project notes especially identified low-relevance `kalite` evidence as a possible issue. A planned improvement is a dynamic cascading relevance threshold: first require high relevance, such as 0.25, then relax the threshold only if too few evidence sentences remain. This would reduce noise while preserving enough examples for training and summarization.

Another limitation is that the reference summaries are teacher-generated rather than human-written. This is suitable for knowledge distillation, but future work should include human evaluation of factuality, usefulness, fluency, aspect purity, and evidence faithfulness. The project also considered moving from T4x2 training to A100-80GB multi-GPU training with bf16 and no quantization, which could reduce quantization noise and allow a higher LoRA rank. However, the final capstone implementation uses the stable Llama 3.1 8B LoRA pipeline because it is reproducible and already performs strongly on the evaluation set.
