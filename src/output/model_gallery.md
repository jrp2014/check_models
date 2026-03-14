# Model Output Gallery

_Generated on 2026-03-14 22:43:52 GMT_

A review-friendly artifact with image metadata, the source prompt, and full generated output for each model.

## 🎯 Action Snapshot

- **Framework/runtime failures:** none.
- **Maintainer signals:** harness-risk successes=6, clean outputs=4/49.
- **Useful now:** 3 clean A/B model(s) worth first review.
- **Review watchlist:** 46 model(s) with breaking or lower-value output.
- **Vs existing metadata:** better=22, neutral=2, worse=25 (baseline B 72/100).
- **Quality signal frequency:** missing_sections=28, context_ignored=19, description_length=12, keyword_count=11, context_echo=10, reasoning_leak=10.
- **Runtime pattern:** decode dominates measured phase time (90%; 49/49 measured model(s)).
- **Phase totals:** model load=171.39s, prompt prep=0.13s, decode=1613.59s, cleanup=5.81s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49.

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (90/100) | Δ+18 | 50.1 tps
- `mlx-community/X-Reasoner-7B-8bit`: 🏆 A (90/100) | Δ+18 | 47.0 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (88/100) | Δ+16 | 47.7 tps

### Watchlist

- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) | Δ-72 | harness
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (0/100) | Δ-72 | harness
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) | Δ-71 | 319.8 tps | context ignored, degeneration, harness, missing sections
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) | Δ-64 | 19.2 tps | harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (11/100) | Δ-60 | 27.8 tps | context ignored, harness, missing sections

## Image Metadata

- **Description**: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK
Here is a caption for the image:

A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn Garden City, England. Locals and visitors take advantage of the clear skies, gathering at the outdoor tables of a popular corner bakery. The scene is set against the town's signature neo-Georgian red-brick architecture, a hallmark of its design as one of Britain's first "garden cities." In the foreground, a delivery driver on a scooter pauses, a distinctly modern sight contrasting with the timeless feel of the planned town centre.
- **Keywords**: Adobe Stock, Any Vision, Corner Building, England, English Town Centre, English town, Europe, Food Delivery, Food Delivery Scooter, GAIL's Bakery, Hertfordshire, Howardsgate, Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills, Sunny Spring Afternoon, Town centre, UK, United Kingdom, Welwyn Garden City, afternoon, architecture, bakery, balcony, blue, blue sky, brick building, brown, building, cafe, city life, courier, customers, daylight, delivery driver, dormer window, facade, gig economy, leisure, neo-georgian, outdoors, pavement, people, red brick building, scooter, shopfront, street, street scene, sunlight, sunny day, tiled roof, town, traditional, travel, urban, urban life, white, windows
- **Date**: 2026-03-14 15:19:38 GMT
- **Time**: 15:19:38

## Prompt

<!-- markdownlint-disable MD028 MD049 -->
>
> Analyze this image for cataloguing metadata.
>
> Use only details that are clearly and definitely visible in the image. If a detail is
> uncertain, ambiguous, partially obscured, too small to verify, or not directly visible,
> leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details that are
> clearly confirmed by the image, correct anything contradicted by the image, and add
> important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting, lighting, action,
> and other distinctive visible details. Omit anything uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects, setting,
> colors, composition, and style. Omit uncertain tags rather than guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords fields.
> \- Do not infer identity, location, event, brand, species, time period, or intent unless
> visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually confirmed):
> \- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK Here
> is a caption for the image: A bright March afternoon brings a touch of warmth to
> Howardsgate in Welwyn Garden City, England. Locals and visitors take advantage of the
> clear skies, gathering at the outdoor tables of a popular corner bakery. The scene is
> set against the town's signature neo-Georgian red-brick architecture, a hallmark of its
> design a...
> \- Keyword hints: Adobe Stock, Any Vision, Corner Building, England, English Town Centre,
> English town, Europe, Food Delivery, Food Delivery Scooter, GAIL's Bakery,
> Hertfordshire, Howardsgate, Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe
> Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
> \- Capture metadata: Taken on 2026-03-14 15:19:38 GMT (at 15:19:38 local time).
<!-- markdownlint-enable MD028 MD049 -->

## Quick Navigation

- **Best cataloging quality:** [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)
- **Fastest generation:** [`prince-canuma/Florence-2-large-ft`](#model-prince-canuma-florence-2-large-ft)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- **Best balance:** [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
- **D/F utility models:** `HuggingFaceTB/SmolVLM-Instruct`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, +11 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD013 MD033 -->

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** Load 0.68s | Gen 1.02s | Total 2.03s
**Throughput:** Prompt 2,584 TPS (582 tok) | Gen 311 TPS (89 tok)
**Assessment:** ❌ F (25/100) | Δ-46 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, missing sections)
**Review:** F 25/100 | Missing sections (keywords); Description sentence violation (3; expected 1-2); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: "Bright March Afternoon in Howardsgate, Welwyn Garden City, England"
> Description: "A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn
> Garden City, England. Locals and visitors take advantage of the clear skies, gathering
> at the outdoor tables of a popular corner bakery. The scene is set against the town's
> signature neo-Georgian architecture, a hallmark of its design a."
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (keywords)
- Description sentence violation (3; expected 1-2)
- Context echo (98% overlap)

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** Load 0.70s | Gen 1.44s | Total 2.47s
**Throughput:** Prompt 2,863 TPS (846 tok) | Gen 273 TPS (164 tok)
**Assessment:** ❌ F (29/100) | Δ-42 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo)
**Review:** F 29/100 | Description sentence violation (3; expected 1-2); Keyword count violation (20; expected 10-18); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Howardsgate, Welwyn Garden City, England, United Kingdom, England
> Description: A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn
> Garden City, England. Locals and visitors take advantage of the clear skies, gathering
> at the outdoor tables of a popular corner bakery. The scene is set against the town's
> signature neo-Georgian red-brick architecture, a hallmark of its design.
> Keywords: Adobe Stock, Any Vision, Corner Building, England, English Town Centre,
> English town, Europe, Food Delivery, Food Delivery Scooter, GAIL's Bakery,
> Hertfordshire, Howardsgate, Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe
> Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Context echo (97% overlap)

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

**Metrics:** Load 0.76s | Gen 1.54s | Total 2.62s
**Throughput:** Prompt 2,550 TPS (582 tok) | Gen 99.9 TPS (80 tok)
**Assessment:** ❌ F (27/100) | Δ-45 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, missing sections)
**Review:** F 27/100 | Missing sections (keywords); Description sentence violation (3; expected 1-2); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: A Bright March Day in Howardsgate, Welwyn Garden City, England
> Description: A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn
> Garden City, England. Locals and visitors gather at the outdoor tables of a popular
> corner bakery. The scene is set against the town's signature neo-Georgian architecture,
> a hallmark of its design a.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (keywords)
- Description sentence violation (3; expected 1-2)
- Context echo (93% overlap)

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Metrics:** Load 0.85s | Gen 1.77s | Total 2.97s
**Throughput:** Prompt 2,856 TPS (846 tok) | Gen 170 TPS (166 tok)
**Assessment:** ❌ F (29/100) | Δ-42 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo)
**Review:** F 29/100 | Description sentence violation (3; expected 1-2); Keyword count violation (20; expected 10-18); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Howardsgate, Welwyn Garden City, England, United Kingdom, UK
>
> Description:
> A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn Garden City,
> England. Locals and visitors take advantage of the clear skies, gathering at the outdoor
> tables of a popular corner bakery. The scene is set against the town's signature
> neo-Georgian red-brick architecture, a hallmark of its design.
>
> Keywords:
> Adobe Stock, Any Vision, Corner Building, England, English Town Centre, English town,
> Europe, Food Delivery, Food Delivery Scooter, GAIL's Bakery, Hertfordshire, Howardsgate,
> Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining,
> Sidewalk, Socializing, Spring, Stonehills
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Context echo (97% overlap)

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** Load 1.02s | Gen 1.88s | Total 3.26s
**Throughput:** Prompt 1,400 TPS (1,796 tok) | Gen 116 TPS (18 tok)
**Assessment:** ❌ F (15/100) | Δ-56 | Output lacks detail
**Review Status:** watchlist (context ignored, missing sections)
**Review:** F 15/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> 5-10 words, concrete and factual, limited to clearly visible content.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** Load 1.07s | Gen 1.94s | Total 3.34s
**Throughput:** Prompt 1,388 TPS (1,796 tok) | Gen 114 TPS (18 tok)
**Assessment:** ❌ F (15/100) | Δ-56 | Output lacks detail
**Review Status:** watchlist (context ignored, missing sections)
**Review:** F 15/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> 5-10 words, concrete and factual, limited to clearly visible content.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

**Metrics:** Load 0.81s | Gen 1.95s | Total 3.11s
**Throughput:** Prompt 3,540 TPS (586 tok) | Gen 306 TPS (232 tok)
**Assessment:** ❌ F (26/100) | Δ-45 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, missing sections)
**Review:** F 26/100 | Missing sections (keywords); Title length violation (81 words; expected 5-10); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: "A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn
> Garden City, England, England, United Kingdom, UK Here is a caption for the image: A
> bright March afternoon brings a touch of warmth to Howardsgate in Welwyn Garden City,
> England. Locals and visitors take advantage of the clear skies, gathering at the outdoor
> tables of a popular corner bakery. The scene is set against the town's signature
> neo-Georgian red-brick architecture, a hallmark of its design a..."
>
> Description: The image depicts a corner bakery on Howardsgate in Welwyn Garden City,
> England. The bakery is a popular spot for locals and visitors, with tables set up
> outside for outdoor dining. The architecture is characteristic of the town's
> neo-Georgian red-brick buildings, a hallmark of its design. The scene is set against a
> backdrop of clear skies, and the atmosphere is warm and inviting. The image is taken on
> March 14, 2026, at 15:19:38 GMT (at 15:19:38 local time).
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (keywords)
- Title length violation (81 words; expected 5-10)
- Description sentence violation (5; expected 1-2)
- Context echo (40% overlap)

---

<a id="model-prince-canuma-florence-2-large-ft"></a>

### ✅ prince-canuma/Florence-2-large-ft

**Metrics:** Load 1.21s | Gen 2.22s | Total 3.75s
**Throughput:** Prompt 4,735 TPS (1,163 tok) | Gen 320 TPS (500 tok)
**Assessment:** ❌ F (0/100) | Δ-71 | Output too short to be useful
**Review Status:** watchlist (context ignored, degeneration, harness, missing sections)
**Review:** F 0/100 | ⚠️HARNESS:stop_token; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- ⚠️HARNESS:stop_token
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Output degeneration (character_loop: &#x27;&lt;s&gt;&#x27; repeated)
- Missing sections (title, description, keywords)
- Unknown tags: &lt;s&gt;
- tokenizer_artifact
- token_leak:&lt;s&gt;

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** Load 1.01s | Gen 2.33s | Total 3.66s
**Throughput:** Prompt 1,364 TPS (1,696 tok) | Gen 113 TPS (73 tok)
**Assessment:** 🟡 C (59/100) | Δ-12 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** C 59/100 | Keyword count violation (6; expected 10-18)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Hertfordshire, England, UK - Howardsgate, Corner Bakery, Neo-Georgian
> Architecture
> Description: A clear day in Howardsgate, Hertfordshire, England, UK, with a popular
> corner bakery.
> Keywords: Hertfordshire, England, UK, Howardsgate, Corner Bakery, Neo-Georgian
> Architecture
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Keyword count violation (6; expected 10-18)

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** Load 3.52s | Gen 4.56s | Total 8.43s
**Throughput:** Prompt 440 TPS (1,603 tok) | Gen 27.8 TPS (14 tok)
**Assessment:** ❌ F (11/100) | Δ-60 | Output lacks detail
**Review Status:** watchlist (context ignored, harness, missing sections)
**Review:** F 11/100 | ⚠️HARNESS:prompt_template; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \- Use only the metadata that is clearly supported by the image.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- ⚠️HARNESS:prompt_template
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)
- output:output_ratio(0.9%)

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** Load 1.40s | Gen 4.97s | Total 6.74s
**Throughput:** Prompt 834 TPS (3,193 tok) | Gen 158 TPS (115 tok)
**Assessment:** ✅ B (68/100) | Δ-4 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** B 68/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Bright March Day at GAIL’s Bakery Corner
>
> Description:
> A sunny outdoor scene at a brick building corner in a small English town, featuring a
> bakery with outdoor seating. People are gathered at tables, and a scooter is parked near
> the entrance under clear blue skies.
>
> Keywords:
> red-brick building, outdoor dining, GAIL's Bakery, Welwyn Garden City, England, scooter,
> sidewalk café, neo-Georgian architecture, outdoor seating, social gathering, spring
> day, clear skies, brick facade, small-town England
<!-- markdownlint-enable MD028 MD049 -->


**Quality Status:** no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

**Metrics:** Load 3.06s | Gen 5.55s | Total 8.96s
**Throughput:** Prompt 1,389 TPS (836 tok) | Gen 110 TPS (500 tok)
**Assessment:** ❌ F (4/100) | Δ-67 | Output too short to be useful
**Review Status:** watchlist (context echo, context ignored, missing sections)
**Review:** F 4/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
>
>
> \-
> 15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15:19:38:15
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)
- Context echo (100% overlap)

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

**Metrics:** Load 2.87s | Gen 6.27s | Total 9.63s
**Throughput:** Prompt 770 TPS (2,365 tok) | Gen 30.3 TPS (79 tok)
**Assessment:** ✅ B (66/100) | Δ-6 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** B 66/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Corner building with GAIL's Bakery
> Description: A red-brick corner building houses GAIL's Bakery with outdoor seating,
> under clear blue skies. A food delivery scooter is parked on the sidewalk.
> Keywords: corner building, red-brick, GAIL's Bakery, outdoor seating, sidewalk, food
> delivery, scooter, clear skies, outdoor cafe, neo-Georgian architecture
<!-- markdownlint-enable MD028 MD049 -->


**Quality Status:** no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Metrics:** Load 3.75s | Gen 7.56s | Total 11.85s
**Throughput:** Prompt 930 TPS (1,580 tok) | Gen 96.1 TPS (500 tok)
**Assessment:** 🏆 A (90/100) | Δ+18 | None identified
**Review Status:** watchlist (missing sections, reasoning leak)
**Review:** A 90/100 | Missing sections (title); Reasoning leak (◁think▷, ◁/think▷)

<!-- markdownlint-disable MD028 MD049 -->
>
> ◁think▷Okay, let's analyze the image for cataloging metadata based on the given rules.
> First, I need to focus only on what's clearly visible and avoid any assumptions or
> guesses. The image shows a brick building with a bakery or cafe on the corner, people
> sitting at outdoor tables, a scooter parked nearby, and a bright sunny day. The building
> has neo-Georgian architecture, which matches the hint about Howardsgate in Welwyn
> Garden City, England. The sky is very blue, indicating it's a clear day. The keyword
> "GAIL's Bakery" is visible on the awning, confirming the bakery's name. The presence of
> outdoor seating and socializing people supports the keywords like "Outdoor Cafe Culture"
> and "Socializing." The scooter is parked on the sidewalk, which is part of the street
> scene. The time mentioned in the metadata (March afternoon) aligns with the bright
> sunlight and shadows visible. I should exclude any uncertain details like exact time or
> specific brands beyond what's visible.◁/think▷Title: Gail's Bakery Corner Cafe  
> Description: A sunny March afternoon in Howardsgate, Welwyn Garden City, England,
> features a neo-Georgian red-brick building housing Gail's Bakery with outdoor seating
> where people socialize. A scooter is parked nearby under a clear blue sky.  
> Keywords: Gail's Bakery, England, Neo-Georgian Architecture, Outdoor Cafe, Socializing,
> Spring, Street Scene, Scooter, Welwyn Garden City<|im_assistant|><|im_user|> f = 2i
> Im{f} = 0, f z = f Re{f} + i Im{f}, f z = f Re{f} - i Im{f}, f z = f Re{f} + Im{f}, f z
> = f Re{f} - Im{f}<|im_assistant|><|im_user|>{2} (z - z^3) z^2 - 2
> z^4.<|im_assistant|><|im_user|> c= \frac{1}{2} \ln (z + \sqrt{2}) e^{-2} z^2 -
> \frac{1}{2} \ln (-z + \sqrt{2}) e^{-2} z^2 + \frac{1}{2} \ln (-z - \sqrt{2}) e^{-2} z^2
> \+ C.<|im_assistant|><|im_user|> c] = 2 E.<|im_assistant|><|im_user|>◁think▷Okay, let's
> analyze the given expression for the function f using the provided rules. The expression
> is f =
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (title)
- Reasoning leak (◁think▷, ◁/think▷)

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Metrics:** Load 3.37s | Gen 7.74s | Total 11.47s
**Throughput:** Prompt 650 TPS (1,580 tok) | Gen 109 TPS (500 tok)
**Assessment:** 🏆 A (89/100) | Δ+17 | None identified
**Review Status:** watchlist (missing sections, reasoning leak)
**Review:** A 89/100 | Missing sections (title); Description sentence violation (3; expected 1-2); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> ◁think▷Got it, let's analyze the image step by step.
>
> First, the title: The image shows a bright sunny day with a brick building, people at
> outdoor cafe tables, a scooter, and a clear blue sky. The building has a bakery sign
> (GAIL's Bakery), so the title should be concise. Let's check the visible elements:
> "Bright sunny day, brick building, outdoor cafe, GAIL's Bakery, scooter, clear sky."
>
> Description: The description needs to be 1-2 sentences. The main subject is a sunny day
> in a town with a brick building (GAIL's Bakery), outdoor cafe seating, people
> socializing, a scooter, and the neo-Georgian architecture. So: "A bright sunny day in a
> town centre with a brick building (GAIL's Bakery) featuring outdoor cafe seating, people
> socializing, a scooter, and clear blue sky."
>
> Keywords: Now list 10-18 unique terms. Visible elements: Adobe Stock (watermark), Any
> Vision (watermark), Corner Building, England, English Town Centre, English town, Europe,
> Food Delivery, Food Delivery Scooter, GAIL's Bakery, Hertfordshire, Howardsgate,
> Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk,
> Socializing. Let's count: Adobe Stock, Any Vision, Corner Building, England, English
> Town Centre, English town, Europe, Food Delivery, Food Delivery Scooter, GAIL's Bakery,
> Hertfordshire, Howardsgate, Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe
> Culture, Outdoor Dining, Sidewalk, Socializing. That's 17 keywords.
>
> Wait, check if all are visible. The watermark "Adobe Stock" and "Any Vision" are on the
> image. The building is a corner building, in England, English town centre, Hertfordshire
> (a county in England), Howardsgate (a street name), neo-Georgian architecture. The
> scene has outdoor cafe (GAIL's Bakery), food delivery scooter, people socializing,
> outdoor dining, sidewalk, etc. So all these are visible. Let's list them.◁/think▷Title:
> Bright sunny day, brick building, GAIL's Bakery, outdoor cafe, scooter, clear sky
>
> Description:
> A bright sunny day in a town centre with a brick building (GAIL's Bakery) featuring
> outdoor cafe seating, people socializing, a scooter, and clear blue sky.
>
> Keywords
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (title)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (53; expected 10-18)
- Keyword duplication (40% duplicated terms)
- Reasoning leak (◁think▷, ◁/think▷)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** Load 4.90s | Gen 8.89s | Total 14.13s
**Throughput:** Prompt 186 TPS (845 tok) | Gen 23.9 TPS (93 tok)
**Assessment:** 🏆 A (89/100) | Δ+18 | None identified
**Review Status:** watchlist (context ignored)
**Review:** A 89/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Red brick building with bakery and outdoor seating
>
> Description:
> A three-story red brick building houses a bakery with outdoor tables and chairs on a
> sidewalk. A scooter is parked in front of the building under a clear blue sky.
>
> Keywords:
> Red brick, building, bakery, outdoor seating, sidewalk, scooter, storefront, windows,
> trees, clear sky, architecture, urban, street, corner building, storefront, daylight,
> England, stone
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** Load 2.86s | Gen 10.05s | Total 13.30s
**Throughput:** Prompt 482 TPS (3,387 tok) | Gen 35.2 TPS (91 tok)
**Assessment:** ✅ B (69/100) | Δ-3 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** B 69/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> \- Red Brick Building with Outdoor Cafe
>
> Description:
> \- A bright, sunny day shows a red brick building with a cafe on the ground floor,
> outdoor seating, and a food delivery scooter parked nearby.
>
> Keywords:
> \- Red Brick, Building, Cafe, Outdoor Seating, Food Delivery Scooter, Sunny Day, Clear
> Sky, People, Street, Urban, Architecture, Corner Building, Outdoor Dining, Sidewalk,
> Europe, England
<!-- markdownlint-enable MD028 MD049 -->


**Quality Status:** no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** Load 6.93s | Gen 10.16s | Total 17.42s
**Throughput:** Prompt 368 TPS (1,603 tok) | Gen 4.65 TPS (25 tok)
**Assessment:** 🟠 D (40/100) | Δ-32 | Lacks visual description of image
**Review Status:** watchlist (context ignored, missing sections)
**Review:** D 40/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> \- Use only the metadata that is clearly visible in the image.
> \- Do not guess, infer, or speculate.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Metrics:** Load 6.11s | Gen 10.41s | Total 17.30s
**Throughput:** Prompt 924 TPS (1,580 tok) | Gen 64.3 TPS (500 tok)
**Assessment:** 🏆 A (91/100) | Δ+20 | None identified
**Review Status:** watchlist (missing sections, reasoning leak)
**Review:** A 91/100 | Missing sections (title); Description sentence violation (5; expected 1-2); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> ◁think▷Got it, let's analyze the image step by step.
>
> First, the title: The image shows a bright sunny day with a red-brick building, a bakery
> with outdoor seating, people, and a scooter. The location is Howardsgate in Welwyn
> Garden City, England. So the title should be concise. Let's check the existing hints: "A
> bright March afternoon..." but the image shows a clear blue sky, so maybe "Bright March
> afternoon at Howardsgate, Welwyn Garden City" but need to be factual. Wait, the
> existing caption mentions "A bright March afternoon..." but the image's sky is clear, so
> maybe "Bright sunny day at Howardsgate, Welwyn Garden City" but need to check. Wait,
> the existing metadata hint says "A bright March afternoon..." but the image's sky is
> clear, so maybe the season is spring (since March is spring). But the image shows people
> in light clothing, so maybe "Bright spring day at Howardsgate, Welwyn Garden City". But
> the existing hints have "Spring" as a keyword. Let's check the image: the trees have no
> leaves, so maybe early spring. But the existing hints say "Spring" as a keyword. So the
> title should be based on visible content. The main elements: red-brick building,
> bakery, outdoor seating, people, scooter, clear sky. So title: "Bright sunny day at
> Howardsgate, Welwyn Garden City".
>
> Description: The description should be 1-2 sentences. The main subject is a red-brick
> building (GAIL's Bakery) with outdoor seating, people socializing, a scooter, clear blue
> sky. So "A bright sunny day in Howardsgate, Welwyn Garden City, featuring a red-brick
> building with outdoor cafe seating, people socializing, and a scooter parked on the
> sidewalk under a clear blue sky." Wait, but the existing hints mention "Outdoor Cafe
> Culture" and "Socializing". Let's check the image: there are red chairs and tables
> outside the bakery, people gathered, so "Outdoor cafe culture" is visible. So
> description: "A sunny day in Howardsgate, Welwyn Garden City, showcasing a red-brick
> building with outdoor cafe seating, people socializing, and a scooter parked on the
> sidewalk under a clear blue sky."
>
> Keywords: Need 10-18 unique terms. From the image: Howardsgate, Welwyn Garden City,
> England, UK,
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (title)
- Description sentence violation (5; expected 1-2)
- Keyword count violation (4; expected 10-18)
- Reasoning leak (◁think▷, let&#x27;s analyze the image)

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** Load 4.55s | Gen 10.85s | Total 15.79s
**Throughput:** Prompt 884 TPS (844 tok) | Gen 41.8 TPS (396 tok)
**Assessment:** 🏆 A (94/100) | Δ+22 | None identified
**Review Status:** watchlist (missing sections)
**Review:** A 94/100 | Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> \## Image Description
>
> The image captures a vibrant street scene on a sunny day in Howardsgate, Welwyn Garden
> City, England. The focus is on a row of red-brick buildings housing various businesses,
> with people enjoying outdoor seating at a bakery called "GAIL's Bakery."
>
> \\*\\*Key elements in the image:\\*\\*
>
> \* \\*\\*Architecture:\\*\\* The buildings are two to three stories tall, constructed from red
> brick with white window frames and some decorative elements. The architectural style
> appears to be neo-Georgian.
> \* \\*\\*GAIL's Bakery:\\*\\* A prominent feature is the "GAIL's Bakery" with a red awning and
> outdoor seating consisting of red tables and chairs. Several people are seated at these
> tables, enjoying food and drinks.
> \* \\*\\*People:\\*\\* Numerous people are visible, some seated at the bakery's outdoor tables,
> others walking along the sidewalk. They appear to be casually dressed and enjoying the
> pleasant weather.
> \* \\*\\*Scooter:\\*\\* A black scooter is parked on the sidewalk in front of one of the
> buildings.
> \* \\*\\*Streetscape:\\*\\* The scene includes a paved sidewalk, a street with a curb, and some
> greenery in the form of small trees planted along the sidewalk.
> \* \\*\\*Weather:\\*\\* The sky is clear and blue, indicating a bright, sunny day. Shadows cast
> by the buildings and people suggest the sun is shining.
>
> \\*\\*Overall Impression:\\*\\*
>
> The image conveys a lively and pleasant atmosphere of a typical town street on a nice
> day. It highlights the combination of commercial activity and social interaction in a
> well-maintained urban environment. The presence of the bakery and outdoor seating
> suggests a popular spot for locals and visitors alike.
>
> \## Keywords:
>
> Howardsgate, Welwyn Garden City, England, UK, street, bakery, GAIL's Bakery, outdoor
> seating, people, sidewalk, red brick, neo-Georgian architecture, sunny day, socializing,
> urban scene, town, cafe, food, drink.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** Load 1.46s | Gen 11.07s | Total 12.87s
**Throughput:** Prompt 1,297 TPS (1,416 tok) | Gen 52.6 TPS (500 tok)
**Assessment:** ❌ F (33/100) | Δ-38 | Mostly echoes context without adding value
**Review Status:** watchlist (repetitive)
**Review:** F 33/100 | Repetitive output (phrase: "stonehills, stonehills, stoneh..."); Title length violation (4 words; expected 5-10); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Welwyn Garden City, England
>
> Description: A clear day in Howardsgate, a town in Welwyn Garden City, England, with
> people enjoying outdoor dining at a bakery. The scene is set against a red-brick
> neo-Georgian building.
>
> Keywords: Welwyn Garden City, England, Howardsgate, neo-Georgian architecture, outdoor
> dining, red-brick building, clear skies, socializing, outdoor cafe, bakery,
> Hertfordshire, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Repetitive output (phrase: &quot;stonehills, stonehills, stoneh...&quot;)
- Title length violation (4 words; expected 5-10)
- Keyword count violation (106; expected 10-18)
- Keyword duplication (89% duplicated terms)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** Load 6.80s | Gen 11.73s | Total 18.87s
**Throughput:** Prompt 186 TPS (845 tok) | Gen 14.2 TPS (96 tok)
**Assessment:** 🏆 A (88/100) | Δ+17 | None identified
**Review Status:** watchlist (context ignored)
**Review:** A 88/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe Culture, Socializing, Spring, Stonehills
Capture, GMT)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Red brick building with bakery and scooter
>
> Description:
> A three-story red brick building houses a bakery with outdoor tables and chairs on a
> sidewalk. A scooter is parked in front of the building, and people are seated at the
> tables on a sunny day.
>
> Keywords:
> Architecture, bakery, brick, building, cafe, England, outdoor cafe, outdoor dining, red
> brick, scooter, sidewalk, storefront, street, town, trees, urban, windows
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe Culture, Socializing, Spring, Stonehills
Capture, GMT)

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** Load 4.56s | Gen 11.94s | Total 16.83s
**Throughput:** Prompt 562 TPS (3,387 tok) | Gen 17.8 TPS (98 tok)
**Assessment:** 🏆 A (89/100) | Δ+18 | None identified
**Review Status:** watchlist (context ignored)
**Review:** A 89/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> \- Red Brick Building with Outdoor Cafe
>
> Description:
> \- A bright, sunny day shows a red brick building with a cafe on the ground floor,
> outdoor seating, and a food delivery scooter parked nearby.
>
> Keywords:
> \- Red Brick, Building, Cafe, Outdoor Seating, Food Delivery Scooter, Sunny Day, Clear
> Sky, People, Street, Urban, Architecture, Windows, Balcony, Potted Plants, Signage,
> Sidewalk, Tables, Chairs
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** Load 2.97s | Gen 12.36s | Total 15.89s
**Throughput:** Prompt 339 TPS (2,365 tok) | Gen 26.7 TPS (128 tok)
**Assessment:** ✅ B (69/100) | Δ-2 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** B 69/100 | Title length violation (4 words; expected 5-10); Description sentence violation (3; expected 1-2)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:  
> Bakery Corner with Scooter
>
> Description:  
> The image shows a corner bakery with red awnings and outdoor seating, located on a sunny
> street. A delivery scooter is parked in front, and people are visible sitting at the
> tables. The building is made of red brick with white-framed windows and a small balcony.
>
> Keywords:  
> Bakery, Corner Building, Delivery Scooter, English Town, Outdoor Cafe, Red Brick,
> Sidewalk, Street Scene, Sunlit, White Windows, Outdoor Dining, Neo-Georgian
> Architecture, Hertfordshire, Howardsgate, Welwyn Garden City
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Title length violation (4 words; expected 5-10)
- Description sentence violation (3; expected 1-2)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

**Metrics:** Load 2.21s | Gen 12.91s | Total 15.46s
**Throughput:** Prompt 324 TPS (3,194 tok) | Gen 50.1 TPS (132 tok)
**Assessment:** 🏆 A (90/100) | Δ+18 | None identified
**Review Status:** strong candidate for first-pass review
**Review:** A 90/100 | Description sentence violation (3; expected 1-2)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Bakery Corner with Outdoor Dining and Scooter
>
> Description:
> The image shows a red-brick corner building with a bakery on the ground floor, featuring
> outdoor seating with red chairs and tables. A scooter is parked on the sidewalk in
> front of the establishment, and several people are seated outside, enjoying the sunny
> day. The architecture includes white-framed windows and a small balcony.
>
> Keywords:
> Bakery, outdoor dining, red-brick building, scooter, sidewalk, white-framed windows,
> neo-Georgian architecture, red chairs, clear blue sky, sunny day, corner building,
> people socializing
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Description sentence violation (3; expected 1-2)

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** Load 3.65s | Gen 13.39s | Total 17.80s
**Throughput:** Prompt 734 TPS (2,875 tok) | Gen 26.8 TPS (241 tok)
**Assessment:** ✅ B (67/100) | Δ-5 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo)
**Review:** B 67/100 | Description sentence violation (6; expected 1-2); Keyword count violation (20; expected 10-18); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Howardsgate in Welwyn Garden City, England, UK.
>
> Description: A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn
> Garden City, England. Locals and visitors take advantage of the clear skies, gathering
> at the outdoor tables of a popular corner bakery. The scene is set against the town's
> signature neo-Georgian red-brick architecture, a hallmark of its design and aesthetic.
> The bakery, with its red awning and inviting atmosphere, is a focal point of the street,
> drawing people in with its promise of freshly baked goods and a warm welcome. The
> sidewalk is bustling with activity, as people engage in conversation and enjoy their
> food and drinks. The clear blue sky and bright sunshine add to the overall sense of
> vibrancy and community, making this a quintessential English town scene.
>
> Keywords: Howardsgate, Welwyn Garden City, England, UK, Neo-Georgian Architecture,
> Corner Building, Outdoor Cafe, Outdoor Dining, Food Delivery, Scooter, GAIL's Bakery,
> Hertfordshire, Spring, Socializing, Sidewalk, Adobe Stock, Any Vision, Europe, English
> Town Centre, English Town.<end_of_utterance>
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Description sentence violation (6; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Context echo (28% overlap)
- Unknown tags: &lt;end_of_utterance&gt;

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

**Metrics:** Load 2.24s | Gen 13.52s | Total 16.12s
**Throughput:** Prompt 307 TPS (3,194 tok) | Gen 47.7 TPS (128 tok)
**Assessment:** 🏆 A (88/100) | Δ+16 | None identified
**Review Status:** strong candidate for first-pass review
**Review:** A 88/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> \\*\\*Title:\\*\\*
> Gail's Bakery Corner with Outdoor Café and Scooter
>
> \\*\\*Description:\\*\\*
> A bright sunny day showcases a brick bakery-café corner, Gail’s Bakery, with outdoor
> seating filled with patrons. A parked food delivery scooter sits on the sidewalk, while
> neo-Georgian architecture frames the scene.
>
> \\*\\*Keywords:\\*\\*
> Bakery, brick building, café, delivery scooter, neo-Georgian architecture, outdoor
> dining, sidewalk, brick facade, red chairs, urban street, brick storefront, clear blue
> sky, brick corner building, brick exterior, brick windows, brick building facade
<!-- markdownlint-enable MD028 MD049 -->


**Quality Status:** no quality issues detected in this run

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** Load 1.52s | Gen 13.65s | Total 15.50s
**Throughput:** Prompt 256 TPS (2,819 tok) | Gen 42.7 TPS (94 tok)
**Assessment:** ✅ B (74/100) | Δ+3 | Missing requested structure
**Review Status:** watchlist (context ignored, missing sections)
**Review:** B 74/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> The image is a photograph of a brick building with a red awning. The building has a
> white facade and a balcony on the second floor. There are several windows visible on the
> building. In front of the building, there is a sidewalk with a tree planted in the
> center. A motorcycle is parked on the sidewalk. The sky is clear and blue, indicating a
> sunny day. There are no visible texts or brands in the image.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** Load 1.51s | Gen 15.04s | Total 16.98s
**Throughput:** Prompt 865 TPS (1,416 tok) | Gen 38.7 TPS (500 tok)
**Assessment:** ❌ F (33/100) | Δ-38 | Mostly echoes context without adding value
**Review Status:** watchlist (repetitive)
**Review:** F 33/100 | Repetitive output (phrase: "stonehills, stonehills, stoneh..."); Title length violation (4 words; expected 5-10); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Welwyn Garden City, England
>
> Description: A clear day in Howardsgate, a town in Welwyn Garden City, England, with
> people enjoying outdoor dining at a bakery. The scene is set against a red-brick
> neo-Georgian building.
>
> Keywords: Welwyn Garden City, England, Howardsgate, neo-Georgian architecture, outdoor
> dining, red-brick building, clear skies, socializing, outdoor cafe, bakery,
> Hertfordshire, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills, stonehills, stonehills,
> stonehills, stonehills, stonehills, stonehills, stonehills
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Repetitive output (phrase: &quot;stonehills, stonehills, stoneh...&quot;)
- Title length violation (4 words; expected 5-10)
- Keyword count violation (106; expected 10-18)
- Keyword duplication (89% duplicated terms)

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** Load 2.58s | Gen 16.47s | Total 19.44s
**Throughput:** Prompt 1,584 TPS (4,675 tok) | Gen 38.6 TPS (500 tok)
**Assessment:** ✅ B (66/100) | Δ-6 | Lacks visual description of image
**Review Status:** watchlist (context ignored, missing sections, reasoning leak, repetitive)
**Review:** B 66/100 | Repetitive output (phrase: "- do not output..."); Excessive bullet points (46); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Do not output reasoning, notes, hedging, or extra sections.
> \- Do not output prompt instructions into the Title, Description, or Keywords fields.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output reasoning, notes, hedging, or extra sections.
> \- Do not output prompt instructions into the Title, Description, or Keywords fields.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output anything uncertain or inferred.
> \- Output only the description text after the label.
> \- Do not output
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Repetitive output (phrase: &quot;- do not output...&quot;)
- Excessive bullet points (46)
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)
- Reasoning leak (do not output reasoning)

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** Load 3.84s | Gen 23.36s | Total 27.65s
**Throughput:** Prompt 152 TPS (2,695 tok) | Gen 19.2 TPS (99 tok)
**Assessment:** ❌ F (7/100) | Δ-64 | Output too short to be useful
**Review Status:** watchlist (harness, missing sections)
**Review:** F 7/100 | ⚠️HARNESS:encoding; Missing sections (description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:ĠCornerĠBakeryĠwithĠOutdoorĠDiningĊĊDescription:ĠAĠred-brickĠbuildingĠwithĠwhite-framedĠwindowsĠhousesĠaĠbakeryĠwithĠoutdoorĠseating.ĠPeopleĠareĠgatheredĠatĠtablesĠonĠtheĠsidewalk,ĠandĠaĠscooterĠisĠparkedĠnearby.ĊĊKeywords:Ġbakery,ĠoutdoorĠdining,Ġred-brickĠbuilding,ĠwhiteĠwindows,Ġsidewalk,Ġscooter,Ġpeople,Ġtables,Ġchairs,ĠclearĠsky,ĠurbanĠsetting,Ġneo-GeorgianĠarchitecture,ĠcornerĠbuilding,Ġsocializing,Ġspring,Ġdaylight
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- ⚠️HARNESS:encoding
- Missing sections (description, keywords)
- Title length violation (60 words; expected 5-10)
- token_encoding:bpe_space_leak(54)

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

**Metrics:** Load 13.44s | Gen 25.70s | Total 39.56s
**Throughput:** Prompt 232 TPS (1,901 tok) | Gen 30.1 TPS (500 tok)
**Assessment:** ✅ B (70/100) | Δ-2 | None identified
**Review Status:** watchlist (missing sections)
**Review:** B 70/100 | Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and
> factual. The image is a street scene with a red-brick building, outdoor seating, a
> scooter, and people. Wait, the context mentions Howardsgate, Welwyn Garden City,
> England. But wait, the user said to use only details clearly visible. Wait, the prompt's
> context has a caption that says Howardsgate, Welwyn Garden City, England. But let me
> check the image description again. The user's initial prompt has a lot of garbled text,
> but the context says the caption is about Howardsgate. But wait, the rules say to use
> only details clearly visible. Wait, no—the context says "Existing metadata hints (high
> confidence; use only when visually confirmed)". Oh, right, the context's description
> hint is a caption, but the user says to use the metadata hints only when visually
> confirmed. Wait, no, the rules say: "Treat the metadata hints below as a draft catalog
> record. Keep only details that are clearly confirmed by the image, correct anything
> contradicted by the image, and add important visible details that are definitely
> present." Wait, the context's description hint is a caption, but is that confirmed?
> Wait, the user's initial prompt has a lot of garbled text, but the context says the
> caption is about Howardsgate. But let's go back. The title: the context's caption says
> "A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn Garden City,
> England." But is Howardsgate clearly visible? Wait, the image has a sign that says
> "GAIL's BAKERY" (from the keyword hints). The building is red-brick, neo-Georgian (from
> keyword hints). The scene has outdoor seating, people, a scooter. Let's start with the
> title. The title needs to be concrete. Maybe "Red-Brick Neo-Georgian Building with
> Outdoor Seating and Scooter". Wait, but the context's caption mentions Howardsgate, but
> is that visible? The user's initial prompt has a lot of garbled text, but the context
> says the caption is about Howardsgate. But the rules say to use only details clearly
> visible. Wait, the context's description hint is a caption, but is that confirmed
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** Load 3.15s | Gen 33.06s | Total 36.55s
**Throughput:** Prompt 1,529 TPS (1,603 tok) | Gen 15.8 TPS (500 tok)
**Assessment:** 🏆 A (80/100) | Δ+9 | None identified
**Review Status:** watchlist (context ignored, missing sections, repetitive)
**Review:** A 80/100 | Repetitive output (phrase: "awning over it. the..."); Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \- The scene is set against the town's signature neo-Georgian red-brick architecture. A
> corner bakery is visible in the lower left corner of the frame. The exterior of the
> building is a red brick with white trim. The windows are divided into three sections.
> The top section has a white awning over it. The middle section has a red awning over it.
> The bottom section has a white awning over it. The windows are divided into three
> sections. The top section has a white awning over it. The middle section has a red
> awning over it. The bottom section has a white awning over it. The windows are divided
> into three sections. The top section has a white awning over it. The middle section has
> a red awning over it. The bottom section has a white awning over it. The windows are
> divided into three sections. The top section has a white awning over it. The middle
> section has a red awning over it. The bottom section has a white awning over it. The
> windows are divided into three sections. The top section has a white awning over it. The
> middle section has a red awning over it. The bottom section has a white awning over it.
> The windows are divided into three sections. The top section has a white awning over
> it. The middle section has a red awning over it. The bottom section has a white awning
> over it. The windows are divided into three sections. The top section has a white awning
> over it. The middle section has a red awning over it. The bottom section has a white
> awning over it. The windows are divided into three sections. The top section has a white
> awning over it. The middle section has a red awning over it. The bottom section has a
> white awning over it. The windows are divided into three sections. The top section has a
> white awning over it. The middle section has a red awning over it. The bottom section
> has a white awning over it. The windows are divided into three sections. The top section
> has a white awning over it. The middle section has a red awning over it. The bottom
> section has a white awning over it. The windows are divided into three sections. The top
> section has a white awning over it. The middle section has a red awning over it. The
> bottom section has a white awning over it. The windows are divided into three sections.
> The top section has a white awning over it. The middle section has a red awning over it.
> The bottom section has a white awning
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Repetitive output (phrase: &quot;awning over it. the...&quot;)
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** Load 2.80s | Gen 36.05s | Total 39.58s
**Throughput:** Prompt 256 TPS (3,478 tok) | Gen 22.7 TPS (500 tok)
**Assessment:** 🏆 A (81/100) | Δ+10 | None identified
**Review Status:** watchlist (missing sections, reasoning leak, repetitive)
**Review:** A 81/100 | Repetitive output (phrase: "a "gail's" sign on..."); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Here are my reasoning steps:
> We need to produce metadata: Title, Description, Keywords. Use only details that are
> clearly visible in the image. The image shows a brick building with a bakery (looks like
> GAIL's Bakery, as sign says "GAIL'S" on the red awning). There's a scooter parked in
> front, a person on a scooter. Outdoor tables with red umbrellas? Actually there are red
> tables and chairs outside. There's a balcony with black railing. The building is red
> brick with white window frames, a dormer window on top. The street is paved, there are
> people walking, some sitting at tables. The sky is clear blue. The location appears to
> be Howardsgate, Welwyn Garden City, England, UK. The date is 2026-03-14 15:19:38 GMT.
> The image is bright, likely March afternoon. The building is neo-Georgian style. The
> bakery is GAIL's. There's a scooter (food delivery scooter). There's a sign "GAIL'S" on
> the awning. There's a small balcony with black wrought iron railing. There's a small
> tree with no leaves. There's a small grass patch. There's a street sign "Howardsgate".
> There's a "GAIL'S" sign on the awning. There's a "Bakery" sign? The awning is red.
> There's a "GAIL'S" sign on the window? The building has a "GAIL'S" sign on the awning.
> There's a "GAIL'S" sign on the window? The building has a "GAIL'S" sign on the awning.
> There's a "GAIL'S" sign on the awning. There's a "GAIL'S" sign on the awning. There's a
> "GAIL'S" sign on the awning. There's a "GAIL'S" sign on the awning. There's a "GAIL'S"
> sign on the awning. There's a "GAIL'S" sign on the awning. There's a "GAIL'S" sign on
> the awning. There's a "GAIL'S" sign on the awning. There's a "GAIL'S" sign on the
> awning. There's a "GAIL
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Repetitive output (phrase: &quot;a &quot;gail&#x27;s&quot; sign on...&quot;)
- Missing sections (title, description, keywords)
- Reasoning leak (here are my reasoning steps)

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** Load 3.36s | Gen 36.18s | Total 39.87s
**Throughput:** Prompt 0 TPS (0.0 tok) | Gen 0 TPS (0.0 tok)
**Assessment:** ❌ F (0/100) | Δ-72 | Empty or minimal output
**Review Status:** watchlist (harness)
**Review:** F 0/100 | ⚠️HARNESS:prompt_template; output:zero_tokens

<!-- markdownlint-disable MD028 MD049 -->
>
>
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- ⚠️HARNESS:prompt_template
- output:zero_tokens

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

**Metrics:** Load 2.19s | Gen 37.04s | Total 39.60s
**Throughput:** Prompt 246 TPS (6,704 tok) | Gen 53.7 TPS (500 tok)
**Assessment:** 🏆 A (85/100) | Δ+14 | None identified
**Review Status:** watchlist (context ignored, missing sections, reasoning leak)
**Review:** A 85/100 | Excessive bullet points (26); Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;think&gt;Got it, let's tackle this. The user wants cataloging metadata with three
> sections: Title, Description, Keywords.
>
> First, the Title. Need 5-10 words, concrete and factual. Looking at the image: there's a
> red-brick building with a bakery, a scooter, outdoor seating. The title should capture
> the main elements. Maybe "Red-brick Corner Building with Bakery" — but need to check the
> visible details. The building is a corner building, red brick, with a bakery (sign
> "BAKERY"). So title could be "Red-brick Corner Building with Bakery".
>
> Next, Description. Need 1-2 sentences, factual, visible. Let's list visible elements:
> red brick building with white window frames, a bakery with red awning, outdoor red
> chairs and table, a motor scooter parked on the street, pedestrians in the background,
> clear blue sky, small tree in a grassy patch on the sidewalk.
>
> So description: "A red-brick corner building with a bakery (red awning) and outdoor
> seating (red chairs, table) is shown. A motor scooter is parked on the street, and
> pedestrians are visible in the background under a clear blue sky."
>
> Now Keywords. Need 10-18 unique terms, based on visible content. Let's extract: red
> brick building, corner building, bakery, red awning, outdoor seating, red chairs, motor
> scooter, street, pedestrians, clear blue sky, small tree, grassy patch, sidewalk.
>
> Now select unique, relevant terms. Let's list possible keywords:
>
> \- Red brick building
> \- Corner building
> \- Bakery
> \- Red awning
> \- Outdoor seating
> \- Red chairs
> \- Motor scooter
> \- Street
> \- Pedestrians
> \- Clear blue sky
> \- Small tree
> \- Grassy patch
> \- Sidewalk
>
> Now pick 10-18. Let's choose the most distinct and visible ones. For example:
>
> \- Red brick building
> \- Corner building
> \- Bakery
> \- Red awning
> \- Outdoor seating
> \- Red chairs
> \- Motor scooter
> \- Street
> \- Pedestrians
> \- Clear blue sky
> \- Small tree
> \- Grassy patch
> \- Sidewalk
>
> But need to ensure uniqueness. Let's check if any are redundant. For example, "red brick
> building" is clear. "Corner building" is also clear. "Bakery" is visible. "Red awning"
> is visible. "
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Excessive bullet points (26)
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** Load 1.87s | Gen 37.58s | Total 39.78s
**Throughput:** Prompt 0 TPS (0.0 tok) | Gen 0 TPS (0.0 tok)
**Assessment:** ❌ F (0/100) | Δ-72 | Empty or minimal output
**Review Status:** watchlist (harness)
**Review:** F 0/100 | ⚠️HARNESS:prompt_template; output:zero_tokens

<!-- markdownlint-disable MD028 MD049 -->
>
>
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- ⚠️HARNESS:prompt_template
- output:zero_tokens

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** Load 2.48s | Gen 41.63s | Total 44.45s
**Throughput:** Prompt 237 TPS (6,704 tok) | Gen 38.8 TPS (500 tok)
**Assessment:** 🏆 A (96/100) | Δ+24 | None identified
**Review Status:** watchlist (context ignored, missing sections, reasoning leak)
**Review:** A 96/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to extract the Title,
> Description, and Keywords based only on what's clearly visible in the image.
>
> Starting with the Title. The instructions say 5-10 words, concrete and factual, limited
> to clearly visible content. Looking at the image: there's a red brick building with a
> bakery (GAIL's) and outdoor seating. Also, a scooter parked in front. So possible title:
> "Red Brick Building with Bakery" or more specific. Wait, the image has "GAIL's" sign,
> so maybe "Red Brick Building with GAIL's Bakery". But need to keep it concise. Let's
> check the image again: the main subject is the red brick building with the bakery
> (GAIL's) and outdoor seating. So a good title would be "Red Brick Building with GAIL's
> Bakery". That's within 5-10 words.
>
> Next, Description. Need 1-2 factual sentences describing main visible subject, setting,
> lighting, action, etc. Let's list visible elements:
>
> \- Red brick building with white-framed windows and a tiled roof.
> \- Ground floor has "GAIL's" sign and a bakery with outdoor red chairs and tables.
> \- A person on a black and white scooter parked on the street.
> \- Sidewalk with a small tree and grass patch.
> \- Clear blue sky, bright daylight.
> \- People in the background near outdoor seating, socializing.
>
> Now, construct a description that includes these key points without inference. Let's
> draft:
>
> "A red brick building with white-framed windows and a tiled roof features a ground-floor
> bakery named 'GAIL's' with outdoor red chairs and tables. A person on a black and white
> scooter is parked on the street, and the scene includes a sidewalk with a small tree
> and grass patch under a clear blue sky."
>
> Wait, but need to keep it concise, 1-2 sentences. Let's simplify:
>
> "Red brick building with white-framed windows and a tiled roof houses 'GAIL's' bakery on
> the ground floor, with outdoor red chairs and tables. A person on a black and white
> scooter is parked on the street, and the scene includes a sidewalk with a small tree and
> grass patch under a clear blue sky."
>
> But maybe even shorter. Let's check the key elements: red brick building, GAIL's bakery
> (ground floor), outdoor red chairs/tables, scooter parked on street, sidewalk
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

**Metrics:** Load 2.15s | Gen 48.15s | Total 50.63s
**Throughput:** Prompt 376 TPS (16,810 tok) | Gen 47.0 TPS (139 tok)
**Assessment:** 🏆 A (90/100) | Δ+18 | None identified
**Review Status:** strong candidate for first-pass review
**Review:** A 90/100 | Title length violation (3 words; expected 5-10); Keyword count violation (24; expected 10-18)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> GAIL's Bakery Exterior
>
> Description:
> A bright, sunny day at GAIL's Bakery in Howardsgate, Welwyn Garden City, with customers
> enjoying outdoor seating and a delivery scooter parked nearby. The bakery's red-brick
> facade and white-framed windows are prominent, and the clear blue sky enhances the
> cheerful atmosphere.
>
> Keywords:
> GAIL's Bakery, Howardsgate, Welwyn Garden City, England, UK, Red Brick, White Windows,
> Outdoor Cafe, Delivery Scooter, Clear Sky, Spring, Bakery, Cafe, Sidewalk, People, Red
> Chairs, Tree, Signage, Motorbike, Street, Daylight, Architecture, Community,
> Socializing.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Title length violation (3 words; expected 5-10)
- Keyword count violation (24; expected 10-18)

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** Load 1.47s | Gen 50.07s | Total 51.87s
**Throughput:** Prompt 353 TPS (16,799 tok) | Gen 74.6 TPS (139 tok)
**Assessment:** ✅ B (65/100) | Δ-6 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo)
**Review:** B 65/100 | Context echo (32% overlap)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Corner Bakery, Howardsgate, Welwyn Garden City
>
> Description:
> A bright spring day in Howardsgate, Welwyn Garden City, England, features a corner
> bakery with red outdoor seating and a scooter parked on the sidewalk. The scene is set
> against a backdrop of red-brick buildings with white-framed windows and a clear blue
> sky.
>
> Keywords:
> Corner Building, England, English Town Centre, English town, Europe, Food Delivery, Food
> Delivery Scooter, GAIL's Bakery, Hertfordshire, Howardsgate, Neo-Georgian Architecture,
> Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring,
> Stonehills
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context echo (32% overlap)

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

**Metrics:** Load 13.98s | Gen 70.08s | Total 84.50s
**Throughput:** Prompt 131 TPS (6,704 tok) | Gen 27.0 TPS (500 tok)
**Assessment:** 🏆 A (84/100) | Δ+12 | None identified
**Review Status:** watchlist (context ignored, missing sections, reasoning leak)
**Review:** A 84/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT); Missing sections (title, description); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, the Title. It needs to be 5-10
> words, concrete and factual. Looking at the image, the main subject is a brick building
> with a bakery (GAIL's) on the corner, a scooter, and outdoor seating. So maybe "Corner
> brick building with GAIL's Bakery and outdoor seating". Wait, but let's check: the image
> has a brick building, GAIL's sign, outdoor tables. Let's make it concise. Maybe "Brick
> building with GAIL's Bakery and outdoor seating". That's 8 words, good.
>
> Next, Description. Need 1-2 factual sentences. The main subject is the brick building,
> the bakery, the scooter, people at outdoor tables. Lighting is bright, clear sky. So: "A
> brick building with a GAIL's Bakery storefront features outdoor red tables and chairs
> where people gather. A scooter is parked on the sidewalk near the building, and the
> scene is set under a clear blue sky." Wait, but let's make sure it's factual. The
> building is brick, GAIL's is visible, outdoor seating with red chairs, scooter parked,
> clear sky. Yes.
>
> Keywords: Need 10-18 unique terms. Let's list visible ones: Brick Building, GAIL's
> Bakery, Outdoor Seating, Red Chairs, Scooter, Sidewalk, Clear Sky, People Gathering,
> Bakery, Corner Building, Red Tables, Urban Setting, Commercial Building, Street Scene,
> Daytime, England (wait, but the hint says England, but is it definitely visible? The
> hint says existing metadata has England, but the image doesn't show a sign that says
> England. Wait, the rules say only details definitely visible. The hint says "England" is
> a hint, but is it visible? The image doesn't have a sign that says England. So maybe
> omit. Let's list what's definitely visible: Brick Building, GAIL's Bakery, Outdoor
> Seating, Red Chairs, Scooter, Sidewalk, Clear Sky, People Gathering, Bakery, Corner
> Building, Red Tables, Urban Setting, Commercial Building, Street Scene, Daytime. Let's
> count: 14, which is within 10-18. Let's check: Brick Building, GAIL's Bakery, Outdoor
> Seating, Red Chairs, Scooter, Sidewalk, Clear Sky, People Gathering, Bakery, Corner
> Building, Red Tables, Urban Setting,
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description)
- Keyword count violation (47; expected 10-18)
- Keyword duplication (51% duplicated terms)
- Reasoning leak (&lt;think&gt;)
- Unknown tags: &lt;think&gt;

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** Load 4.69s | Gen 73.53s | Total 78.56s
**Throughput:** Prompt 142 TPS (564 tok) | Gen 2.51 TPS (173 tok)
**Assessment:** 🏆 A (84/100) | Δ+12 | None identified
**Review Status:** watchlist (context ignored, missing sections)
**Review:** A 84/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> Gail's Bakery in a Corner of a Red-Brick Building
>
> Gail's Bakery is situated in a corner of a red-brick building, with a white trim and a
> red awning. The building has a balcony on the second floor, and the bakery's storefront
> is on the ground floor. The building is made of red brick with white trim, and the
> balcony has a black railing. The bakery's storefront has a red awning with white letters
> that read "Bakery". The building has a few windows on the second and third floors, and
> the roof is made of brown tiles. The building is situated on a corner, with a sidewalk
> in front of it. There are several people sitting at tables outside the bakery, enjoying
> their food and drinks. The sky is clear and blue, and the sun is shining down on the
> scene.
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** Load 1.47s | Gen 77.86s | Total 79.67s
**Throughput:** Prompt 242 TPS (16,801 tok) | Gen 63.0 TPS (500 tok)
**Assessment:** 🏆 A (93/100) | Δ+22 | None identified
**Review Status:** watchlist (context ignored, harness, long context)
**Review:** A 93/100 | ⚠️HARNESS:long_context; Context ignored (missing: England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Got it, let's tackle this step by step. First, I need to analyze the image for metadata,
> focusing only on clearly visible details. Let's start with the Title.
>
> Title: Needs 5-10 words, concrete and factual, from visible content. The image shows a
> bakery with a scooter, brick building, outdoor seating. The caption mentions
> Howardsgate, Welwyn Garden City, but the image itself—wait, the user says to use only
> details clearly visible. The image has a bakery with "GAIL'S BAKERY" sign, brick
> building, outdoor tables, scooter. Let's check the visible elements. The main subject is
> the bakery, so maybe "GAIL'S BAKERY OUTDOOR SEATING" but need to be concrete. Wait, the
> image has "GAIL'S BAKERY" sign, brick building, outdoor area with red chairs, scooter.
> Let's see: the title should be factual. Maybe "GAIL'S BAKERY OUTDOOR SEATING" but let's
> check if that's visible. The sign says "GAIL'S BAKERY" so that's part of the title.
> Wait, the title needs to be 5-10 words. Let's see: "GAIL'S BAKERY OUTDOOR SEATING" is 7
> words. That's okay.
>
> Description: 1-2 sentences. Main visible subject: a brick building with "GAIL'S BAKERY"
> sign, outdoor seating with red chairs, people sitting, a scooter parked. Setting: sunny
> day, blue sky, brick building. Lighting: bright sun, shadows. Action: people sitting at
> tables, scooter parked. Distinctive details: red chairs, outdoor tables, brick building,
> blue sky. Let's make sure: "A brick building with 'GAIL'S BAKERY' sign features outdoor
> seating with red chairs, people gathered at tables, and a scooter parked on the
> sidewalk under a clear blue sky." Wait, but need to be factual. The image shows the
> bakery, outdoor area, people, scooter. So description: "A brick building with 'GAIL'S
> BAKERY' sign features outdoor seating with red chairs, people gathered at tables, and a
> scooter parked on the sidewalk under a clear blue sky."
>
> Keywords: 10-18 unique terms. From visible: GAIL'S BAKERY, brick building, outdoor
> seating, red chairs, people, scooter,
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- ⚠️HARNESS:long_context
- Context ignored (missing: England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)
- Title length violation (143 words; expected 5-10)
- Description sentence violation (9; expected 1-2)
- Keyword count violation (6; expected 10-18)
- long_context_context_drop(16801tok)

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** Load 2.71s | Gen 86.75s | Total 89.81s
**Throughput:** Prompt 141 TPS (563 tok) | Gen 6.08 TPS (500 tok)
**Assessment:** 🏆 A (83/100) | Δ+11 | None identified
**Review Status:** watchlist (context ignored, missing sections)
**Review:** A 83/100 | Context ignored (missing: Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title
> Gail's Bakery on a Sunny Day in Howardsgate, England
>
> Description
> The image shows a corner bakery on a sunny day in Howardsgate, England. The building is
> a red-brick structure with a white trim and a red awning. The bakery is located on the
> ground floor, and there are several tables and chairs outside where people are sitting
> and enjoying their food. The sky is clear and blue, and the atmosphere is lively and
> inviting.
>
> Keywords
> Gail's Bakery, Howardsgate, England, Red Brick, White Trim, Red Awning, Corner Bakery,
> Outdoor Seating, Sunny Day, Lively Atmosphere, Inviting, Blue Sky, Clear Sky, People
> Sitting, Food, Tables, Chairs, Potted Plants, Trees, Sidewalk, Street, Road, Vehicles,
> Trees, Building, Architecture, Urban, City, Town, Community, Neighborhood, Local,
> Culture, Lifestyle, Daily Life, Routine, Habits, Traditions, Customs, Practices,
> Activities, Events, Celebrations, Festivals, Holidays, Occasions, Gatherings, Meetings,
> Socializing, Communication, Interaction, Relationships, Community, Neighborhood, Local,
> Culture, Lifestyle, Daily Life, Routine, Habits, Traditions, Customs, Practices,
> Activities, Events, Celebrations, Festivals, Holidays, Occasions, Gatherings, Meetings,
> Socializing, Communication, Interaction, Relationships, Community, Neighborhood, Local,
> Culture, Lifestyle, Daily Life, Routine, Habits, Traditions, Customs, Practices,
> Activities, Events, Celebrations, Festivals, Holidays, Occasions, Gatherings, Meetings,
> Socializing, Communication, Interaction, Relationships, Community, Neighborhood, Local,
> Culture, Lifestyle, Daily Life, Routine, Habits, Traditions, Customs, Practices,
> Activities, Events, Celebrations, Festivals, Holidays, Occasions, Gatherings, Meetings,
> Socializing, Communication, Interaction, Relationships, Community, Neighborhood, Local,
> Culture, Lifestyle, Daily Life, Routine, Habits, Traditions, Customs, Practices,
> Activities, Events, Celebrations, Festivals, Holidays, Occasions, Gatherings, Meetings,
> Socializing, Communication, Interaction, Relationships, Community, Neighborhood, Local,
> Culture, Lifestyle, Daily Life, Routine, Habits, Traditions, Customs, Practices,
> Activities, Events, Celebrations, Festivals, Holidays, Occasions, Gatherings, Meetings,
> Socializing, Communication, Interaction
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Welwyn Garden City, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Spring, Stonehills
Capture, GMT)
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

**Metrics:** Load 5.61s | Gen 88.24s | Total 94.18s
**Throughput:** Prompt 208 TPS (16,824 tok) | Gen 71.2 TPS (500 tok)
**Assessment:** 🏆 A (86/100) | Δ+15 | None identified
**Review Status:** watchlist (context ignored, hallucination, missing sections, refusal)
**Review:** A 86/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT); Refusal detected (explicit_refusal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> The user wants me to generate cataloguing metadata (Title, Description, Keywords) based
> on the provided image.
> I need to be strict about what is "clearly and definitely visible." I cannot infer
> things that aren't there, even if the prompt hints suggest them (like specific location
> names unless visible on signs).
>
> \\*\\*1. Analyze the Image:\\*\\*
> \*   \\*\\*Subject:\\*\\* A red brick building with white window frames. It looks like a shop on
> the ground floor and apartments/offices above.
> \*   \\*\\*Shop:\\*\\* There are signs that say "GAIL's" and "BAKERY". There are red tables and
> chairs outside. People are sitting and standing.
> \*   \\*\\*Foreground:\\*\\* A white scooter/motorcycle with a delivery box on the back is parked
> on the street. A person wearing a helmet is sitting on it.
> \*   \\*\\*Building details:\\*\\* Red brick facade, white sash windows, dormer windows on the
> roof. A small balcony with a black railing on the left.
> \*   \\*\\*Setting:\\*\\* Paved sidewalk, street curb. Blue sky.
> \*   \\*\\*Lighting:\\*\\* Bright sunlight, casting strong shadows. Looks like a clear day.
> \*   \\*\\*Text visible:\\*\\* "GAIL's", "BAKERY", "STONEHILLS" (on a plaque above the shop).
>
> \\*\\*2. Draft Title:\\*\\*
> \*   \*Constraint:\* 5-10 words, concrete, factual.
> \*   \*Draft 1:\* Gail's Bakery storefront with delivery scooter parked outside. (9 words)
> \- Good.
> \*   \*Draft 2:\* Red brick building with Gail's Bakery and outdoor seating. (9 words) -
> Good.
> \*   \*Draft 3:\* Delivery scooter parked in front of Gail's Bakery on a sunny day. (11
> words) - Too long.
> \*   \*Selection:\* Red brick building housing Gail's Bakery with outdoor seating. (9
> words) - Wait, let's look closer. The sign says "STONEHILLS" too.
> \*   \*Refined Selection:\* Gail's Bakery storefront on a red brick building with a
> delivery scooter. (11 words) - Still a bit long.
> \*   \*Refined Selection 2:\* Delivery scooter parked outside Gail's Bakery on a red brick
> building. (11 words).
> \*   \*Refined Selection 3:\* Gail's Bakery
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Socializing, Spring, Stonehills
Capture, GMT)
- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Contains question/quiz content

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Metrics:** Load 1.01s | Gen 95.11s | Total 96.44s
**Throughput:** Prompt 179 TPS (16,810 tok) | Gen 166 TPS (126 tok)
**Assessment:** ❌ F (29/100) | Δ-43 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo)
**Review:** F 29/100 | Description sentence violation (3; expected 1-2); Context echo (96% overlap)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Howardsgate, Welwyn Garden City, England, United Kingdom, UK
> Description: A bright March afternoon brings a touch of warmth to Howardsgate in Welwyn
> Garden City, England. Locals and visitors take advantage of the clear skies, gathering
> at the outdoor tables of a popular corner bakery. The scene is set against the town's
> signature neo-Georgian red-brick architecture, a hallmark of its design a...
> Keywords: Howardsgate, Welwyn Garden City, England, United Kingdom, UK, Neo-Georgian
> Architecture, Outdoor Cafe, Outdoor Dining, Spring, Stonehills
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Description sentence violation (3; expected 1-2)
- Context echo (96% overlap)

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

**Metrics:** Load 15.05s | Gen 96.63s | Total 112.07s
**Throughput:** Prompt 194 TPS (16,824 tok) | Gen 53.7 TPS (500 tok)
**Assessment:** ✅ B (74/100) | Δ+3 | Missing requested structure
**Review Status:** watchlist (hallucination, missing sections, refusal)
**Review:** B 74/100 | Refusal detected (explicit_refusal); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> The user wants me to generate cataloguing metadata (Title, Description, Keywords) based
> on the provided image.
> I need to be strict about what is "clearly and definitely visible." I cannot infer
> things that aren't there, even if the prompt hints suggest them (like specific location
> names unless visible on a sign).
>
> \\*\\*1. Analyze the Image:\\*\\*
> \*   \\*\\*Subject:\\*\\* A red brick building with white window frames. It looks like a shop on
> the ground floor and apartments/offices above.
> \*   \\*\\*Foreground:\\*\\* A white scooter/motorcycle with a delivery box on the back. A person
> wearing a helmet is sitting on it.
> \*   \\*\\*Shop:\\*\\* There are signs that say "GAIL's" and "BAKERY". There are red tables and
> chairs outside. People are sitting at them.
> \*   \\*\\*Building details:\\*\\* Red brick facade, white sash windows, dormer windows on the
> roof. A small balcony with a black railing.
> \*   \\*\\*Setting:\\*\\* Paved sidewalk/street. Blue sky. Trees (bare branches, suggesting early
> spring or late autumn).
> \*   \\*\\*Lighting:\\*\\* Bright sunlight, sharp shadows.
> \*   \\*\\*Text visible:\\*\\* "GAIL's", "BAKERY", "STONEHILLS" (on a street sign).
>
> \\*\\*2. Evaluate Metadata Hints vs. Image:\\*\\*
> \*   \*Hint: Howardsgate, Welwyn Garden City, England, United Kingdom, UK.\* -> I see a
> street sign "STONEHILLS". I do \*not\* see "Howardsgate". I should probably stick to
> "Stonehills" if I mention the street, or just omit the specific street name if I want to
> be safe, but "Stonehills" is clearly visible. The prompt says "Reuse metadata terms
> only when they are clearly supported by the image." So I can use "Stonehills" but not
> "Howardsgate" or "Welwyn Garden City" unless I see it. Wait, looking closely at the
> street sign, it says "STONEHILLS". The prompt hints mention "Howardsgate". I will stick
> to what is visible: "Stonehills".
> \*   \*Hint: GAIL's Bakery.\* -> Visible. "GAIL's" and "BAKERY" signs are clear.
> \*   \*Hint: Neo-Georgian Architecture.\* -> The building is red brick with sash windows
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Contains question/quiz content

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

**Metrics:** Load 3.74s | Gen 204.74s | Total 208.95s
**Throughput:** Prompt 93.0 TPS (16,824 tok) | Gen 21.5 TPS (500 tok)
**Assessment:** 🏆 A (94/100) | Δ+22 | None identified
**Review Status:** watchlist (hallucination, missing sections, reasoning leak)
**Review:** A 94/100 | Missing sections (title, description, keywords); Reasoning leak (description hint:, keyword hints:); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> The user wants me to analyze the image and generate metadata (Title, Description,
> Keywords) based on strict rules.
>
> \\*\\*1. Analyze the Image:\\*\\*
> \*   \\*\\*Subject:\\*\\* A corner building made of red brick. It has white window frames.
> \*   \\*\\*Ground Floor:\\*\\* There are shop fronts. One sign says "GAIL's". Another sign says
> "BAKERY". There are red awnings.
> \*   \\*\\*Upper Floors:\\*\\* Two stories of brick with white sash windows. There are dormer
> windows on the roof.
> \*   \\*\\*Foreground:\\*\\* A white scooter/motorcycle is parked on the street corner. A rider
> is on it, wearing a helmet.
> \*   \\*\\*Mid-ground:\\*\\* Outdoor seating area with red chairs and tables. People are sitting
> and standing around.
> \*   \\*\\*Background:\\*\\* Other brick buildings, trees (bare branches, suggesting early spring
> or late autumn), blue sky.
> \*   \\*\\*Lighting:\\*\\* Bright sunlight, casting strong shadows. Blue sky.
> \*   \\*\\*Specific Details:\\*\\* "STONEHILLS" sign is visible on the brickwork. "GAIL's" sign
> is visible. "BAKERY" is on the awning.
>
> \\*\\*2. Evaluate Existing Metadata Hints:\\*\\*
> \*   \*Description hint:\* Mentions "Howardsgate, Welwyn Garden City, England". This is
> specific location data. I can see "STONEHILLS" on the building, which is a known
> location in Welwyn Garden City, but I should stick to what is visually confirmed. The
> prompt says "Use only details that are clearly and definitely visible". The text
> "STONEHILLS" is visible. The text "GAIL's" is visible. The text "BAKERY" is visible. The
> location "Welwyn Garden City" is not explicitly written in big letters, though
> "STONEHILLS" is a strong indicator. I will focus on the visual elements.
> \*   \*Keyword hints:\* "GAIL's Bakery", "Neo-Georgian Architecture", "Outdoor Cafe", "Food
> Delivery Scooter". These seem accurate. "Stonehills" is visible on the building.
>
> \\*\\*3. Draft Title:\\*\\*
> \*   Needs to be 5-10 words.
> \*   Concrete and factual.
> \*   Draft 1: White scooter parked in front of Gail's bakery on a sunny day. (9 words) -
> Good.
> \*  
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, keyword hints:)
- Contains question/quiz content

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

**Metrics:** Load 5.48s | Gen 216.66s | Total 222.47s
**Throughput:** Prompt 95.8 TPS (16,824 tok) | Gen 12.3 TPS (500 tok)
**Assessment:** ✅ B (72/100) | Δ+0 | Missing requested structure
**Review Status:** watchlist (context echo, missing sections, reasoning leak)
**Review:** B 72/100 | Missing sections (title, description, keywords); Reasoning leak (description hint:, keyword hints:); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> The user wants me to analyze the image and generate cataloging metadata based on
> specific rules.
>
> \\*\\*1. Analyze the Image:\\*\\*
> \*   \\*\\*Subject:\\*\\* A corner building made of red brick. It has white window frames.
> \*   \\*\\*Ground Floor:\\*\\* There are shop fronts. One sign says "GAIL's". Another sign says
> "BAKERY". There are red tables and chairs outside.
> \*   \\*\\*Upper Floors:\\*\\* There are sash windows on the first floor and dormer windows on
> the roof.
> \*   \\*\\*Foreground:\\*\\* A white scooter/motorbike is parked on the street corner. A person
> wearing a helmet is sitting on it, looking at a phone.
> \*   \\*\\*Background/Context:\\*\\* People are walking on the sidewalk. There are other brick
> buildings in the distance. The sky is clear blue.
> \*   \\*\\*Lighting:\\*\\* Bright sunlight, casting distinct shadows. It looks like a sunny day.
> \*   \\*\\*Specific Details:\\*\\*
> \*   Sign "STONEHILLS" is visible on the brickwork above the bakery window.
> \*   Red awning over the bakery window.
> \*   Potted plants on the window sill/ledge.
> \*   A bollard (greenish pole) near the scooter.
> \*   Trees without leaves (suggesting early spring or late autumn, but the prompt
> hints at March).
>
> \\*\\*2. Evaluate Existing Metadata Hints:\\*\\*
> \*   \*Description hint:\* Mentions Howardsgate, Welwyn Garden City, England. Mentions
> "neo-Georgian red-brick architecture". Mentions "popular corner bakery".
> \*   \*Keyword hints:\* Adobe Stock, Any Vision, Corner Building, England, English Town
> Centre, English town, Europe, Food Delivery, Food Delivery Scooter, GAIL's Bakery,
> Hertfordshire, Howardsgate, Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe
> Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills.
> \*   \*Capture metadata:\* 2026-03-14. (This is a future date, likely a placeholder or
> error in the prompt's context, but I should treat the visual evidence as primary. The
> prompt says "Taken on 2026-03-14...". I will ignore the specific date for the
> description unless it's visually verifiable, which it isn't,
<!-- markdownlint-enable MD028 MD049 -->


⚠️ **Quality Warnings:**
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, keyword hints:)
- Context echo (100% overlap)

---

<!-- markdownlint-enable MD013 MD033 -->
