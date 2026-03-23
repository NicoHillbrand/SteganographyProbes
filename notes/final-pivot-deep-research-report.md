> **Superseded by:** [final-analysis-report.md](final-analysis-report.md) (2026-03-23) — the definitive single source of truth for all findings.

# Pivoting Your Steganography‑Interpretability Hackathon Toward the Highest‑Value Results

## Crux assumptions and the cheapest ways to check them

Your project is already doing something many teams skip: aggressively separating “I saw a metric move” from “I learned something real.” The last 18 hours should be built around the **few assumptions that, if false, invalidate whole branches of work**—and the fastest possible checks that would change your next decision.

### The core cruxes

**Crux: You have any non‑trivial “stegano‑something” signal that isn’t surface text leakage.**  
If false, every head list, steering result, and headline AUC becomes a fancy way to rediscover punctuation.  
Cheapest check: **strong text baselines matched to the exact task** (char n‑grams / TF‑IDF / simple lexical features), ideally per‑method, plus a grouped split if there’s carrier reuse. This is exactly the right first line of defense because *probes are correlational by default*—they only show decodability, not use. citeturn2search3turn1search5

**Crux: The representation you can decode is causally used by the model (not just present).**  
If false, you can get great probe AUCs forever and still have no steering or behavioral leverage.  
Cheapest check: **activation patching or targeted ablation** with a *metric that reflects the thing you claim is used*. Patching/ablation is the baseline causal tool here; there are known pitfalls, but it’s still the right “first causal step.” citeturn1search5turn1search0

**Crux: Your attribution method (your “GCM” pipeline) is specific, not a high‑variance head detector.**  
If false, steering is theater, and the top heads are whichever components have big gradients under your metric.  
Cheapest check: **permutation nulls + negative controls** (shuffle labels; swap in an “obviously irrelevant” contrast; verify rankings collapse). This aligns with the broader mechanistic‑interp playbook: causal claims require *rigorous baselines and falsification tests*, not just pretty heatmaps. citeturn2search2turn1search6turn1search16

**Crux: Your behavioral endpoint has enough signal to be worth measuring (power).**  
If false, a 4–5h “validation run” just estimates noise more precisely.  
Cheapest check: **raise the base rate** (choose an eval set where the behavior actually occurs), or use a **continuous metric** (logit/probability shift for the target token, probe score shift, etc.). If you stick with a rare binary endpoint, you need *a lot* of samples to confidently detect small effects.

### A simple rule that prevents “busywork science”
Before committing to a run, answer:  
**If this result comes back null (or positive), what do we do differently next?**  
If the honest answer is “we’d shrug and continue anyway,” the run is not a crux check—it’s procrastination with CUDA.

## What your internal evidence currently says is real vs. fake signal

From your attached audit and inventory, your project already contains a clear “map of land vs swamp”:

### The defensible core you can build a paper around
**The Llama‑3‑8B synonym presence detection finding is the one thing that looks like a real representational effect rather than a surface artifact**, because it survives stronger text baselines (including TF‑IDF char n‑grams) on the dataset you used there. That puts it in the rare category of: “probe learns something beyond obvious textual cues.”  

This matters because it aligns with what mechanistic interpretability considers the next step: moving from decodability to causal localization/verification. citeturn1search5turn2search3

### The “don’t waste another GPU hour unless you need a methods demo” region
Several of your Qwen results (suffix markers, whitespace formatting differences, etc.) are **excellent examples of why baselines matter**, but they are (by your own audit) **dominated by surface features**. That’s not a failure; it’s a clean negative result: “this setup measures surface artifacts.”  

If you keep running head attribution + steering on those, you are mostly learning: “transformers have heads that notice weird punctuation.” (Shocking. Next you’ll tell me water is wet.)

### The critical implication for your next 18 hours
You should treat the Qwen surface‑marker branch as **a methods appendix / sanity demonstration**, not the main storyline—unless your deliverable explicitly requires an end‑to‑end run to prove your pipeline works.

## What the literature suggests you should do next, and why it matches your project

Your project sits at the intersection of two literatures:

### Steganography in and with language models
Recent work frames LLM steganography as both (a) a *capability* (can models encode/decode hidden bits?) and (b) a *safety concern* (covert channels for agents, exfiltration, collusion). citeturn3search2turn5search11turn5search1turn3search1  
Key takeaways that directly map onto your decisions:

- **Steganography can emerge under optimization pressure** (not just explicit prompting), especially in multi‑agent or RL‑shaped settings. citeturn5search6turn5search10  
- There are realistic threat models where a model can be turned into a *carrier* of hidden messages (fine‑tune‑embedded payloads; extraction attacks like Unconditional Token Forcing; defenses that make extraction harder). citeturn3search0turn5search1  
- Modern “semantic” steganography is explicitly designed to evade naïve surface detection by moving the payload into higher‑level choices (concept/entity selection, semantic mappings). citeturn3search1turn3search48  
- The long arc goes back before LLMs: neural text steganography with language models has been studied since at least the LSTM era. citeturn3search12  

**Your project’s differentiator:** you’re not just asking “can we hide bits?”—you’re asking **what is represented internally, where, and can we causally intervene**. That’s the interesting seam.

### Mechanistic interpretability and causal localization
The strongest mechanistic‑interp tradition is extremely consistent about one thing:  
**correlation is cheap; causation needs interventions + nulls + careful metrics.** citeturn1search5turn2search3turn2search2  

A quick map of relevant tools:

- **Activation patching / causal tracing**: localize computations by swapping internal activations between clean and corrupted runs. citeturn1search5turn1search4  
- **Attribution patching (gradient approximations) and variants**: scalable heuristics for “where to patch,” but with known failure modes and the need for validation. citeturn1search1turn4search4turn1search16  
- **Causal scrubbing**: a principled framework for testing interpretability hypotheses via behavior‑preserving resampling ablations—basically “interpretability, but with an auditor that hates you (in a good way).” citeturn2search2turn2search8  
- **Circuit work exemplars** (IOI, ROME): show what “good” looks like when you go from localization → mechanism → validation criteria. citeturn4search0turn4search5turn4search13  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["activation patching heatmap transformer interpretability","causal tracing ROME plot transformers","mechanistic interpretability circuit diagram IOI GPT-2"],"num_per_query":1}

The key lesson for your pivot:  
If you want one *high‑impact* result fast, you should aim for **a small, validated causal claim** rather than another broad correlational benchmark.

## The highest‑value experiments to run in your remaining sprint

Below is a blunt prioritization (strong opinions included, as requested). Each item is framed as: motivation → expected outcomes → decision value → cheapest viable implementation.

### Make one causal claim on the one dataset that isn’t trivially surface‑detectable
**Goal:** Convert your best surviving representational result (Llama synonym detection) into a **causal localization result**: “these layers/heads/MLPs are causally mediating synonym‑stego detectability.”

**Why this is the most promising**
- It aligns with the mechanistic‑interp “gold standard” workflow: decodability → intervention tests → localized mechanism. citeturn1search5turn2search3turn4search0  
- It directly answers the research question in your outline (“where is it represented; is there a direction; can we steer”).  
- It’s far more publishable than “we re‑discovered +++ detectors,” because it’s harder to dismiss with a baseline.

**Cheapest version that still counts**
- Pick a simple metric: probe logit (or AUC proxy) for “is_stego” on the synonym subset.
- Do **layerwise residual stream patching or ablation** first (coarse scan).
- Only then do head‑level patching/ablation in the few layers that matter (fine scan).  
TransformerLens makes patching/ablation workflows relatively straightforward, and it’s designed for this kind of work. citeturn1search10turn1search0  

**Expected outcomes**
- If you see a small number of layers/components drive most of the effect, that’s a strong mechanistic footprint.
- If the effect is diffuse, you still learn something: “representation exists but is distributed,” which guides next steps (e.g., sparse feature methods, or different probes).

**Decision impact**
- Positive: you pivot the paper around a mechanistic story.
- Negative/null: you stop chasing mechanistic localization in this sprint and position the work as “robust detection exists, but causal circuit is elusive under current tooling.”

### Replace “rare keyword present” endpoints with continuous metrics for steering tests
Your current Step‑3 style runs are expensive because they’re measuring a rare binary outcome. The literature on patching repeatedly emphasizes metric choice as a major source of confusion and false conclusions. citeturn1search5turn1search16  

**Better metrics (in approximate order of usefulness per effort)**
- **Probe score shift** under steering (continuous, low variance).
- **Target token log‑prob shift** (continuous) rather than “did the full response contain the word.”
- **KL divergence** can be useful, but it can also mostly measure “you changed the model,” not “you changed the causal variable,” unless carefully interpreted.

**Why this matters**
A steering method can have a real internal effect while producing no change in a rare final keyword. If you only track rare keywords, you’re flying with the cockpit instruments taped over.

### If you must finish the Qwen GCM chain, do it as a controlled methods demo and stop
You can justify completing a Qwen suffix‑marker GCM chain **only** as:
- a template‑fixed end‑to‑end demonstration (regen heads → permutation null → steer), and
- a negative‑control‑rich example that shows you know how to avoid interpretability self‑delusion.

This is consistent with the broader patching/attribution literature: gradient approximations are fast but can mislead; you need nulls and validation, and you should expect some “high variance component” failure modes. citeturn1search6turn1search16turn1search5  

**Concrete advice for the remaining sprint**
- **Downsample the expensive validation**. Don’t run 1000 pairs if the first 100 already show essentially zero effect on a continuous metric.
- **Add an explicit permutation null** (label shuffle). If your attribution survives label shuffle, it’s not finding your concept—it’s finding your code path.  
This is the sort of control that makes reviewers relax their shoulders.

### Turn your “marker/keyword sensitivity under RLHF” into a crisp, testable story
Your internal results strongly suggest that “compliance” depends on superficial form (marker choice, position) and keyword identity. That’s a legitimate behavioral contribution, and it fits the broader safety framing: systems optimized to be helpful/safe can nevertheless exhibit brittle pattern‑based suppression.  

A natural bridge to the refusal‑direction literature is: alignment often corresponds to relatively low‑dimensional control signals in the residual stream, and prompt patterns can modulate those signals. citeturn0search8turn1search5  

**Hackathon‑sized extension**
- Use a small set of marker/keyword variants and measure:
  - refusal rate,
  - probability of target token,
  - and (if feasible) projection onto a “refusal direction” style basis (even if approximate). citeturn0search8  
This can become: “covert‑channel compliance is gated by alignment heuristics that are sensitive to superficial prompt signatures.”

Even if you don’t fully mechanize it, it’s a coherent narrative that links behavior to known alignment control phenomena.

## Longer‑term directions that are genuinely fruitful

If you want the project to graduate from “cool hackathon demo” to “research platform,” these are the forward‑looking bets with strong literature alignment.

### Separate “stego presence” from “stego intent”
A recurring theme in safety‑motivated steganography work is that **intent** (trying to hide) is different from **presence** (a hidden code exists), and both differ from **successful covert communication**. citeturn3search2turn5search11  

Your Qwen suffix marker datasets mostly test “presence of a literal marker.” The next generation of your dataset should target:
- same surface distribution,
- different latent “intent to hide,”
- and ideally decoupled from trivial tokenization artifacts.

That’s how you get datasets where “TF‑IDF = 1.0” doesn’t instantly end the conversation.

### Move from explicit prompting to optimization‑induced or multi‑agent emergence
The most worrying steganography results in the recent literature involve **emergence under optimization pressure** and **agent interaction**, not just “please do steganography.” citeturn5search6turn5search10turn5search11  

If your long‑term roadmap includes “implicit instruction steganography” and “training‑induced steganography,” you’re aligned with where the field is actually moving.

### Treat steganography as a security problem: carrier models, trojans, and exfiltration
Threat‑model work like TrojanStego (fine‑tuned covert leakage) and “hidden text in weights” (UTF/UTFC) shows that the attack surface can be:  
- *model outputs as covert channels* and/or  
- *model weights as carriers of hidden payloads*. citeturn5search1turn3search0  

A strong future direction is to build a unified eval harness:
- prompt‑based stego,
- optimization‑induced stego,
- trojaned/fine‑tuned stego,
- and then ask whether one monitoring approach generalizes.

## Applying your decision rubric to your current “should we keep running this?” questions

This section answers your checklist directly, but in the order that tends to actually change decisions.

### Motivation and fit to your research question
If the research question is “do models represent steganography internally, and can we read/steer it,” then:
- **GCM on suffix markers fits only as a methods validation** (because it’s about surface feature processing).
- **Causal tests on the non‑surface synonym signal fit as the mainline** (because they test genuine internal representation and its use). citeturn2search3turn1search5  

### Have you derisked enough already
For the suffix‑marker branch: yes. You’ve already learned the critical thing (“it’s a surface artifact”), and additional runs mostly derisk *implementation*, not *the scientific claim*.  

For the synonym branch (Llama): no—you have a real claim, but it’s still correlational until you do an intervention test.

### Expected result and usefulness
For your current GCM Step‑3 style long validations: your own notes suggest the likely result is null or ambiguous (“noise on surface artifacts”). A null here probably doesn’t change strategy—so it’s low decision value.

A causal localization attempt on the non‑surface signal has asymmetric value:
- If it works, you get a headline.
- If it fails, you learn quickly where the bottleneck is (distributed representation, metric mismatch, patching confounds), which still guides future work. citeturn1search5turn1search16  

### One model and one dataset first
Strong yes—**especially** for causal work. The interpretability literature is full of “changed 4 variables at once and learned nothing” tragedies (the saddest genre). Start with:
- one model,
- one dataset slice,
- one metric,
- one intervention type. citeturn1search5turn2search2  

### Paper value near deadline
Near deadline, the highest ROI is:
- **one clean, defensible mechanistic result**, or
- **one clean behavioral result with a crisp interpretation and controls**.

A long run that confirms “still noisy” is rarely paper‑moving unless it’s required for a methods section and already framed that way.

### Are you changing too many variables at once
If your GCM/steering evaluation changes: head list (template fixes), dataset, metric, and sample size all at once, you won’t know what mattered. The cheapest simplification is:
- freeze everything,
- change one degree of freedom,
- add a null control,
- stop early when the effect is clearly too small to matter.

## A concrete 18‑hour sprint plan that follows the Golden Rule

This is one reasonable ordering if your goal is “maximum publishable insight per hour,” not “maximum GPU utilization per hour.”

**Stage one: lock the one real claim into one causal test** (high value, medium risk)
- Choose the Llama synonym setup (the non‑surface win).
- Do coarse layerwise patching/ablation with a continuous metric (probe score / log‑prob). citeturn1search5turn1search0  
- If you identify a small set of layers, do targeted head/MLP tests there.

**Stage two: produce a “methods credibility” artifact if you need it**
- Finish a minimal Qwen GCM chain only if you need an end‑to‑end demonstration.
- Use permutation nulls and negative controls so it reads as rigorous, not wishful thinking. citeturn4search4turn1search16  

**Stage three: write down the story you can defend**
- Steganography capability context + why it matters for AI safety / covert channels. citeturn5search11turn3search2turn5search1  
- Your key empirical finding(s) + controls + what you ruled out.
- A tight “limitations and next steps” that points to the next dataset generation you actually want (intent vs presence; optimization‑induced emergence). citeturn5search6turn3search1  

If you do only one thing: **trade one long, low‑signal validation run for one causal intervention test on the one non‑trivial signal.** That swap is the difference between “we ran lots of experiments” and “we learned something that survives contact with a reviewer.”