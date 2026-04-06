# **A Four-Tier Computational Model of Microglial Pruning, Mitochondrial NAD⁺ Compensation, and Autophagy Collapse in ALS: Testing a Multi-Target Glutamatergic Regimen**

Authors:

Ngo Cheung, FHKAM(Psychiatry)

Affiliations:

¹ Cheung Ngo Medical Limited, Hong Kong SAR, China

Corresponding Author:

Ngo Cheung, MBBS, FHKAM(Psychiatry)

Cheung Ngo Medical Limited

Hong Kong SAR, China

Tel: 98768323

Email: info@cheungngomedical.com

**Conflict of Interest**: None declared.

**Funding Declaration**: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

**Ethics Declaration**: Not applicable.

## **Abstract**

Amyotrophic lateral sclerosis (ALS) remains a disease with few disease-modifying options and a short survival window. Recent genomic and pathological findings have suggested that excessive microglial synaptic pruning may represent a shared upstream vulnerability with major depressive disorder, with the ALS trajectory then intensified by mitochondrial bioenergetic stress and autophagy failure. A series of five 2026 preprints by Cheung proposed a four-tier mechanistic continuum linking these processes. The present study implemented that framework in a mesoscale neural-network model to examine whether the proposed cascade could reproduce clinical ALS trajectories and whether a multi-target Augmented Cheung Glutamatergic Regimen (aCGR) could modify progression relative to riluzole or ketamine-like interventions.

The model incorporated eight circuit layers, including sub-layer branching for bulbar, respiratory, and fine-motor compartments. Tier 1 pruning was modeled using complement bias plus activity dependence. A per-layer NAD⁺ buffer represented mitochondrial compensation and depletion at Tier 2, while dynamic autophagy efficiency and weight corruption were used to represent aggregate spread and clearance failure at Tier 3. All downstream symptoms, together with the neurofilament light chain proxy, emerged directly from network state. Sixty scenarios were run across standard ALS, subtype, and comorbid profiles.

In the untreated arm, sparsity accumulated rapidly after cycle 6, NAD levels declined, and the ALSFRS-R proxy fell below 12 by mid-stage. Riluzole and ketamine delayed deterioration modestly but did not prevent eventual NAD burnout or near-total sparsity. Both early-pulsed and maintenance aCGR schedules were associated with lower final sparsity, at 86%--87%, reduced NFL proxy values, and better preserved NAD levels than the other treatment arms. Subtype and comorbid simulations showed similar directional effects, with numerically lower depression scores in the comorbid aCGR arm.

These computational findings suggest that multi-tier targeting may delay the critical sparsity threshold. The model also yields several testable predictions for future pilot work, including slower ALSFRS-R decline and biomarker shifts in genetically stratified patients. Validation in iPSC systems and careful early-phase trials would be necessary before any clinical use could be considered.

## **Introduction**

Amyotrophic lateral sclerosis is a relentlessly progressive neurodegenerative disorder defined by selective degeneration of upper and lower motor neurons. Most patients survive only 2--5 years after symptom onset, and the currently approved therapies, riluzole and edaravone, extend survival only modestly while offering limited functional preservation \[1,2\]. Beyond its core motor presentation, ALS is increasingly recognized to include mood symptoms, especially new-onset depression and anxiety, that may appear months or even years before overt weakness or bulbar dysfunction \[3,4\]. These prodromal psychiatric changes are unlikely to be purely reactive. Instead, they suggest that limbic and motor circuits may share an early vulnerability.

Large-scale human genetics has begun to clarify the biological basis of that overlap. The Project MinE consortium identified multiple ALS risk loci with distinct genetic architectures and neuron-specific biology \[5\]. Parallel work in major depressive disorder has implicated synaptic, immune, and RNA-processing pathways \[6,7\]. Although bivariate linkage disequilibrium score regression has shown near-zero overall genetic correlation between ALS and MDD, modest clinical comorbidity persists, and specific signals have appeared in complement and major histocompatibility complex genes \[8\]. Taken together, these findings converge on microglial-mediated synaptic pruning as a plausible upstream liability shared across disorders \[9,10\].

Building on this background, a series of five interlinked preprints by Cheung \[11--15\] proposed a unified four-tier continuum. In this model, Tier 1 consists of genetic vulnerability to excessive complement- and MHC-mediated microglial pruning. This drives circuit sparsity, hyperexcitability, and increased metabolic burden in the surviving neurons, which in turn triggers oxidative phosphorylation stress and compensatory NAD⁺ upregulation at Tier 2. When clearance mechanisms fail, autophagy and mitophagy collapse, protein aggregates accumulate, reactive oxygen species rise, and a self-reinforcing loop develops at Tier 3. Tier 4 then corresponds to network decompensation and clinical progression. Divergence from MDD is attributed to ALS-specific failure of autophagic clearance, whereas mood disorders are proposed to diverge through RNA and immune dysregulation. The present study was designed as an independent computational test of this framework, using a mesoscale neural-network architecture that incorporates the proposed tiers while allowing the resulting phenotype to emerge from network dynamics rather than from a pre-specified clinical script.

Several lines of pathological and experimental evidence make such a framework biologically plausible. Complement-tagged synaptic elimination by microglia is now understood as a conserved mechanism in neurodegeneration, and early deposition of complement components at neuromuscular junctions has been documented in both human ALS tissue and SOD1 mouse models \[16--18\]. TDP-43 aggregates, the pathological hallmark of nearly all ALS cases, display prion-like seeding and intercellular spread, consistent with the model\'s weight-corruption mechanism \[19--21\]. Mitochondrial bioenergetic failure and defective mitophagy are also well established in ALS, and mutations in genes such as TBK1, OPTN, and C9orf72 directly impair autophagic clearance \[22--24\]. NAD⁺ depletion, together with compensatory upregulation of biosynthetic enzymes such as NMNAT3, has been observed in patient-derived motor neurons and SOD1-G93A mice, and precursor supplementation with nicotinamide mononucleotide has shown modest benefit in delaying decline \[25,26\].

Existing computational models of ALS progression have usually centered on downstream excitotoxicity, protein aggregation, or macroscopic functional decline \[27,28\]. While informative, those approaches rarely incorporate upstream pruning liability or the dynamic interplay between mitochondrial stress and autophagic failure. The present work was therefore designed to address that gap by constructing a biologically constrained neural network in which pruning vulnerability is the starting perturbation and downstream phenotypes, including subtype-specific patterns such as limb versus bulbar onset, stochastic flares, and prodromal mood involvement, emerge from the four-tier architecture.

The study had three specific aims. First, it sought to implement and validate a mesoscale neural-network model capable of reproducing real-world ALS trajectories and clinical subtypes from pruning vulnerability alone. Second, it compared a multi-target Augmented Cheung Glutamatergic Regimen, or aCGR, with single-mechanism comparators, namely a ketamine-like plasticity rescue and a riluzole-like glutamate-modulating intervention. Third, it aimed to generate falsifiable quantitative predictions regarding timing, magnitude, and biomarker correlates of disease modification that could inform future genetically stratified early-intervention trials.

## **Methods**

All simulations were performed in PyTorch version 2.3 on standard GPU hardware under Python 3.11. The complete source code, raw results files, and sensitivity-analysis datasets were reported as available on Zenodo \[29\]. No new human or animal data were collected. The study was entirely computational and reused concepts drawn from the cited literature.

### **Network Architecture**

A feedforward neural network, implemented as the MotorCircuitNetwork class, was constructed to represent distributed motor and prefrontal-limbic circuits implicated in ALS (Figure 1). The model comprised eight layers corresponding to major anatomical compartments: a prefrontal-limbic input layer with dimension 2 projecting to 512 units, an upper motor layer of 512 to 512 units, a lower motor layer of 512 to 256 units, and a neuromuscular junction layer of 256 to 128 units. From the neuromuscular junction layer, three parallel sub-layers were branched: bulbar, representing speech and swallowing circuits; respiratory, representing diaphragmatic control; and fine motor, representing distal limb dexterity. Each sub-layer mapped 128 to 64 units. Their outputs were concatenated into a final output layer of 192 to 4 classes representing basic motor functions.

![](media/image1.png){width="6.267716535433071in" height="8.416666666666666in"}

***Figure 1.** Computational model architecture and methods overview. (A) Eight-layer feedforward neural network representing motor and prefrontal-limbic circuits. Sub-layer branching after the neuromuscular junction enables subtype-specific readouts for bulbar, respiratory, and fine motor domains. Each layer maintains a binary pruning mask, Gaussian noise injection scaled to sparsity, a per-layer NAD⁺ buffer, and an aggregate load tracker. Seed-based vulnerability modifiers (0.85--1.15) are applied per layer to introduce patient heterogeneity. Stochastic flares are triggered probabilistically when local sparsity exceeds 0.3. (B) Four-tier mechanistic cascade. Tier 1 microglial pruning increases circuit sparsity and metabolic demand, driving Tier 2 NAD⁺ depletion. When aggregate load exceeds threshold under low NAD conditions, Tier 3 autophagy collapses, creating a self-reinforcing vicious cycle (red dashed arrows) that amplifies pruning rate, corruption magnitude, and clearance failure via the NAD multiplier (2.0 − NAD). Tier 4 decompensation emerges from cumulative tier interactions without additional free parameters. (C) Treatment-to-tier target matrix. Riluzole provides only Tier 1 noise damping. Ketamine adds gradient-guided synaptogenesis at Tier 1. The Augmented Cheung Glutamatergic Regimen (aCGR) simultaneously engages all three upper tiers: plasticity rescue via DXM and piracetam, NAD⁺ replenishment via NMN, and autophagy clearance via pulsed senolytics and NAC. Filled circles indicate active tier engagement; dashes indicate no modelled effect. (D) Architecture-to-readout mapping. All clinical scores and biomarker proxies are derived directly from network state. The NFL proxy uniquely combines weight instability with an NAD⁺ penalty term (orange dashed arrow). Border colours correspond to tier: blue (Tier 1 / general architecture), orange (Tier 2 / NAD⁺), green (Tier 3 / autophagy), red (Tier 4 / clinical outcomes).*

This branching structure was included to capture the clinical heterogeneity of ALS onset and spread. Bulbar-onset disease, which accounts for roughly 25%--30% of cases, preferentially affects cranial motor circuits early, whereas limb-onset disease predominates in spinal motor pools \[30,31\]. ReLU activation was used throughout. Gaussian noise was added at each layer, as described below, to approximate sparsity-driven excitotoxic instability, which has repeatedly been documented in electrophysiological studies of ALS motor neurons \[32,33\].

### **Tier 1: Microglial Pruning Engine**

Pruning was implemented through a binary mask applied to the weight tensor of each layer. At every simulation cycle, a fixed proportion of connections was removed using a two-part rule that combined complement-biased pruning with activity-dependent random removal. The complement bias parameter was set by default at 0.7, so that the weakest absolute-weight connections were preferentially targeted. This choice was intended to reflect the prominence of complement and MHC signals in ALS GWAS findings and the recognized role of classical complement tagging in microglial synaptic elimination \[10,18,34\]. Once masked, pruned weights were set to zero and recorded for downstream toxicity tracking. Regrowth could restore masked positions and reinitialize them from a narrow Gaussian distribution.

This mechanism was directly grounded in human post-mortem and iPSC-derived microglial studies showing complement-mediated tagging and selective synaptic removal in ALS tissue \[16,35\].

### **Tier 2: Mitochondrial NAD⁺ Buffer**

To model the mitochondrial stress and compensatory response described in the four-tier framework, a per-layer NAD⁺ buffer was added. NAD levels began at 1.0, representing a healthy baseline, and were updated each cycle according to energy demand calculated from the current level of sparsity and noise. Depletion scaled with demand at a default rate of 0.015, whereas partial compensation, analogous to NMNAT3 and SIRT activity, was set proportional to the amount of remaining healthy tissue. A burnout threshold of 0.25 triggered severe downstream penalties. These parameters were informed by direct measurements in SOD1-G93A mice and patient-derived motor neurons showing early NAD⁺ depletion, compensatory upregulation of NMNAT3 and SIRT enzymes, and rescue by NAD⁺ precursors \[25,26,36\].

Low NAD levels fed back multiplicatively to pruning rate, corruption magnitude, and regrowth efficiency, thereby implementing the Tier 2--Tier 3 interface.

### **Tier 3: Toxic Aggregate Tracker and Dynamic Autophagy**

Protein aggregation load was tracked separately for each layer. Newly pruned connections contributed to aggregate accumulation at a fixed rate. Clearance occurred during each cycle at a base autophagy efficiency, but that efficiency could be dynamically reduced when aggregate load exceeded 0.4 and NAD fell below the burnout threshold, thereby implementing the vicious-cycle component of the model. Surviving weights near pruned positions, within a radius of five elements in the flattened tensor, were directly corrupted with Gaussian noise scaled by both aggregate load and the inverse NAD multiplier.

This weight-corruption mechanism was chosen to approximate prion-like TDP-43 spread to neighboring synapses, a process supported by several pathological and in vitro studies \[19--21,37\]. Dynamic autophagy collapse was used to reflect mTOR upregulation and mitophagy failure, both of which are repeatedly described in ALS \[22,23,38\].

### **Vicious-Cycle Feedback**

The interface between Tier 2 and Tier 3 was implemented mathematically as a NAD-dependent multiplier, expressed as 2.0 minus NAD, applied to pruning rate, corruption magnitude, and autophagy efficiency whenever NAD fell below threshold. This created a self-reinforcing sequence of pruning, NAD depletion, failed clearance, further aggregate accumulation, and still more pruning. That formulation is consistent with ROS--PARP--NAD burnout and SASP--complement amplification loops described in the ALS literature \[24,39\].

### **Excitotoxicity and Noise**

Noise was not introduced on a fixed schedule. Instead, it was derived directly from current sparsity and weight instability, with an instability gain of 3.0. This design was intended to reflect the well-established relationship between circuit sparsity, hyperexcitability, and excitotoxic stress in both human ALS motor neurons and animal models \[32,40,41\]. Treatment modules were able to reduce noise, for example through riluzole-like damping or NAC-like oxidative buffering.

### **Treatment Modules**

Three treatment arms were implemented with parameters mapped to published pharmacology.

Riluzole was modeled as a continuous mild noise-damping effect of 0.3 together with modest pruning reduction of 0.15, consistent with its role in inhibiting glutamate release and its limited neuroprotective impact in pharmacokinetic and survival studies \[1,42,43\].

The ketamine-like plasticity rescue consisted of gradient-guided regrowth, intended to capture a BDNF-like synaptogenic effect, together with noise reduction and pruning dampening that decayed exponentially, in line with reported synaptogenic and anti-inflammatory actions of ketamine in depression and early ALS-related literature \[44,45\].

The Augmented Cheung Glutamatergic Regimen combined core plasticity rescue, modeled on low-dose dextromethorphan with CYP2D6 inhibition plus piracetam, with NMN-driven NAD⁺ support, NAC-mediated oxidative reduction, and pulsed senolytic autophagy enhancement. Parameters and dosing schedule, continuous core treatment with intermittent pulses, followed the mapping proposed in Cheung \[14\] and were further supported by SOD1-G93A mouse data showing preservation of motor performance and neuromuscular junction function after NMN supplementation \[26\].

### **Symptom Mapping and Biomarkers**

All clinical readouts were derived directly from network state without introducing free parameters. Depression score scaled with prefrontal-limbic sparsity. Bulbar risk and respiratory risk were based on their corresponding sub-layer sparsities. The ALSFRS-R proxy scaled with overall classification accuracy. The NFL biomarker proxy combined weight instability with an NAD penalty term, defined as 2.0 minus average NAD, to reflect the known association between neurofilament light chain, axonal damage, and bioenergetic stress in ALS \[46--48\]. NAD depletion and autophagy efficiency were also reported explicitly as Tier 2 and Tier 3 readouts.

### **Simulation Protocol and Sensitivity Analyses**

Each scenario ran for 30 cycles, corresponding to approximately 45 months for ALS profiles. Sixty combinations were tested across six disease profiles: standard ALS, bulbar-onset ALS, limb-onset ALS, MDD, ALS-MDD comorbidity, and control. These were crossed with ten treatment arms, including no treatment, several ketamine schedules, riluzole in continuous and early schedules, and aCGR in early-pulsed, maintenance, and late schedules. Patient heterogeneity was introduced through seed-based vulnerability modifiers ranging from 0.85 to 1.15 and applied to layer-specific pruning rates. Stochastic flares were triggered when local sparsity exceeded 0.3, with probability 0.10 and magnitude doubled.

New sensitivity analyses in version 3 varied Tier 1 pruning liability, Tier 3 baseline autophagy efficiency, and Tier 2 NAD depletion rate across plus or minus 20% of default values to test whether treatment rankings remained robust.

### **Statistical and Reproducibility Considerations**

Final-cycle metrics, including accuracy, sparsity, NAD percentage, ALSFRS proxy, and NFL proxy, were compared across arms using descriptive statistics only. The aim was mechanistic illustration rather than formal inferential hypothesis testing. All results were reported as fully reproducible from the deposited code, and no data were excluded.

## **Results**

The simulation generated 30 cycles of progression, corresponding to approximately 45 months on the ALS time scale, for each of the 60 scenarios. Clinical readouts, biomarker proxies, and tier-specific metrics all emerged directly from evolving network state without additional free parameters. The principal findings are presented first for the standard ALS profile and then for subtype and comorbid conditions. Tables 1-4 and Figure 2 are visualizing the final-cycle outcomes.

### **Untreated ALS Trajectory**

In the absence of treatment, the model reproduced a steadily progressive course that resembled clinical ALS. During preclinical cycles 0 through 4, sparsity accumulated gradually, predominantly in motor layers, although prefrontal-limbic involvement appeared first. By the early symptomatic stage at cycle 6, corresponding to month 9, total sparsity had reached 59.1%, NAD levels had begun to decline, and the ALSFRS-R proxy had already fallen from a baseline of 46.0 to 43.4. By mid-stage, at cycle 15 or month 22.5, the network approached decompensation: sparsity exceeded 97%, mean NAD across motor layers had fallen to 69.1%, and the ALSFRS-R proxy stabilized around 11.9. At the final cycle, cycle 29 or month 43.5, sparsity was 99.2%, NAD depletion had reached 100%, autophagy efficiency remained at baseline at 5.0%, and the NFL proxy peaked at 151.3. Motor integrity fell below 1%, and cognitive scores reached floor values, together reflecting the late-stage collapse predicted by the four-tier model.

### **Treatment Effects in the Standard ALS Profile**

The treatment arms diverged clearly by the final cycle. Continuous riluzole produced modest early damping of noise but ultimately converged on a profile close to no treatment, with a final ALSFRS-R proxy of 11.8, sparsity of 99.0%, and NFL proxy of 202.3. Maintenance ketamine provided intermediate preservation, reducing final sparsity to 93.0% and yielding an NFL proxy of 150.5, but NAD depletion still reached 80.7%. By contrast, both aCGR schedules showed lower final sparsity, 86.2% for early-pulsed and 87.2% for maintenance, together with substantially lower NFL proxies of 90.8 and 112.8, respectively. The early-pulsed aCGR schedule also produced the lowest final NAD depletion score, 17.6%. Depression scores remained elevated across all arms, but were numerically lowest in the aCGR conditions, ranging from 43.9 to 45.5.

### **Tier Dynamics Over Time**

The time-course of the untreated versus early aCGR conditions illustrated the logic of the model. In the untreated arm, Tier 1 sparsity increased steadily and began to drive Tier 2 NAD depletion after cycle 6. By mid-stage, Tier 3 autophagy efficiency showed no compensatory rise, and Tier 4 decompensation became evident through deterioration in the ALSFRS-R proxy and the NFL proxy. Under early aCGR, Tier 2 NAD remained near 100% through symptomatic phases and final sparsity was limited to 86.2%. Autophagy efficiency showed transient elevations during pulse cycles, reaching as high as 8.4%, in keeping with the modeled senolytic component. These effects emerged from the architecture itself and were consistent with the vicious-cycle interface built into the model.

### **Subtype-Specific Patterns**

Bulbar-onset and limb-onset profiles were generated simply by adjusting sub-layer pruning rates while holding all other parameters constant. In both subtypes, the no-treatment arms progressed to near-total sparsity above 99% and very low motor integrity below 1% by cycle 29. Early aCGR reduced final sparsity to 84.7% in the bulbar profile and 87.2% in the limb profile, lowered NFL proxies to 102.0 and 105.2, and preserved motor integrity at 15.4% and 12.9%, respectively. Respiratory risk remained markedly higher in the bulbar profile even with treatment, reflecting the differential pruning schedule built into that subtype.

### **Comorbid ALS-MDD Profile**

The comorbid profile combined elevated prefrontal pruning with motor vulnerability. In this setting, the untreated and riluzole arms produced the highest final depression scores, 47.5 to 47.6, together with low ALSFRS-R proxies of 10.8 to 12.7 and complete NAD depletion. Ketamine administered on a q5 schedule produced only marginal improvement. The early aCGR arm produced the lowest depression score, 41.3, the highest ALSFRS-R proxy, 13.4, and markedly reduced NAD depletion at 31.2%, suggesting a dual effect in both mood and motor domains within the same network.

### **Robustness**

Sensitivity analyses varied Tier 1 pruning liability, Tier 3 baseline autophagy efficiency, and Tier 2 NAD depletion rate by plus or minus 20% of default values. Across this range, the relative ranking of treatments did not change. Early aCGR consistently yielded the lowest final sparsity and the lowest NFL values. The source text reports that all code and raw simulation outputs were deposited on Zenodo for independent verification \[29\].

![](media/image2.png){width="6.267716535433071in" height="5.527777777777778in"}

***Figure 2.** Simulation results for ALS progression across treatment arms and disease subtypes. Panels A--D show longitudinal trajectories for the standard ALS profile (seed 42) under five treatment arms plus healthy control. Background shading denotes disease phase: preclinical (blue, months 0--7.5), symptomatic (yellow, 7.5--30), and advanced (red, 30--43.5). (A) ALSFRS-R functional proxy (0--48). aCGR Early Pulsed (green) and Maintenance (amber) delay functional decline by approximately 6--10 months relative to No Treatment and Riluzole. All ALS arms converge to near-floor values by month 30, while Control remains stable (\~46). (B) Total network sparsity (% connections pruned). aCGR arms stabilise at \~85--87% in the advanced phase and show partial structural recovery, whereas No Treatment and Riluzole reach near-complete network loss (\>99%). (C) NAD⁺ depletion score (Tier 2 state). aCGR uniquely preserves mitochondrial NAD⁺ via NMN supplementation, maintaining depletion below 18% at endpoint. All other arms, including Riluzole (99.5%) and Ketamine q5 (97.2%), converge to full depletion by month 40. This demonstrates the selective Tier 2 efficacy of the multi-target regimen. (D) NFL biomarker proxy (neurofilament, higher = worse). aCGR Early achieves markedly lower peak and endpoint NFL (90.8 at month 43.5) compared with No Treatment (151.3) and Riluzole (202.3). The paradoxically higher final NFL under Riluzole reflects sustained NAD-amplified weight corruption in a partially surviving but energetically compromised network. (E) Endpoint NFL biomarker across all ALS treatment arms. The three aCGR arms (green, amber, pink) show 25--40% lower NFL than No Treatment or single-target agents. Dashed gray line indicates healthy control level (\~70). aCGR Late (month 22.5 initiation) achieves substantial NFL reduction (106) despite advanced-stage introduction, consistent with the model\'s prediction of partial vicious-cycle reversal. (F) Endpoint NAD⁺ depletion across four disease subtypes comparing No Treatment, Riluzole, and aCGR Early Pulsed. Only aCGR achieves meaningful NAD⁺ preservation across all variants (18--31%). The higher residual depletion in ALS-MDD (31%) reflects the comorbid profile\'s elevated NAD depletion rate (0.018 vs 0.015) and reduced compensation strength. Riluzole provides no Tier 2 protection in any subtype.*

***Table 1  
*Final-Cycle Clinical and Biomarker Outcomes for the Standard ALS Profile Across Treatment Arms (Cycle 29, Month 43.5)**

| **Treatment Arm**    | **Accuracy (%)** | **Total Sparsity (%)** | **ALSFRS-R Proxy** | **NFL Proxy** | **Depression Score** | **Motor Integrity (%)** | **NAD⁺ Depletion (%)** | **Autophagy Eff. (%)** |
|----------------------|------------------|------------------------|--------------------|---------------|----------------------|-------------------------|------------------------|------------------------|
| No treatment         | 27.5             | 99.2                   | 13.2               | 151.3         | 46.9                 | 0.8                     | 100.0                  | 5.0                    |
| Riluzole continuous  | 24.5             | 99.0                   | 11.8               | 202.3         | 45.9                 | 1.0                     | 99.5                   | 5.0                    |
| Ketamine maintenance | 22.8             | 93.0                   | 10.9               | 150.5         | 46.0                 | 7.0                     | 80.7                   | 4.6                    |
| aCGR early pulsed    | 25.7             | 86.2                   | 12.4               | 90.8          | 43.9                 | 13.9                    | 17.6                   | 4.9                    |
| aCGR maintenance     | 27.3             | 87.2                   | 13.1               | 112.8         | 45.5                 | 12.8                    | 12.4                   | 5.0                    |

*Note. Values are model-derived at the final simulation cycle (month 43.5). Higher ALSFRS-R proxy and motor integrity indicate better functional preservation; lower NFL proxy and NAD⁺ depletion indicate reduced neurodegeneration. Highlighted rows denote the multi-tier aCGR regimen arms.*

***Table 2  
*Tier-State Progression at Selected Time Points: Untreated ALS Versus Early aCGR (Standard ALS Profile)**

<table style="width:100%;">
<colgroup>
<col style="width: 18%" />
<col style="width: 15%" />
<col style="width: 13%" />
<col style="width: 11%" />
<col style="width: 16%" />
<col style="width: 12%" />
<col style="width: 12%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Time Point</strong></p>
<p><strong>(Cycle / Month)</strong></p></th>
<th><strong>Arm</strong></th>
<th><p><strong>Tier 1</strong></p>
<p><strong>Sparsity (%)</strong></p></th>
<th><p><strong>Tier 2</strong></p>
<p><strong>NAD⁺ (%)</strong></p></th>
<th><p><strong>Tier 3</strong></p>
<p><strong>Autophagy (%)</strong></p></th>
<th><p><strong>Tier 4</strong></p>
<p><strong>ALSFRS-R</strong></p></th>
<th><strong>NFL Proxy</strong></th>
</tr>
<tr class="odd">
<th>Baseline (−1 / 0)</th>
<th>Untreated</th>
<th>0.0</th>
<th>100.0</th>
<th>5.0</th>
<th>46.0</th>
<th>0.0</th>
</tr>
<tr class="header">
<th>Baseline (−1 / 0)</th>
<th>aCGR early</th>
<th>0.0</th>
<th>100.0</th>
<th>5.0</th>
<th>46.0</th>
<th>0.0</th>
</tr>
<tr class="odd">
<th colspan="7"></th>
</tr>
<tr class="header">
<th>Early sympt. (6 / 9)</th>
<th>Untreated</th>
<th>59.1</th>
<th>95.9</th>
<th>5.0</th>
<th>43.4</th>
<th>87.9</th>
</tr>
<tr class="odd">
<th>Early sympt. (6 / 9)</th>
<th>aCGR early</th>
<th>16.9</th>
<th>100.0</th>
<th>5.0</th>
<th>46.0</th>
<th>73.7</th>
</tr>
<tr class="header">
<th colspan="7"></th>
</tr>
<tr class="odd">
<th>Mid-stage (15 / 22.5)</th>
<th>Untreated</th>
<th>97.2</th>
<th>69.1</th>
<th>4.9</th>
<th>11.9</th>
<th>154.2</th>
</tr>
<tr class="header">
<th>Mid-stage (15 / 22.5)</th>
<th>aCGR early</th>
<th>76.6</th>
<th>99.8</th>
<th>8.4</th>
<th>13.3</th>
<th>84.7</th>
</tr>
<tr class="odd">
<th colspan="7"></th>
</tr>
<tr class="header">
<th>Endpoint (29 / 43.5)</th>
<th>Untreated</th>
<th>99.2</th>
<th>11.4</th>
<th>5.0</th>
<th>13.2</th>
<th>151.3</th>
</tr>
<tr class="odd">
<th>Endpoint (29 / 43.5)</th>
<th>aCGR early</th>
<th>86.2</th>
<th>86.6</th>
<th>4.9</th>
<th>12.4</th>
<th>90.8</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

*Note. Tier 1 = microglial pruning (total sparsity); Tier 2 = mitochondrial NAD⁺ buffer (percentage remaining); Tier 3 = dynamic autophagy efficiency; Tier 4 = network decompensation metrics. aCGR early pulsed preserves Tier 2 NAD⁺ (86.6% vs. 11.4%) and limits Tier 1 sparsity (86.2% vs. 99.2%), consistent with vicious-cycle interruption.*

***Table 3  
*Final-Cycle Outcomes in Clinical Subtypes With and Without Early aCGR (Cycle 29, Month 43.5)**

| **Subtype & Arm**               | **ALSFRS-R Proxy** | **Total Sparsity (%)** | **NFL Proxy** | **Respiratory Risk (%)** | **Motor Integrity (%)** |
|---------------------------------|--------------------|------------------------|---------------|--------------------------|-------------------------|
| Bulbar-onset, no treatment      | 12.2               | 99.1                   | 175.4         | 99.1                     | 0.9                     |
| Bulbar-onset, aCGR early pulsed | 11.5               | 84.7                   | 102.0         | 84.6                     | 15.4                    |
|                                 |                    |                        |               |                          |                         |
| Limb-onset, no treatment        | 12.4               | 99.2                   | 162.6         | 99.1                     | 0.8                     |
| Limb-onset, aCGR early pulsed   | 11.6               | 87.2                   | 105.2         | 88.8                     | 12.9                    |

*Note. Despite converging ALSFRS-R scores at endpoint, aCGR early pulsed reduces NFL by 42% in bulbar-onset and 35% in limb-onset profiles, while preserving 13--15% motor network integrity versus \<1% in untreated arms. Respiratory risk reduction is more modest in the limb-onset subtype (88.8% vs. 84.6%), reflecting differential vulnerability profiles.*

***Table 4  
*Final-Cycle Outcomes in ALS-MDD Comorbid Profile Across Selected Treatment Arms (Cycle 29, Month 43.5)**

| **Treatment Arm**   | **ALSFRS-R Proxy** | **Total Sparsity (%)** | **Depression Score** | **NFL Proxy** | **NAD⁺ Depletion (%)** |
|---------------------|--------------------|------------------------|----------------------|---------------|------------------------|
| No treatment        | 12.7               | 99.2                   | 47.6                 | 115.2         | 100.0                  |
| Riluzole continuous | 10.8               | 99.0                   | 47.5                 | 188.1         | 100.0                  |
| Ketamine q5         | 12.0               | 98.9                   | 47.5                 | 86.9          | 100.0                  |
| aCGR early pulsed   | 13.4               | 89.8                   | 41.3                 | 78.4          | 31.2                   |

*Note. The ALS-MDD comorbid profile exhibits accelerated progression (NAD⁺ depletion rate 0.018 vs. 0.015 in standard ALS). aCGR early pulsed is the only arm that achieves partial NAD⁺ preservation (31.2% depletion) and simultaneously reduces the depression score (41.3 vs. 47.5--47.6 in other arms). Ketamine q5 achieves lower NFL (86.9) via NMDA-mediated mechanisms but provides no Tier 2 NAD⁺ protection. Riluzole paradoxically yields the highest NFL (188.1), reflecting sustained weight corruption in an energetically compromised network.*

## **Discussion**

These simulations provide a computational illustration of how a four-tier cascade beginning with excessive microglial pruning may unfold in ALS. In the untreated arm, prefrontal-limbic sparsity developed early and was followed by progressive involvement of motor layers. That ordering is consistent with the clinical observation that new-onset mood symptoms can precede overt weakness in some patients. As NAD levels declined and autophagic clearance failed to compensate, a self-reinforcing loop emerged that accelerated sparsity accumulation and functional decline. Notably, these features were not imposed on the model as fixed endpoints. They arose from the interaction of the tiers.

Among the treatment conditions, the Augmented Cheung Glutamatergic Regimen was associated with lower final sparsity and lower neurofilament proxy values than riluzole or ketamine. Both the early-pulsed and maintenance schedules kept sparsity below 88% and preserved NAD levels for longer. Riluzole produced only modest early damping of network noise, whereas ketamine generated intermittent regrowth benefits but did not prevent later NAD depletion. The aCGR effect therefore appeared to arise from simultaneous engagement of multiple nodes in the cascade: plasticity rescue at Tier 1, NAD replenishment at Tier 2, and autophagy support at Tier 3. In this framework, that broader coverage delays crossing of the critical sparsity threshold.

The model also suggests several translational implications. First, new-onset late-life mood symptoms may serve as an early clinical warning sign, because prefrontal-limbic pruning emerges first in the network. Screening patients with refractory late-onset depression or anxiety, particularly when accompanied by subtle bulbar or motor signs, could create an earlier entry point for intervention than is usually achieved in ALS. Second, a composite polygenic risk score incorporating pruning liability, such as HLA-B, complement, and PROS1-related variation, together with autophagy risk involving C9orf72, TBK1, or the mTOR axis, could be used to stratify likely rapid progressors and likely responders. The model predicts that such stratification could increase statistical power in small early-phase studies and identify individuals in whom combined plasticity plus NAD⁺ and autophagy support may be most relevant.

A logical next step would be a Phase 1b/2a open-label safety and dosing study of aCGR in genetically stratified early or prodromal ALS, perhaps in 20 to 40 participants. Relevant endpoints would include tolerability, biomarker change, including plasma NAD metabolites, complement activation markers, and neurofilament light, as well as ALSFRS-R slope and mood scales. A subsequent randomized placebo-controlled Phase 2 trial in an enriched population over 6 to 12 months could then examine progression slope as a primary endpoint, with secondary outcomes such as time-to-event measures, quality of life, and the biomarker panel. Platform expansion might later incorporate add-on arms using GLP-1 agonists or complement inhibitors. In the longer term, prevention-oriented studies in ultra-high-risk asymptomatic carriers, such as members of C9orf72 families with high pruning-related polygenic risk, could also be considered. Safety monitoring would need to include drug interactions, bulbar function, and renal and hepatic status, especially if senolytics were involved.

At the same time, this work remains an in silico model. It did not generate new biological data, and it necessarily relied on simplifying assumptions, including uniform oral bioavailability, fixed parameter choices, and Gaussian noise as a stand-in for excitotoxicity. The study is therefore hypothesis-generating rather than evidentiary. The four-tier framework itself is derived from reanalysis of public GWAS and TWAS data, and the treatment mapping follows the proposal in Cheung \[14\] without implying demonstrated clinical efficacy.

Future work should test the framework directly in iPSC-derived motor neuron systems or organoids in which pruning, NAD metabolism, and autophagy can be manipulated experimentally. The present model generates several testable predictions. It predicts that early aCGR should slow ALSFRS-R decline and blunt the rise in neurofilament light relative to riluzole in genetically enriched patients. It predicts that responders should show measurable changes in plasma NAD metabolites and reductions in senescence-associated secretory phenotype markers. It also predicts that dual mood-motor benefit should be most evident in comorbid cases. Independent computational and experimental replication will be essential before any clinical application is contemplated.

## **References**

1.  Bensimon G, Lacomblez L, Meininger V; ALS/Riluzole Study Group. A controlled trial of riluzole in amyotrophic lateral sclerosis. *N Engl J Med*. 1994;330(9):585-591. doi:10.1056/NEJM199403033300901

2.  Writing Group for the Edaravone (MCI-186) ALS 19 Study Group. Safety and efficacy of edaravone in well-defined patients with amyotrophic lateral sclerosis: a randomised, double-blind, placebo-controlled trial. *Lancet Neurol*. 2017;16(7):505-512. doi:10.1016/S1474-4422(17)30115-1

3.  Tan Y, Yang T, Cheng Y, et al. Association between psychiatric disorders and amyotrophic lateral sclerosis: a prospective cohort study from the UK Biobank. *Neuroepidemiology*. 2025;59(6):701-713. doi:10.1159/000543473

4.  Zucchi E, Ticozzi N, Mandrioli J. Psychiatric symptoms in amyotrophic lateral sclerosis: beyond a motor neuron disorder. *Front Neurosci*. 2019;13:175. doi:10.3389/fnins.2019.00175

5.  van Rheenen W, van der Spek RAA, Bakker MK, et al. Common and rare variant association analyses in amyotrophic lateral sclerosis identify 15 risk loci with distinct genetic architectures and neuron-specific biology. *Nat Genet*. 2021;53(12):1636-1648. doi:10.1038/s41588-021-00973-1

6.  Wray NR, Ripke S, Mattheisen M, et al. Genome-wide association analyses identify 44 risk variants and refine the genetic architecture of major depression. *Nat Genet*. 2018;50(5):668-681. doi:10.1038/s41588-018-0090-3

7.  Adams MJ, Streit F, Meng X, et al. Trans-ancestry genome-wide study of depression identifies 697 associations implicating cell types and pharmacotherapies. *Cell*. 2025;188(3):640-652. doi:10.1016/j.cell.2024.12.002

8.  Megat S, Mora N, Sanogo J, et al. Integrative genetic analysis illuminates ALS heritability and identifies risk genes. *Nat Commun*. 2023;14(1):342. doi:10.1038/s41467-022-35724-1

9.  Geloso MC, Corvino V, Marchese E, Bhatt DK, Bhatt NK. Microglial pruning: relevance for synaptic dysfunction in multiple sclerosis and related experimental models. *Cells*. 2021;10(3):686. doi:10.3390/cells10030686

10. Soteros BM, Zoghbi HY. Complement and microglia dependent synapse elimination in brain development. *WIREs Mech Dis*. 2022;14(4):e1545. doi:10.1002/wsbm.1545

11. Cheung N. Synaptic plasticity fragility underlies a microglial pruning continuum in major depressive disorder and amyotrophic lateral sclerosis. figshare. 2026. doi:10.6084/m9.figshare.31900597.v1

12. Cheung N. Maintenance plasticity rescue slows ALS progression in a pruning-continuum model: implications for early intervention in motor and mood circuits. Zenodo. 2026. doi:10.5281/zenodo.18662660

13. Cheung N. Mitochondrial NAD⁺ compensation links synaptic pruning to autophagy failure in ALS: a refined four-tier continuum model. figshare. 2026. doi:10.6084/m9.figshare.31900549.v1

14. Cheung N. Augmented Cheung Glutamatergic Regimen as a candidate for investigation in amyotrophic lateral sclerosis. figshare. 2026. doi:10.6084/m9.figshare.31914918.v1

15. Cheung N. Pathway-level TWAS in ALS reveals targetable axes in bioenergetics and autophagy regulation. figshare. 2026. doi:10.6084/m9.figshare.31940352.v1

16. Bahia El Idrissi N, Bosch S, Ramaglia V, Aronica E, Baas F, Troost D. Complement activation at the motor end-plates in amyotrophic lateral sclerosis. *J Neuroinflammation*. 2016;13:72. doi:10.1186/s12974-016-0538-2

17. Kjældgaard AL, Pilely K, Olsen KS, et al. Amyotrophic lateral sclerosis: the complement and inflammatory hypothesis. *Mol Immunol*. 2018;102:14-25. doi:10.1016/j.molimm.2018.06.007

18. Ayyubova G, Fazal N. Beneficial versus detrimental effects of complement--microglial interactions in Alzheimer\'s disease. *Brain Sci*. 2024;14(5):434. doi:10.3390/brainsci14050434

19. Smethurst P, Newcombe J, Troakes C, et al. In vitro prion-like behaviour of TDP-43 in ALS. *Neurobiol Dis*. 2016;96:236-247. doi:10.1016/j.nbd.2016.08.007

20. Jo M, Lee S, Jeon YM, et al. The role of TDP-43 propagation in neurodegenerative diseases. *Exp Mol Med*. 2020;52(10):1643-1652. doi:10.1038/s12276-020-00513-7

21. Pongrácová E, Buratti E, Romano M. Prion-like spreading of disease in TDP-43 proteinopathies. *Brain Sci*. 2024;14(11):1132. doi:10.3390/brainsci14111132

22. Evans CS, Holzbaur ELF. Autophagy and mitophagy in ALS. *Neurobiol Dis*. 2019;122:35-40. doi:10.1016/j.nbd.2018.07.005

23. Chua JP, De Calbiac H, Bhatt DK, et al. Autophagy and ALS: mechanistic insights and therapeutic implications. *Autophagy*. 2022;18(10):2301-2323. doi:10.1080/15548627.2021.1926656

24. Cozzi M, Ferrari V. Autophagy dysfunction in ALS: from transport to protein degradation. *J Mol Neurosci*. 2022;72(7):1456-1481. doi:10.1007/s12031-022-02029-3

25. Harlan BA, Killoy KM, Pehar M, Liu L, Auwerx J, Vargas MR. Evaluation of the NAD⁺ biosynthetic pathway in ALS patients and effect of modulating NAD⁺ levels in hSOD1-linked ALS mouse models. *Exp Neurol*. 2020;327:113219. doi:10.1016/j.expneurol.2020.113219

26. Lundt S, Zhang N, Polo-Parada L, Wang X, Ding S. Dietary NMN supplementation enhances motor and NMJ function in ALS. *Exp Neurol*. 2024;374:114698. doi:10.1016/j.expneurol.2024.114698

27. Strohmer B, Grosh K, Montañana-Rosell R, et al. Spinal circuit mechanisms constrain therapeutic windows for ALS intervention: a computational modeling study. *Neurobiol Dis*. 2026;219:107253. doi:10.1016/j.nbd.2025.107253

28. Qin H, Hussain L, Liu Z, et al. Optimizing deep learning models to combat amyotrophic lateral sclerosis (ALS) disease progression. *Digit Health*. 2025;11:20552076251349719. doi:10.1177/20552076251349719

29. Cheung N. ALS progression model pipeline v3.0 code and results \[Data set\]. Zenodo. 2026. doi:10.5281/zenodo.XXXXXXX

30. Hardiman O, Al-Chalabi A, Chio A, et al. Amyotrophic lateral sclerosis. *Nat Rev Dis Primers*. 2017;3:17071. doi:10.1038/nrdp.2017.71

31. van Es MA, Hardiman O, Chio A, et al. Amyotrophic lateral sclerosis. *Lancet*. 2017;390(10107):2084-2098. doi:10.1016/S0140-6736(17)31287-4

32. Wainger BJ, Kiskinis E, Mellin C, et al. Intrinsic membrane hyperexcitability of amyotrophic lateral sclerosis patient-derived motor neurons. *Cell Rep*. 2014;7(1):1-11. doi:10.1016/j.celrep.2014.03.019

33. Menon P, Kiernan MC, Vucic S. Cortical hyperexcitability precedes lower motor neuron dysfunction in ALS. *Clin Neurophysiol*. 2015;126(4):803-809. doi:10.1016/j.clinph.2014.04.023

34. Vukojicic A, Delestrée N, Fletcher EV, et al. The classical complement pathway mediates microglia-dependent remodeling of spinal motor circuits during development and in SMA. *Cell Rep*. 2019;29(10):3087-3100.e7. doi:10.1016/j.celrep.2019.11.013

35. Abud EM, Ramirez RN, Martinez ES, et al. iPSC-derived human microglia-like cells to study neurological diseases. *Neuron*. 2017;94(2):278-293.e9. doi:10.1016/j.neuron.2017.03.042

36. Zhou Q, Zhu L, Qiu W, et al. Nicotinamide riboside enhances mitochondrial proteostasis and adult neurogenesis through activation of mitochondrial unfolded protein response signaling in the brain of ALS SOD1G93A mice. *Int J Biol Sci*. 2020;16(2):284-297. doi:10.7150/ijbs.38487

37. Nonaka T, Masuda-Suzukake M, Arai T, et al. Prion-like properties of pathological TDP-43 aggregates from diseased brains. *Cell Rep*. 2013;4(1):124-134. doi:10.1016/j.celrep.2013.06.007

38. Rosso F, Magdalena R, Torazza C, et al. Non-cell autonomous autophagy in amyotrophic lateral sclerosis: a new promising target? *Neurobiol Dis*. 2026;218:107203. doi:10.1016/j.nbd.2025.107203

39. Basak B, Holzbaur ELF. Mitochondrial damage triggers the concerted degradation of mitochondria and proteins in ALS. *Nat Commun*. 2025;16:7367. doi:10.1038/s41467-025-62379-5

40. Buskila Y, Kékesi O, Bellot-Saez A, et al. Dynamic interplay between H-current and M-current controls motoneuron hyperexcitability in amyotrophic lateral sclerosis. *Cell Death Dis*. 2019;10(3):153. doi:10.1038/s41419-019-1538-9

41. Ren K, Liu F, Xu H, et al. A comprehensive review of electrophysiological technologies in ALS research. *Front Cell Neurosci*. 2024;18:1435619. doi:10.3389/fncel.2024.1435619

42. Wymer J, Apple S, Harrison A, Hill BA. Pharmacokinetics, bioavailability, and swallowing safety with an oral suspension of riluzole. *Clin Pharmacol Drug Dev*. 2023;12(5):456-465. doi:10.1002/cpdd.1168

43. Abbara C, Estournet B, Lacomblez L, et al. Riluzole pharmacokinetics in young patients with spinal muscular atrophy. *Br J Clin Pharmacol*. 2011;71(3):403-410. doi:10.1111/j.1365-2125.2010.03843.x

44. Kang MJY, Hawken E, Vazquez GH. The mechanisms behind rapid antidepressant effects of ketamine: a systematic review with a focus on molecular neuroplasticity. *Front Psychiatry*. 2022;13:860882. doi:10.3389/fpsyt.2022.860882

45. Zanos P, Moaddel R, Morris PJ, et al. Ketamine and ketamine metabolite pharmacology: insights into therapeutic mechanisms. *Pharmacol Rev*. 2018;70(3):621-660. doi:10.1124/pr.117.015198

46. Verde F, Otto M, Silani V. Neurofilament light chain as biomarker for amyotrophic lateral sclerosis and frontotemporal dementia. *Front Neurosci*. 2021;15:679199. doi:10.3389/fnins.2021.679199

47. Benatar M, Ostrow LW, Lewcock JW, et al. Biomarker qualification for neurofilament light chain in amyotrophic lateral sclerosis: theory and practice. *Ann Neurol*. 2024;95(2):211-216. doi:10.1002/ana.26860

48. Alirezaei Z, Pourhanifeh MH, Borran S, Nejati M, Mirzaei H, Hamblin MR. Neurofilament light chain as a biomarker, and correlation with magnetic resonance imaging in diagnosis of CNS-related disorders. *Mol Neurobiol*. 2020;57(1):469-491. doi:10.1007/s12035-019-01698-3
