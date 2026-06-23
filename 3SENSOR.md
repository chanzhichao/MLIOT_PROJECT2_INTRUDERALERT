# 🔬 3-Sensor Fusion Pipeline & Decision Logic

This document covers the custom algorithmic modifications designed for this project's sensor fusion pipeline, utilizing telemetry from a PIR motion sensor, a microphone array, and camera computer vision.

---

## 🛠️ Sensor Fusion Pipeline Modifications

### A. PIR Adaptive Decay Baseline
Standard PIR sensors yield binary high/low states that fluctuate rapidly when a target stands relatively still, exposing the pipeline to localized false-negative assessments. 
* To stabilize the vector, we implement an **adaptive decay baseline initialized at $0.5$**.
* When active motion ceases, the system gracefully decays the value from $0.5 \to 0$ over an extended temporal window rather than dropping it instantly. 
* This lingering logic keeps the pipeline aware that a target may still occupy the space despite temporary physical stillness.

### B. High-Fidelity Microphone & Audio Processing
* **Hardware Sample Rate:** Hardcoded strictly at **$48000 \text{ Hz}$** to match the target device hardware capabilities and prevent downsampling driver exceptions.
* **IMPORTANT: 1-2s Acoustic Sliding Window:** Audio arrays are evaluated using a moving $1$ to $2$-second sliding window framework. 
  > 💡 **The Structural Reason:** Without an integrated windowing matrix, the deployment framework evaluates minute sound *bits* rather than sequential frequency signatures.
  >
  > For example, a glass break event consists of a highly distinct *High $\Leftrightarrow$ Low $\Leftrightarrow$ High* frequency transition pattern. Evaluating single static moments reduces the system to matching individual isolated pitches—destroying the predictive capability of the underlying TFLite model.
  >
  > It is the programmatic equivalent of demanding a human identify a complex classical symphony from an isolated individual note rather than an entire musical measure.
  >
  > Like those videos where the person goes "Try to guess the classical piece from one note / one chord" 🎹🎵︎
  > 
* **Max Delta Decibel Layering ($\Delta \text{ dB}$):** The pipeline independently tracks real-time differential amplitude metrics parallel to the TFLite inference array. This introduces raw volume acceleration as an independent feature layer. If an edge-case sound signature bypasses frequency recognition, a sharp, sudden structural volume spike will still trigger system escalations.

### C. Spatial Camera Boundaries (Tripwire Architecture)
The visual confirmation pipeline utilizes an explicit **Dual-Stage Bounding Box** verification condition:
1. **Classification Bounding Box:** Detects and localizes human forms alongside their independent inference model confidence values.
2. **Area of Interest (AoI) Bounding Box / Tripwire:** A user-defined polygon tracking restricted geographic space.
* *The Trigger Rule:* Detection events are only promoted to alarm states if the spatial intersection of both boxes occurs simultaneously.

---

## 🧠 The "1 + 1" Classification Paradox (Go to *Audio Classification Analogy Logic.md* for the detailed description for this problem.)

When evaluating live acoustic anomalies, standard classifiers encounter a fundamental cognitive block. In real-world environments, baseline background noise does not magically vanish when an intruder makes a sound; the signature is purely additive:

$$\text{Total Acoustic Profile} = \text{Background Baseline} + \text{Additive Anomaly}$$

This forces the AI model into an abstract statistical loop similar to asking a child, *"What is 1 + 1?"*:
* **Choice A:** *"More than 1"* (Confidence: **70%**)
* **Choice B:** *"Exactly or Around 2"* (Confidence: **20%**)

From a structural perspective, Choice A is entirely valid—the machine identifies the overwhelming background data with massive statistical predictability and treats the anomaly as negligible noise. However, Choice B is the target event we must capture. 

Simply forcing manual weights onto global post-prediction thresholds does not solve the nature of the problem—it merely causes the system to mistake normal environmental fluctuations (like an AC unit shifting speeds) for real threats, spiking false alarms. 

> 🔍 *The Core Question:* How do we programmatically introduce systemic skepticism into our architecture so it looks past a highly confident "Normal" status (Choice A) to flag the underlying, low-confidence anomaly (Choice B)?
>
> ### 📈 The Solution: Dynamic Standard Deviation ($\sigma$) Shifts

To force the machine to look past Choice A ("Normal") and identify Choice B ("The Anomaly"), we cannot rely on static thresholds. Instead, the pipeline implements a **Standard Deviation ($\sigma$) Shift Mechanism**. 

#### The Intuition: Redefining "Normal"
Think of standard deviation as the boundary of what the machine considers acceptable "background chatter." 
* In a completely quiet room, the background audio variance is tiny (a narrow standard deviation).
* In a noisy environment, the background variance is naturally wider.

If we use a fixed threshold, a small anomaly in a quiet room will be completely ignored, while a normal sound spike in a noisy room will trigger a false alarm. 

#### The Logic of the Shift
Instead of asking, *"Is this sound loud?"*, the algorithm continuously calculates the running mean ($\mu$) and the standard deviation ($\sigma$) of the environmental baseline. It then calculates the trigger threshold dynamically:

$$\text{Trigger Threshold} = \mu + (k \cdot \sigma)$$

Where $k$ is our tuning coefficient (the number of standard deviations away from the average). 

When the audio signature shifts by a specific multiple of standard deviations (e.g., $3\sigma$), the system mathematically realizes that the current input is statistical noise *no longer*. 

#### How this solves the "1 + 1" Paradox:
By applying this standard deviation shift, we are effectively modifying the child's logic. We are telling the machine: 
1. Establish the statistical weight of the baseline "1".
2. If the incoming signal deviates from that baseline by more than $3\sigma$, **strip the baseline out of the equation.**
3. Evaluate *only* the remaining deviation. 

> This mathematical "skepticism" forces the pipeline to ignore the massive 70% confidence of the background and immediately amplify the low-confidence 20% anomaly, triggering the intruder alert accurately regardless of how loud or quiet the room is.

### 🍒 The "Cognitive Dissonance" Analogy

To fully appreciate this architecture, consider a child who has just successfully solved 100 traditional, pristine math questions in a row. Suddenly, they are presented with a variation where the answer to $1 + 1$ is explicitly presented as *"Around 2"*. 

The child does not hesitate because they lack the intelligence to solve it, nor because they are looking for an easy out. They freeze because **they recognize that this question violates the traditional baseline of the previous 100 examples.** They sense a hidden variance—a shift in context. 

When facing this dilemma, an engineering shortcut may try to "gaslight" the child. For example:
> **Repeated Fine Tuning** - Telling the child their worldview is wrong,
>
> **Removing the class** - Telling the child to ignore,
>
> Or straight up just create an even bigger dataset - Giving a child more math questions to solve, drilling to no end.

But suppressing that instinct is fundamentally wrong. If you tell the child their worldview is wrong, their imagination becomes limited, and fails to imagine beyond any creative intruder. If you force the child to blindly ignore the background, you destroy their situational awareness. If you give too much data, the child becomes confused. The moment the baseline shifts, the system becomes completely blind. 

Instead of gaslighting the system, the **Standard Deviation Shift** validates that human-like skepticism:
* The previous 100 questions represent our steady-state **Environmental Baseline ($\mu$)**.
* The sudden odd framing represents the **Statistical Deviation ($\sigma$)**.

Instead of allowing the machine to blindly process the data through its normal routine, the standard deviation threshold forces the system to mimic that human "pause." It acknowledges that while the baseline signal is heavily present, the subtle architectural shift indicates that a non-traditional event—an intruder—is actively altering the environment.

---
 
> # Final Decision
> - Camera bounding box intersect OR
> - Audio Frequency ANN Prediction Class standard deviation shift > 3.0 ($\Delta\sigma >3.0$) OR
> - PIR status + (Decibel Spike + Moving Average) > 13.0

Case 1: An intruder gets close to a specific sensitive location --- Detected by camera

Case 2: A crowded, noisy place when suddenly a person screaming, the standard deviation 'shifts' because the classfication detects the person screaming, using the "1+1" analogy in *Audio Classification Analogy Logic.md*

Case 3: It's a rainy day with thunderstorms, The Moving Average will prevent random significant Decibel spikes
