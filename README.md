# When Fixes Teach: Repair-Aware Contrastive Learning for Optimization-Resilient Binary Vulnerability Detection
Deep learning (DL)-based vulnerability detection in source code are prevalent, yet detecting vulnerabilities in binary code using this paradigm remains underexplored. The few works typically treat input instructions as individual entities, failing to extract and leverage fine-grained information due to their inability to account for the inherent connections and correlations between code segments and the impact of compilation optimizations. To address these challenges, this paper proposes DELTA, a novel approach that incorporates Dynamic contrastive lEarning with vuLnerabiliTy repair Awareness to fine-tune pre-trained models, significantly enhancing the accuracy and efficiency of vulnerability detection in binary code. DELTA proceeds by standardizing assembly instructions and utilizing function pairs that represent code before and after vulnerability repair along with their versions compiled under different optimization settings as contrastive learning samples. Building on these rich and diverse training signals, DELTA fine-tunes CodeBERT using contrastive learning augmented with masked language modeling, resulting in a feature encoder CMBERT, which is adept at capturing nuanced vulnerability patterns in binary code and remain resilient to the impacts of compilation optimizations. DELTA evaluated on the Juliet dataset achieves an average performance improvement of 8.04% in detection accuracy and 7.13% in F1 score compared to alternative methods. It also demonstrates superior generalization capability on a real-world test set with average gains of 8.73% in accuracy and 8.10% in F1 over the baseline models.
# Design of DELTA
![image](https://github.com/user-attachments/assets/e6ee17ed-5469-4200-a323-b71fec1a417b)
# Dataset
To evaluate DELTA, we construct a binary vulnerability dataset from the Juliet Test Suite for C/C++(https://zenodo.org/records/4701387). The Juliet Test Suite, developed by NIST SARD, provides synthetic yet systematically designed code samples with ground-truth labels across diverse CWE vulnerability categories, making it a widely-adopted benchmark for vulnerability detection research. Our dataset construction pipeline transforms the original source code into binary representations through the following stages:

1. **Multi-Configuration Compilation**: We compile the Juliet source code using MSVC across eight optimization configurations. These settings provide comprehensive coverage of the optimization landscape, including Od (no optimization), O1 (size optimization), O2 (speed optimization), Ox (maximum optimization) and so on. This diversity captures how different compiler transformations, including function inlining, loop unrolling, dead code elimination, and register allocation, affect binary-level vulnerability manifestation, enabling evaluation of model robustness to optimization-induced variations as well as model generalization ability across the optimization settings.

2. **Binary Disassembly and Decompilation**: The produced binary files are disassembled using IDA Pro v7.5, one of the most well-recognized reverse engineering framework, to identify the functions and extract their corresponding assembly instruction sequences. We also extract for each function its decompiled C-style pseudo-code, since some of the baseline models to be discussed operates on this form of input.

3. **Ground Truth Labeling**: The Juliet Test Suite’s function naming conventions systematically encode vulnerability status and type. We leverage these conventions to assign binary labels, functions representing pre-repair vulnerabilities are labeled as “1” (vulnerable), while post-repair functions are labeled as “0” (non-vulnerable). Also, we correlate each vulnerable function to its CWE-ID, so as to inspect the models’ performance under different vulnerability types. After processing and validation, a final binary code vulnerability dataset is constructed, containing a total of 504,610 function samples with balanced class distribution, i.e., 50% vulnerable functions and 50% correspondingly repaired benign functions.

**The datasets used in our experiments can be downloaded from the following links**
https://drive.google.com/drive/folders/1YbIyphNUownj11Tu9slMkKKmF9sviHeS?usp=drive_link
   
# Source Code
## Step1: Assembly Instruction Normalization
```
python ./normalization.py
```
## Step2: Contrastive Learning Fine-tuning
```
python ./pretraining.py --[parameter]...
```
## Step3: Train and Evaluate our model
```
python ./classifier.py --[parameter]...
```
