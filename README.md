# When Fixes Teach: Repair-Aware Contrastive Learning for Optimization-Resilient Binary Vulnerability Detection
Deep learning (DL)-based vulnerability detection in source code are prevalent, yet detecting vulnerabilities in binary code using this paradigm remains underexplored. The few works typically treat input instructions as individual entities, failing to extract and leverage fine-grained information due to their inability to account for the inherent connections and correlations between code segments and the impact of compilation optimizations. To address these challenges, this paper proposes DELTA, a novel approach that incorporates Dynamic contrastive lEarning with vuLnerabiliTy repair Awareness to fine-tune pre-trained models, significantly enhancing the accuracy and efficiency of vulnerability detection in binary code. DELTA proceeds by standardizing assembly instructions and utilizing function pairs that represent code before and after vulnerability repair along with their versions compiled under different optimization settings as contrastive learning samples. Building on these rich and diverse training signals, DELTA fine-tunes CodeBERT using contrastive learning augmented with masked language modeling, resulting in a feature encoder CMBERT, which is adept at capturing nuanced vulnerability patterns in binary code and remain resilient to the impacts of compilation optimizations. DELTA is evaluated on the Juliet Test Suite dataset, achieving an average performance improvement of 8.04% in detection accuracy and 7.13% in F1 score compared to alternative methods.
# Design of DELTA
![image](https://github.com/user-attachments/assets/e6ee17ed-5469-4200-a323-b71fec1a417b)
# Dataset
To evaluate tthe effectiveness of Clever, we use the Juliet Test Suite dataset, which is generated from version 1.3 of the Juliet Test Suite for C/C++ (SARD, 2017)(https://samate.nist.gov/SARD). Developed by NIST SARD, the Juliet Test Suite consists of certified code samples designed to benchmark static vulnerability detection tools and has been widely adopted in vulnerability detection research. The dataset, originally provided in source code form, is compiled into executable files and subsequently processed to extract assembly code using reverse engineering tools. The dataset construction process is outlined as follows:

1. **Compilation**: The Juliet Test Suite is organized by vulnerability categories. For the experiments in this study, the dataset is compiled using the MSVC compiler with eight different compilation options applied. 

2. **Decompilation**: Decompilation is the process of converting binary code into assembly code segments. We use IDA Pro v7.5, a reverse engineering tool, to decompile the .o files generated during compilation and convert them into assembly language, allowing us to extract key information such as program instructions and function names.

3. **Construction of the Vulnerability Detection Dataset**: Based on the standardized function names, functions representing pre-repair vulnerabilities labeled as "1" (vulnerable), while post-repair functions are labeled as "0" (non-vulnerable). To enhance the comprehensiveness of the dataset, binary code samples compiled under eight distinct optimization options (O1, O2, Od, Og, Oi, Os, Ot, Ox) are collected. After careful selection and organization, a final binary code vulnerability dataset is constructed, containing a total of 504,610 function samples.
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
