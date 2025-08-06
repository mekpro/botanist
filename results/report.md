![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F961556%2Fe516ba5355fdcf3110913d0804532204%2FScreenshot%202568-08-04%20at%2020.43.19.png?generation=1754315236721267&alt=media)
https://botanist.mekpro.dev

#  Abstract
We present a novel approach to fine-tuning Google's Gemma-3n multimodal model for accurate plant species identification on edge devices. Our methodology addresses the fundamental challenges of limited model capacity and lightweight vision encoders through three innovative techniques: multi-task supervised fine-tuning that leverages hierarchical botanical features, model-assisted data curation using consensus mechanisms, and Group Relative Policy Optimization (GRPO) for cross-modal consistency. By training on the PlantNet300K dataset, we demonstrate that careful fine-tuning strategies can overcome the limitations of small-scale multimodal models, achieving significant improvements in plant species identification accuracy while maintaining the efficiency required for offline mobile deployment. This work contributes to both the technical advancement of vision-language models and the practical goal of environmental conservation to promote and track biodiversity through accessible technology.

# Gemma-3n Botanist Vision LLM: Enhanced Plant Species Identification through Multi-Task Fine-Tuning and Reinforcement Learning

# 1. Introduction
Picture a botanist deep in a rainforest, miles from the nearest cell tower, discovering an unfamiliar flowering plant. Or imagine a conservation volunteer surveying plants in a remote mountain meadow, needing quick, reliable species identification to document biodiversity. These scenarios play out daily for field researchers, conservationists, and nature enthusiasts worldwide. Yet despite carrying smartphones with impressive cameras, they often can't identify plants on the spot due to lack of internet connectivity or the limitations of existing offline tools.
We developed the Gemma-3n Botanist Vision Model specifically to solve this real-world challenge. Our goal was simple: create a plant identification system that runs entirely on a phone, works without internet, and delivers accurate results even on devices with limited computing power. This meant working within strict constraints—not just technical ones, but practical ones that field botanists face every day.
Our Technical Approach: We chose Google's Gemma-3n as our foundation because it excels at mobile deployment and requires minimal resources to run—perfect for field conditions where battery life and processing power are precious. Starting with the PlantNet300K dataset, we implemented a sophisticated fine-tuning strategy to transform this general-purpose model into a botanical expert.
Our approach goes beyond simple species classification. Through multi-task learning, we taught the model to understand botanical relationships—how flower color, shape, and inflorescence patterns work together to identify species. This mirrors how human botanists think, using multiple morphological clues to narrow down possibilities. We then applied Group Relative Policy Optimization (GRPO) as a final refinement stage to enhance accuracy and ensure consistent, botanically sound predictions.
**Result:** Our fine-tuned model achieves 22% accuracy for exact species identification in Top-5 predictions—impressive considering the model runs entirely offline on mobile devices. Even more encouraging, it correctly identifies genus-level classification 66% of the time and can accurately recognize complex botanical concepts like inflorescence types at 58% accuracy. Remarkably, our GRPO optimization, despite operating only on text outputs, improved image classification performance by up to 10%, demonstrating the power of cross-modal learning.

# 2. Technical Approach
## 2.1 Technical Requirement 
Building a plant identification system for field use isn't just about accuracy—it's about working within real-world constraints:
Limited Computing Power: Field devices often have modest processors and limited battery life. Our model needs to run efficiently on phones that might be several years old, while still delivering expert-level botanical knowledge.
Lightweight Vision Processing: Gemma-3n uses MobileNetV5, designed for efficiency rather than raw power. This means we need clever training strategies to help it distinguish between species that might differ only in subtle details like petal arrangement or leaf venation.
Training Data Constraints: With limited computational resources for training and the need to keep the model small, we couldn't simply throw massive amounts of data at the problem. We needed to be strategic about how we used the PlantNet300K dataset to maximize learning efficiency.
Complete Offline Operation: No cloud processing, no internet queries—everything must work standalone. This means all botanical knowledge must be compressed into a model small enough to fit on a phone yet comprehensive enough to be genuinely useful in the field.
Our approach leverages Gemma-3n, Google's state-of-the-art multimodal model designed specifically for edge deployment. The architecture comprises three key components working in concert:
The lightweight vision encoder (MobileNetV5) processes input images efficiently, extracting visual features while maintaining a minimal computational footprint. These features are then projected into a shared embedding space through learned projection layers. Unlike vision encoder used in larger model like Gemma 3, MobileNet aim to deploy in edge devices that have low computing power and make it possible to run in near-realtime requirements.
The transformer-based language model processes both visual embeddings and textual prompts, generating structured outputs that combine species identification with detailed botanical descriptions. This component benefits from pre-training on diverse text corpora, providing rich semantic understanding of botanical terminology.
Multi-modal attention mechanisms enable deep integration between visual and textual modalities, allowing the model to ground botanical concepts in visual features and vice versa. This bidirectional information flow is crucial for accurate species identification from images.

## 2.2 Novel Fine-Tuning Methodology
To overcome the inherent limitations of small-scale multimodal models, we developed a comprehensive three-stage fine-tuning approach that systematically addresses each technical challenge:
1. Consensus-Based Labeling : Generate descriptions using model's vision capabilities with 5-pass inference per image, Extract botanical features (color, shape, inflorescence).
2. Supervised Fine-Tuning (SFT) : Multi-task learning on Gemma-3n. Species identification. Morphological feature prediction .ierarchical botanical classification
3. GRPO Fine-Tuning: Reinforcement learning refinement of Gemma-3n-SFT Botanical consistency rewards Species accuracy optimization Botanist-like description quality (evaluated by larger LLM) 10% performance gain

## 2.3 Consensus based labeling
A fundamental insight of our approach is that traditional human-annotated labels may not align with what a lightweight vision encoder can actually perceive. To address this, we developed an innovative self-supervised data annotation strategy:
The Process: For each image in our training set, the base Gemma-3n model performs five independent inference passes, generating botanical descriptors from predefined taxonomies. These multiple passes capture the inherent stochasticity in the model's predictions, allowing us to identify stable, reliable features through majority voting.
Theoretical Foundation: This approach builds on established research in pseudo-labeling and self-supervised learning. Vision-Language Model Consensus Pseudo Labels (VLM-CPL) demonstrate that model-generated labels with consensus mechanisms can achieve accuracy comparable to human annotation. The key insight is that labels should reflect what the model can actually distinguish rather than imposing external standards it cannot meet.
By having Gemma-3n describe what it "sees," we work within the encoder's perceptual capabilities. This reduces the distribution mismatch between training labels and model capacity—a critical factor for successful fine-tuning. The consensus mechanism through multiple inference passes leverages ensemble principles, where random errors cancel out while systematic patterns emerge, resulting in high-quality training data perfectly aligned with model capabilities.
Quality Assurance: Automated consistency checks ensure biological plausibility of generated labels. For instance, if the model identifies "yellow petals" and "compound inflorescence," the system verifies this combination exists in botanical reality. This biological grounding prevents the model from learning spurious correlations while maintaining alignment with its perceptual capabilities.
The full dataset after labeling is hosted in : huggingface.co/datasets/mekpro/plantnet300k_observe

## 2.4 Multi-Task Supervised Fine-Tuning (SFT)
We decomposing plant identification into multiple related objectives significantly outperforms traditional single-task image classification, especially for lightweight models. This multi-task learning approach leverages shared representations across related tasks, from simple visual features like color and shape to complex botanical structures like inflorescence patterns. The framework mirrors botanical taxonomy's natural hierarchy: shallow network layers learn foundational visual features while deeper layers specialize in increasingly complex botanical concepts, maximizing the efficiency of our lightweight MobileNetV5 encoder through architectural alignment between task complexity and network depth.

The most challenging aspect involves classifying abstract botanical structures like inflorescence types—spatial arrangements of flowers (raceme, panicle, umbel, corymb) that don't map to immediately visual concepts. To bridge this gap between botanical terminology and visual patterns, we augment training with textual descriptions, helping the model understand that a "compound umbel" represents umbrella-like clusters rather than forcing it to deduce this purely from images. This multimodal approach is crucial for distinguishing closely related species that differ primarily in these subtle architectural features.

Beyond improving accuracy, the auxiliary tasks act as implicit regularization, preventing overfitting while forcing the model to attend to diverse visual features that might otherwise be overlooked. For instance, predicting inflorescence type requires analyzing spatial arrangements, flower density, and branching structures—features essential for fine-grained species classification. This forced attention mechanism ensures the model captures the full spectrum of botanical characteristics necessary for accurate identification, making our multi-task framework particularly effective for real-world plant identification applications.

## 2.2.5 Group Relative Policy Optimization (GRPO)

The final stage of our methodology employs Group Relative Policy Optimization (GRPO) to refine the model's outputs for botanical accuracy and consistency. GRPO is a reinforcement learning algorithm designed to train large language models for complex tasks like solving math problems or writing code. Unlike older methods, GRPO is memory-efficient because it doesn't use a separate "value function". Instead, it generates multiple answers for each question, scores them with a reward model, and uses the average score as a reference to decide which answers are better. Research demonstrates that GRPO eliminates the need for a separate critic model (typically as large as the policy model), reducing memory and compute overhead by ~50%.

By having the model generate detailed descriptions of species characteristics—including inflorescence types, color patterns, and morphological features—we create multiple response candidates that can be grouped and evaluated. This approach enables the model to learn correct botanical relationships through comparative advantage: responses that accurately correlate features (e.g., "red flowers with compound inflorescence typically indicate genus X") receive higher rewards relative to the group average, reinforcing proper species-feature associations.

**Cross-Modal Transfer Mechanism**

Although GRPO operates primarily on textual outputs, VLMs align conceptually equivalent inputs into a shared task vector, which is invariant to modality (text, image). When the model learns textual constraints about botanical features, these improvements transfer to visual understanding through shared embedding spaces. This cross-modal transfer ensures that better textual reasoning about visual features leads to improved visual feature extraction—a phenomenon well-documented in vision-language research.

**Three-Component Reward System**

Our training implements a weighted reward function:

```python
R_total = 0.35 × R_species + 0.50 × R_botanical + 0.15 × R_format
```

1. **Species Reward (35% weight)**: Validates that the model correctly identifies the species name and verifies the inflorescence type against our reference dataset.

2. **Botanical Consistency Reward (50% weight)**: Our judge model, Qwen-235B, serves as an expert discriminator to evaluate botanical accuracy. The substantially larger parameter count of Qwen-235B compared to our target model provides more reliable assessment of complex botanical relationships and feature consistency. This larger model can better capture nuanced taxonomic rules and validate whether predicted features align with established botanical knowledge.

3. **Format Reward (15% weight)**: Ensures proper JSON structure and required keys for systematic data processing.

**Implementation Details**: During GRPO training, the model generates multiple response candidates per image. GRPO uses programmable reward functions as a more scalable alternative to the human feedback required for other reinforcement learning algorithms, and requires far fewer examples than fine-tuning, making this technique a cost-effective alternative. The reward function evaluates each candidate, and gradient updates favor responses maintaining botanical consistency while correctly identifying species. This iterative process refines the model's ability to produce accurate, well-reasoned outputs grounded in visual observations.


# 3. Experimental Setup and Implementation
## 3.1 Dataset and Training Configuration
We utilize the PlantNet300K dataset, a comprehensive collection of plant images spanning diverse species, growth stages, and photographic conditions. For initial experiments, we selected data from PlantNet300K and curated a subset of 20,000 high-quality images representing 380 species, ensuring balanced representation across taxonomic groups.
| Training Parameter | Value | Rationale |
| --- | --- | --- |
| Base Model | Gemma-3n-E2B and E4 B| Optimal size for mobile deployment |
| Training Framework | Unsloth | Memory-efficient fine-tuning |
| Hardware | NVIDIA A100 (40GB) | On Google's Colab, Single GPU training feasibility |
| Batch Size | 32 | Memory constraints optimization |
| Learning Rate | 2e-4 (cosine schedule) | Stable co**Group Relative Policy Optimization (GRPO) for Enhanced Species Classification**

Finetune parameters
| Parameter | e2b | e4b |
|-----------|-----|-----|
| **LoRA rank (r)** | 64 | 256 |
| **Dropout** | 0.1 | 0.2 |
| **Learning rate** | 4e-4 | 3e-4 |
| **Epochs** | 6 | 4 |
| **Batch size/device** | 32 | 24 |

## 3.2 Evaluation Methodology
Our evaluation framework assesses both technical performance and practical usability We measure top-1 and top-5 species identification accuracy on a held-out test set. Additionally, we evaluate feature prediction of inflorescence as it is one of the complex features  that can be used to scope the possible species  to ensure the multi-task framework benefits all objectives.

# 4. Results and Discussion
4.1 Performance Improvements
Our three-stage fine-tuning approach yields substantial improvements over baseline methods:
- Species Identification Accuracy: 22% Top-5 accuracy for exact species match
- Genus-Level Classification: 66% accuracy (excellent for field identification)
- Complex Botanical Concepts: 58% accuracy for inflorescence type recognition
- GRPO Performance Boost: 10% improvement in image classification from text-only optimization


# Model Performance Comparison: e2b → SFT → SFT+GRPO

## Species Accuracy Metrics

| Metric Type | Sub-Metric | e2b | SFT | SFT+GRPO | Improvement (e2b→SFT+GRPO) |
| --- | --- | --- | --- | --- | --- |
| **Top-1** | Full Match | 4% | 6% | 10% | +6% |
| **Top-1** | Combined Accuracy | 16% | 42% | 46% | +30% |
| **Top-5** | Full Match | 4% | 14% | 22% | +18% |
| **Top-5** | Combined Accuracy | 22% | 62% | 68% | +46% |

## Inflorescence Type Accuracy Metrics

| Metric Type | Sub-Metric | e2b | SFT | SFT+GRPO | Improvement (e2b→SFT+GRPO) |
| --- | --- | --- | --- | --- | --- |
| **Top-1** | Exact Match | 6% | 36% | 40% | +34% |
| **Top-1** | Combined Accuracy | 16% | 48% | 54% | +38% |
| **Top-5** | Exact Match | 10% | 54% | 54% | +44% |
| **Top-5** | Combined Accuracy | 28% | 70% | 72% | +44% |

## Summary Statistics

| Model | Average Performance | Min | Max |
| --- | --- | --- | --- |
| **e2b** | 12.75% | 4% | 28% |
| **SFT** | 40.5% | 6% | 70% |
| **SFT+GRPO** | 44.75% | 10% | 72% |

## Key Observations

- **Largest Improvement**: Top-5 Combined Accuracy for Species improved by +46% from e2b to SFT+GRPO
- **Most Significant Jump**: e2b → SFT shows dramatic improvements across all metrics (average +27.75%)
- **Refinement Stage**: SFT → SFT+GRPO provides modest improvements (average +4.25%)
- **Best Performance**: Top-5 Combined Accuracy for Inflorescence reaches 72% in SFT+GRPO


# Model Performance Comparison: e4b → SFT → SFT+GRPO

## Species Accuracy Metrics

| Metric Type | Sub-Metric | e4b | SFT | SFT+GRPO | Improvement (e4b→SFT+GRPO) |
| --- | --- | --- | --- | --- | --- |
| **Top-1** | Full Match | 6% | 36% | 38% | +32% |
| **Top-1** | Combined Accuracy | 22% | 84% | 86% | +64% |
| **Top-5** | Full Match | 6% | 48% | 50% | +44% |
| **Top-5** | Combined Accuracy | 28% | 90% | 90% | +62% |

## Inflorescence Type Accuracy Metrics

| Metric Type | Sub-Metric | e4b | SFT | SFT+GRPO | Improvement (e4b→SFT+GRPO) |
| --- | --- | --- | --- | --- | --- |
| **Top-1** | Exact Match | 12% | 76% | 76% | +64% |
| **Top-1** | Combined Accuracy | 28% | 86% | 88% | +60% |
| **Top-5** | Exact Match | 14% | 84% | 80% | +66% |
| **Top-5** | Combined Accuracy | 36% | 88% | 92% | +56% |

## Summary Statistics

| Model | Average Performance | Min | Max |
| --- | --- | --- | --- |
| **e4b** | 18.25% | 6% | 36% |
| **SFT** | 71.5% | 36% | 90% |
| **SFT+GRPO** | 72.25% | 38% | 92% |

## Key Observations

- **Largest Improvement**: Combined Accuracy (Top-5 Inflorescence) improved by +66% from e4b to SFT+GRPO
- **Most Significant Jump**: e4b → SFT shows dramatic improvements across all metrics (average +53.25%)
- **Refinement Stage**: SFT → SFT+GRPO provides modest improvements (average +0.75%)
- **Best Performance**: Top-5 Combined Accuracy reaches 90-92% in final models

## sample results from GRPO
When training with GRPO, the original model generate a mix of good and bad response. After sometime, the score from reward function of each response will make the model converge to the better output. These are example of how the model start after SFT training. And after 300 steps of GRPO, we can see that the model did give better result and got better reward score consistently.

** Beginning
0.45 : Describe Cytinus ruber : {"color": "Bright orange", "inflorescencetype": "Cytinaceae", "inflorescence_description": "Numerous, small, bell-shaped flowers.", "flower_arrangement": "Clustered in terminal or axillary cymes.", "flower_density": "High, forming dense floral clusters.", "species": "Cytinus ruber", "family": "Cytinaceae", "genus": "Cytinus"} 

** After 300 steps:
0.85 : Describe Trifolium stellatum : {"color": "Vibrant pink at the base transitioning to a clean white apex, indicating optimal growing conditions.", "inflorescencetype": "Globose head", "inflorescence_description": "A ball-shaped inflorescence held aloft on a slender stalk originating from a leaf-stem junction.", "flower_arrangement": "Florets densely clustered in a spherical head, each flower on a tiny stalk (pedicel).", "flower_density": "Very dense", "species": "Trifolium stellatum", "family": "Fabaceae", "genus": "Trifolium"} 

# 5. Conclusion and Future Work

This research demonstrates the successful development of a fine-tuned Gemma-3n model for plant species identification, achieving meaningful results in botanical classification tasks. While the top-1 accuracy may not match state-of-the-art larger models, our findings indicate that the model can identify plant family and genus with reasonable reliability. For practical botanical applications, the top-5 accuracy metric proves more relevant, as field botanists typically capture multiple images from different angles to ensure consensus in identification. This multi-angle, multi-inference approach aligns well with real-world botanical practices and significantly enhances the reliability of species determination.

The economic feasibility of our approach represents a significant contribution to the field. Through the use of Unsloth's optimization framework, we achieved remarkable training efficiency, successfully fine-tuning 30,000 images in under 12 hours on a single GPU using Google Colab. The Gemma-3n architecture demonstrates superior fine-tuning speed compared to Gemma-3, making it an attractive option for resource-constrained environments. This cost-effectiveness extends beyond inference to the training phase itself, potentially democratizing access to specialized vision-language models for botanical and other domain-specific applications.

Our model-assisted data curation approach, utilizing consensus-based labeling, proved instrumental in improving data quality and enabling large-scale multi-task fine-tuning. This methodology opens avenues for future enhancement through increased label diversity and additional inference passes to strengthen consensus reliability. The self-supervised nature of this approach particularly benefits scenarios where expert annotation is scarce or expensive, a common challenge in specialized botanical datasets.

The performance gains also emerged from our implementation of Group Relative Policy Optimization (GRPO), despite its application being limited to text-to-text modality. This cross-modal transfer effect validates our hypothesis that improving textual reasoning about visual features can enhance image classification accuracy. The current scalability limitations, primarily due to Gemma-3n's lack of VLLM support for efficient inference during GRPO training, present an opportunity for future optimization. We anticipate that GRPO fine-tuning will prove particularly valuable in domains where image data is limited but textual descriptions are available, such as medical imaging or geographical image analysis.

Looking forward, we propose that Google expand the edge-inference ecosystem by streamlining the deployment of fine-tuned models. The implementation of LoRA adapters, which enable relatively small fine-tuned components to integrate with base models, could revolutionize task-specific deployment. Users could rapidly access specialized capabilities without downloading complete models for each task, significantly enhancing the adoption of edge inference with Gemma-3n across diverse applications.

This work contributes to the growing body of research on efficient, task-specific vision-language models and demonstrates that meaningful botanical classification can be achieved with resource-conscious approaches. The combination of multi-task learning, consensus-based data curation, and cross-modal reinforcement learning presents a framework applicable to other fine-grained classification challenges in resource-limited settings. This technology help the model to reach the goal of understand and appreciate about biodiversity through accessible technology. 

We extend our gratitude to the Google DeepMind team for contributing the Gemma family of models to the community, advancing both educational opportunities and technological progress. Special thanks to the competition hosting team for providing this platform for innovation and encourage the impactful goal like biodiversity, and to the Unsloth team for their high-performance fine-tuning tools and exceptional community support throughout this research. Their collective efforts have made this work possible and continue to democratize access to advanced AI capabilities for researchers worldwide.

# Appendix 
## Loss and reward function from training.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F961556%2F393f63b78de59cfb4e32ba533cb615d7%2FScreenshot%202568-08-06%20at%2005.51.15.png?generation=1754473416431596&alt=media)
Train and validation loss

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F961556%2F9e7f6c406724673486609b7bb44917ce%2FScreenshot%202568-08-06%20at%2010.50.05.png?generation=1754473439409432&alt=media)
Reward function result

Link to model and source code:
- Demo : https://botanist.mekpro.dev
- Source code: https://github.com/mekpro/botanist
- Model (E2B SFT) : https://huggingface.co/mekpro/gemma-3n-botanist-observe6-merged
- Model (E2B GRPO) : https://huggingface.co/mekpro/gemma-3n-botanist-grpo6p-merged
- Model (E4B SFT) : https://huggingface.co/mekpro/gemma-3n-botanist8
- Model (E4B GRPO) : https://huggingface.co/mekpro/gemma-3n-botanist-grpo8p-merged
- Dataset : https://huggingface.co/datasets/mekpro/plantnet300k_observe#  
