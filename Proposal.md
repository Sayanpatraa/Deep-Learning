# **CanisNet: Dog Breed Recognition & Image Generation System**
### Authors: Adam Stuhltrager | Junhua Deng | Sameer Batra | Sayan Patra

---

## **1. Introduction**

This project aims to build a fully integrated dog-breed understanding pipeline using modern deep learning. We address two major tasks:

1. **Fine-grained Dog Breed Classification** using transfer learning (ResNet).  
2. **Text-to-Image Dog Generation** using diffusion-based generative models (Stable Diffusion).

Dog-breed recognition is challenging due to subtle visual differences. Combining classification with image generation demonstrates a powerful “understand + create” AI system.

---

## **2. Objectives**

### **Primary Objectives**
- Build an accurate dog breed classifier using **transfer learning** with a pretrained ResNet.
- Implement a **text-to-image generation pipeline** capable of creating realistic dog images from breed names.
- Integrate classifier + generator into one system for real-time breed prediction and image synthesis.

### **Secondary Objectives**
- Use a **feedback mechanism** where generated images help improve the classifier.
- Analyze misclassifications and redesign augmentation strategies.
- Use generated images to balance underrepresented dog breeds.

---

## **3. Dataset**

**Dataset:** Kaggle – *Dog Breed Identification*  
**Link:** https://www.kaggle.com/competitions/dog-breed-identification/

- **10,222 labeled images**
- **120 dog breeds**
- Fine-grained, complex, and highly diverse images  
- Will create an **80/20 train-validation split** (ps)

---

## **4. System Architecture Overview**

The pipeline consists of two core modules:
Image → ResNet Classifier → Breed Label → Text-to-Image Generator → New Dog Images

Both modules use **transfer learning** for efficient and high-quality results.

---

## **5. Component 1: Dog Breed Classification (Transfer Learning)**

### **5.1 Model Choice — ResNet**

We will use **ResNet-34 or ResNet-50**, pretrained on ImageNet.

**Why ResNet?**
- Skip connections → easier training of deep models  
- Strong fine-grained classification performance  
- Benefits significantly from transfer learning  
- Reliable, stable PyTorch implementation  

---

### **5.2 Transfer Learning Strategy**

#### **Stage 1 — Feature Extraction**
- Freeze all pretrained layers  
- Train only final classification layer (120 breeds)

#### **Stage 2 — Partial Fine-Tuning**
- Unfreeze last 1–2 residual blocks  
- Train with low learning rate

#### **Stage 3 — Full Fine-Tuning (Optional)**
- Unfreeze entire network  
- Use *discriminative learning rates*:  
  - Early layers: 1e−5  
  - Middle layers: 1e−4  
  - Final classifier: 1e−3  

---

### **5.3 Data Augmentation**
- RandomResizedCrop(224×224)  
- Random Horizontal Flip  
- Color Jitter  
- Gaussian blur / rotation  
- MixUp / CutMix for robustness

---

### **5.4 Classification Evaluation Metrics**
- Top-1 accuracy  
- Top-5 accuracy  
- Per-class precision, recall, F1  
- Confusion matrix  
- Hardest breeds analysis  

---

## **6. Component 2: Text-to-Image Dog Generation**

### **6.1 Model Choice — Stable Diffusion / SDXL**

We will use pretrained Stable Diffusion models via **HuggingFace Diffusers**.

---

### **6.2 Prompt Generation**

**Automatic prompts via classifier output:**


**User-controlled prompts:**
"A realistic photo of a Shiba Inu puppy playing in the grass."

---

### **6.3 Output Characteristics**
- 512×512 or 1024×1024 images  
- Multiple samples per request  
- Optional upscaling (4×)

---

## **7. Feedback Mechanism (Self-Improvement)**

### **7.1 Classifier ↔ Generator Feedback Loop**
1. Classifier predicts breed  
2. Generator creates synthetic breed images  
3. Feed synthetic images back into classifier  
4. If classifier misidentifies them:
   - Adjust prompts  
   - Improve augmentation  
   - Add synthetic examples to training  

This helps especially underrepresented breeds.

---

### **7.2 Misclassification Feedback**
- Identify breeds with low F1  
- Generate additional synthetic samples for these breeds  
- Retrain classifier → Validate → Iterate  

---

## **8. System Architecture Diagram**

    ┌────────────────────────────┐
    │        User Input          │
    │  (Image or Breed Prompt)   │
    └──────────────┬─────────────┘
                   ▼
      ┌──────────────────────┐
      │ ResNet Classifier    │
      │   (Transfer Learning)│
      └──────────┬───────────┘
                 ▼
  ┌────────────────────────────────────┐
  │ Text-to-Image Generator (SD/SDXL)  │
  └───────────────┬────────────────────┘
                  ▼
    ┌──────────────────────────────┐
    │  Generated Dog Images (HD)   │
    └──────────────────────────────┘

---

## **9. Evaluation Strategy**

### **Classification**
- Validation accuracy (Top-1, Top-5)  
- Confusion matrix  
- Per-class F1  
- Similar-breed misclassification study  

### **Generation**
- Fidelity to breed characteristics  
- Visual realism  
- Classifier-assisted validation  
- User preference ratings  

---

## **10. Project Timeline**

### **Week 1 — Setup**
- Read ResNet & Stable Diffusion papers  
- Download dataset  
- Implement PyTorch data pipeline  
- Train baseline classifier  

### **Week 2 — Transfer Learning + Diffusion Setup**
- Fine-tune ResNet  
- Implement augmentations  
- Set up Stable Diffusion inference  
- Generate first dog images  

### **Week 3 — Full Integration & Report**
- Connect classifier → generator  
- Add feedback mechanism  
- Evaluate full system  
- Prepare final report & presentation  

---

## **11. Expected Deliverables**
- ResNet dog-breed classifier  
- Stable Diffusion-based dog generator  
- Real-time “upload → predict → generate” pipeline  
- Evaluation report + visuals  
- Gallery of generated dog images  
- Full code with documentation  

---

## **12. References**
- TBD 

---


