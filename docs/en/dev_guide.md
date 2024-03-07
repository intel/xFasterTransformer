# xFasterTransformer Development Guide  

This document describes how to add support for a new LLM model in xFasterTransformer, which is a highly optimized LLM inference framework on Xeon.  
Generally, to incorporate support for a new model, we need to:  
-	Comprehend the structure of the model. For instance, we need to determine whether Attention and MLP are arranged in serial or in parallel, what type of positional embedding is used, and so on.
-	Delve into some details for the implementation. This includes understanding what type of attention mask is used. While most models utilize a casual attention mask, some may differ. We also need to ascertain whether the residual input is the value before or after normalization.
-	Select a proper data type to balance the accuracy and performance.  

Below content give more details for how to add the support, which is organized into preparing, implementation and verification.
## 1. Preparing: Get familiar with xFT and the new LLM model
### 1.1 How xFT works
### 1.2 New model investigation

## 2. Implementation: Add code in xFT for the new model

### 2.1 Code for Decoder Block
#### 2.1.1 Attention
#### 2.1.2 MLP (FFN)
### 2.2 Code for inference logic before Decoder Block
### 2.3 Coding for inference logic after Decoder Block
## 3. Verification: Debug accuracy and performance
### 3.1 Correctness checking
### 3.2 Performance debug
