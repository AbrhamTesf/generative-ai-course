# Week 3: Fine-tuning Foundation Models

## 📚 Lecture Overview

This week we explore the art and science of fine-tuning foundation models. You'll learn about different fine-tuning techniques, from full fine-tuning to parameter-efficient methods like LoRA and QLoRA, and how to prepare and optimize your training data.

## 🎯 Learning Objectives

By the end of this week, you will:
- Understand the fundamentals of model fine-tuning
- Master parameter-efficient fine-tuning methods (LoRA, QLoRA)
- Learn data preparation and preprocessing techniques
- Implement fine-tuning workflows with best practices
- Evaluate and deploy fine-tuned models

## 📖 Lecture Content

### 1. Understanding Model Fine-tuning

**What is Fine-tuning?**
- Adapting pre-trained models to specific tasks or domains
- Transfer learning for language models
- Benefits: Better performance, domain adaptation, task specialization

**Types of Fine-tuning:**

**Full Fine-tuning:**
```python
# Updates all model parameters
model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = AdamW(model.parameters(), lr=5e-5)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

**Parameter-Efficient Fine-tuning (PEFT):**
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- AdaLoRA
- Prefix Tuning
- Prompt Tuning

### 2. LoRA (Low-Rank Adaptation)

**LoRA Concept:**
- Freezes pre-trained model weights
- Injects trainable rank decomposition matrices
- Reduces trainable parameters by 10,000x
- Maintains model quality

**LoRA Implementation:**
```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Which layers to adapt
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 1,769,472 || all params: 124,439,808 || trainable%: 1.42%
```

**LoRA Training:**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
```

### 3. QLoRA (Quantized LoRA)

**QLoRA Benefits:**
- 4-bit quantization for memory efficiency
- Enables fine-tuning on consumer hardware
- Maintains model performance
- Reduces memory usage by 75%

**QLoRA Implementation:**
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply QLoRA
model = get_peft_model(model, lora_config)
```

### 4. Data Preparation

**Data Formatting:**
```python
def format_instruction_data(instruction, input_text, output_text):
    """Format data for instruction fine-tuning"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    }

# Example data
data = [
    format_instruction_data(
        "Translate to French",
        "Hello, how are you?",
        "Bonjour, comment allez-vous?"
    ),
    format_instruction_data(
        "Summarize the text",
        "Long text here...",
        "Summary here..."
    )
]
```

**Data Preprocessing:**
```python
from transformers import AutoTokenizer

def preprocess_function(examples):
    """Tokenize and prepare data for training"""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Create labels (same as inputs for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)
```

### 5. Training Strategies

**Hyperparameter Tuning:**
```python
# Learning rate scheduling
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=2e-4)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)
```

**Gradient Accumulation:**
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 16
    max_grad_norm=0.3,  # Gradient clipping
    warmup_ratio=0.1,
    weight_decay=0.01,
)
```

**Mixed Precision Training:**
```python
training_args = TrainingArguments(
    fp16=True,  # 16-bit precision
    bf16=False,  # or use bfloat16
    dataloader_pin_memory=False,
)
```

### 6. Evaluation and Monitoring

**Evaluation Metrics:**
```python
from evaluate import load
import numpy as np

# Load metrics
rouge = load("rouge")
bleu = load("bleu")

def compute_metrics(eval_preds):
    """Compute evaluation metrics"""
    predictions, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE
    rouge_scores = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Compute BLEU
    bleu_score = bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_score["bleu"]
    }
```

**Training Monitoring:**
```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: loss = {logs.get('loss', 0):.4f}")

# Add callback to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[CustomCallback()]
)
```

### 7. Model Deployment

**Saving and Loading:**
```python
# Save the fine-tuned model
trainer.save_model("./my-finetuned-model")

# Load the model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model = PeftModel.from_pretrained(base_model, "./my-finetuned-model")
```

**Inference:**
```python
def generate_response(prompt, max_length=100):
    """Generate response using fine-tuned model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```

## 🛠️ Workshop Preparation

**Pre-workshop Tasks:**
1. Set up GPU environment (if available)
2. Install PEFT: `pip install peft bitsandbytes`
3. Prepare a small dataset for fine-tuning
4. Review Hugging Face Transformers documentation

**Workshop Goals:**
- Implement LoRA fine-tuning on a small model
- Prepare and format training data
- Monitor training progress and metrics
- Deploy and test fine-tuned model

## 📚 Additional Resources

**Reading Materials:**
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

**Videos:**
- [LoRA Fine-tuning Tutorial](https://www.youtube.com/watch?v=YQpV7WVTJXE)
- [QLoRA Explained](https://www.youtube.com/watch?v=2xK74MqYvdg)
- [Fine-tuning Best Practices](https://www.youtube.com/watch?v=1pedAIvTWXk)

**Tools & Platforms:**
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [Google Colab](https://colab.research.google.com/) - Free GPU access

## 📝 Assignment

**Due Date**: End of Week 3

**Tasks:**
1. Fine-tune a small language model using LoRA
2. Prepare a custom dataset for your use case
3. Implement evaluation metrics
4. Deploy the fine-tuned model

**Submission Requirements:**
- Fine-tuned model and training code
- Dataset preparation pipeline
- Evaluation results and analysis
- Documentation of the fine-tuning process

## 🎯 Next Week Preview

**Week 4: Multimodal AI Applications**
- Working with text, image, audio, and video models
- Building multimodal AI systems
- Cross-modal understanding and generation
- Real-world multimodal applications

## 💡 Advanced Concepts

**Multi-task Fine-tuning:**
- Training on multiple tasks simultaneously
- Task-specific adapters
- Continual learning strategies
- Catastrophic forgetting prevention

**Model Compression:**
- Knowledge distillation
- Pruning techniques
- Quantization methods
- Model architecture search

## 🚀 Best Practices

**Data Quality:**
- Clean and validate your training data
- Ensure data diversity and representativeness
- Handle class imbalance and bias
- Use data augmentation techniques

**Training Efficiency:**
- Start with small models and datasets
- Use appropriate learning rates
- Monitor for overfitting
- Implement early stopping

**Evaluation:**
- Use multiple evaluation metrics
- Test on held-out data
- Perform human evaluation
- Monitor for bias and fairness

## 📞 Support

- **Element Support**: `#support`
- **Office Hours**: TBD
- **Email**: course-instructor@example.com

---

*Remember: Fine-tuning is both an art and a science. Experiment, iterate, and learn from your results!* 