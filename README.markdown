# Spam Email Detector LLM

## Overview
This repository contains a spam email detector built using a fine-tuned GPT-2 model (gpt2-small, 124M parameters) to classify text messages as "spam" or "not spam." The model is trained on the [SMS Spam Collection dataset](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip) from the UCI Machine Learning Repository. The project includes data preprocessing, model fine-tuning, and evaluation to achieve high accuracy in spam detection.

## Features
- Downloads and preprocesses the SMS Spam Collection dataset.
- Balances the dataset by undersampling the majority class ("ham").
- Splits data into training (70%), validation (10%), and test (20%) sets.
- Uses a custom `SpamDataset` class for tokenizing and preparing data for the GPT-2 model.
- Fine-tunes a pre-trained GPT-2 model with a binary classification head (spam vs. not spam).
- Evaluates model performance with accuracy and loss metrics.
- Visualizes training and validation loss/accuracy over epochs.
- Provides a function to classify new text inputs as spam or not spam.
- Saves the trained model for reuse.

## Requirements
- Python 3.7+
- Libraries: `torch`, `pandas`, `tiktoken`, `numpy`, `matplotlib`
- Internet access to download the dataset and GPT-2 weights

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/spam-email-detector-llm.git
   cd spam-email-detector-llm
   ```
2. Install dependencies:
   ```bash
   pip install torch pandas tiktoken numpy matplotlib
   ```
3. Ensure the `gpt_download3.py` script is included in the project directory to download GPT-2 weights.

## Usage
1. **Run the Script**:
   Execute the main script to download the dataset, preprocess it, fine-tune the model, and evaluate performance:
   ```bash
   python spam_email_detector.py
   ```
   This will:
   - Download and extract the dataset.
   - Balance and split the dataset.
   - Tokenize data using the GPT-2 tokenizer.
   - Load and fine-tune the GPT-2 model for 4 epochs.
   - Save the trained model as `email_classifier.pth`.
   - Generate plots (`loss-plot.pdf`, `accuracy-plot.pdf`) for training/validation metrics.

2. **Classify New Text**:
   Use the `classify_review` function to classify new messages:
   ```python
   from spam_email_detector import classify_review, model, tokenizer, device, train_dataset

   text = "your bank account got credited by 10000000 rs click on the link to redeem"
   result = classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length)
   print(result)  # Output: spam
   ```

3. **Load a Saved Model**:
   To load the trained model for inference:
   ```python
   import torch
   from spam_email_detector import GPTModel, BASE_CONFIG

   model = GPTModel(BASE_CONFIG)
   model.load_state_dict(torch.load("email_classifier.pth"))
   model.eval()
   ```

## Project Structure
- `spam_email_detector.py`: Main script for data processing, model training, and evaluation.
- `gpt_download3.py`: Utility script to download GPT-2 model weights.
- `train.csv`, `validation.csv`, `test.csv`: Generated files containing preprocessed dataset splits.
- `email_classifier.pth`: Saved model weights after training.
- `loss-plot.pdf`, `accuracy-plot.pdf`: Generated plots for training/validation metrics.

## Model Details
- **Base Model**: GPT-2 small (124M parameters) with a custom classification head.
- **Configuration**:
  - Vocabulary size: 50,257
  - Context length: 1,024
  - Embedding dimension: 768
  - Number of layers: 12
  - Number of attention heads: 12
  - Dropout rate: 0.0
- **Training**:
  - Optimizer: AdamW (learning rate: 5e-5, weight decay: 0.1)
  - Epochs: 4
  - Batch size: 8
  - Fine-tuned on the last transformer block and final layer normalization.

## Results
- **Training Accuracy**: ~98% (after 4 epochs)
- **Validation Accuracy**: ~97% (after 4 epochs)
- **Test Accuracy**: ~96% (after 4 epochs)
- Training time: ~2-5 minutes on a GPU-enabled device (varies by hardware).

## Example Classifications
```python
# Spam example
text_1 = "your bank account got credited by 10000000 rs click on the link to redeem"
print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))
# Output: spam

# Not spam example
text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))
# Output: not spam
```

## Limitations
- Trained on SMS data, which may not fully generalize to other text formats (e.g., emails).
- Small dataset size may limit performance on diverse or complex spam messages.
- Requires computational resources (preferably a GPU) for efficient training.

## Future Improvements
- Experiment with larger GPT-2 models (medium, large) for better performance.
- Incorporate additional datasets for improved generalization.
- Implement cross-validation for robust evaluation.
- Add data augmentation to increase dataset diversity.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, improvements, or new features. Follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- [SMS Spam Collection dataset](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip) from UCI Machine Learning Repository.
- GPT-2 model weights sourced from the Hugging Face model hub via `gpt_download3.py`.
- Built with PyTorch and inspired by open-source NLP tutorials.

## Contact
For issues, questions, or suggestions, please open an issue on this repository or contact the maintainer at <your-email>.

For pricing or subscription details related to xAI products, visit [x.ai/grok](https://x.ai/grok) or [help.x.com](https://help.x.com/en/using-x/x-premium).