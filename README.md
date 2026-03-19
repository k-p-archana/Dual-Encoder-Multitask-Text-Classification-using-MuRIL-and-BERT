# Dual-Encoder-Multitask-Text-Classification-using-MuRIL-and-BERT
This project implements a Dual Encoder Multitask Text Classification system that leverages the complementary strengths of MuRIL (Multilingual Representations for Indian Languages) and Multilingual BERT for robust multilingual understanding. The model is specifically designed to handle multiple classification tasks simultaneously, making it suitable for complex real-world NLP applications involving diverse datasets and labels.

The pipeline begins with Byte Pair Encoding (ByteBPE) tokenization. This step ensures efficient handling of subword units, especially beneficial for morphologically rich and low-resource languages. The tokenized input is then passed to two parallel encoders: MuRIL, which is optimized for Indian languages, and Multilingual BERT, which provides strong general multilingual contextual representations.

Both encoders independently generate contextual embeddings, specifically extracting the [CLS] token representations. These embeddings are then concatenated and passed through a fusion network, which consists of fully connected layers, normalization, and dropout. This late fusion strategy enables the model to combine semantic knowledge from both encoders effectively, resulting in a richer shared representation.

The fused representation is fed into task-specific classification heads, allowing the model to perform different classification tasks (e.g., sentiment analysis, offensive content detection, etc.) within a single architecture. To handle class imbalance and improve generalization, the model incorporates class-weighted cross-entropy loss with label smoothing.

Additionally, the system uses learnable task-specific uncertainty weights, enabling dynamic adjustment of task importance during training. Optimization is performed using AdamW, along with a cosine learning rate scheduler with warm-up, ensuring stable and efficient convergence.

Overall, this project demonstrates an advanced approach to multilingual multitask learning, combining efficient tokenization, dual-encoder fusion, and adaptive training strategies to achieve improved performance across tasks.
