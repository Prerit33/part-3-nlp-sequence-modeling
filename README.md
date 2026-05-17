# part-3-nlp-sequence-modeling

### 1. Why RNNs Struggle with Long-Term Dependencies

Traditional Recurrent Neural Networks (RNNs) process text sequentially, updating their hidden state word by word. However, they suffer from a mathematical issue known as the **vanishing gradient problem**.
When an RNN tries to backpropagate an error across a long sentence (say, 50 words), the gradients (the signals used to update the network's weights) get repeatedly multiplied by values smaller than 1. Eventually, the signal "vanishes" to near zero. As a result, standard RNNs have a short-term memory—by the time they reach the end of a long paragraph, they have completely forgotten the context from the first few sentences.

### 2. How LSTMs Help with Memory

Long Short-Term Memory networks (LSTMs) were invented specifically to solve the vanishing gradient problem. They do this by introducing a **Cell State**—a sort of internal "conveyor belt" that runs straight down the entire chain of the sequence with very few interactions.
To control what goes on or off this conveyor belt, LSTMs use three specialized mathematical structures called **Gates**:

* **Forget Gate:** Decides what irrelevant information to throw away.
* **Input Gate:** Decides what new, relevant information to add.
* **Output Gate:** Decides what information to output for the current step.
This architecture allows an LSTM to explicitly "remember" a crucial subject mentioned at the beginning of a sentence and carry it all the way to the end.

### 3. What Attention Solves in Sequence-to-Sequence Tasks

Before the "Attention" mechanism, models translating text (like English to French) used an Encoder-Decoder architecture. The Encoder would compress the *entire* English sentence into a single, fixed-size mathematical vector (the "bottleneck"), and hand it to the Decoder. For long sentences, this bottleneck was devastating—it's like trying to memorize a whole page of text before you start translating.
**Attention** solved this by eliminating the bottleneck. Instead of a single vector, the model gives the Decoder access to the *entire history* of the input sentence. At every step of translation, the Attention mechanism dynamically calculates "weights," allowing the model to **focus (attend) only on the specific input words that are most relevant** to the word it is currently generating.

### 4. Why Transformers are Important in Modern NLP

Transformers completely threw away the recurrent (RNN/LSTM) structure. Instead of reading words sequentially (one by one), Transformers use a mechanism called **Self-Attention** to read and process *all the words in a sequence simultaneously*.

This is incredibly important for two reasons:

1. **Unprecedented Context:** Self-attention allows every word in a sentence to mathematically weigh its relationship against *every other word* in that sentence, capturing deep, nuanced, long-range context better than any previous model.
2. **Massive Parallelization:** Because they process everything at once rather than sequentially, Transformers can be trained highly efficiently across thousands of GPUs.

This ability to scale training to internet-sized datasets is what made Generative AI and Large Language Models (LLMs) like GPT, BERT, and Gemini possible today.
