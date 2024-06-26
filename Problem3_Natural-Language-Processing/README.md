# Natural Language Processing (a big picture)
Natural Language Processing (NLP), Natural Language Understanding (NLU), and Natural Language Generation (NLG) are all closely related subfields within the broader field of Natural Language Technology (NLT).
# Natural Language Technology (NLT) 
Natural Language Technology (NLT) is an umbrella term that includes 
Natural Language Technology (NLT) is a field that uses various technologies and methodologies for dealing with human language.
- *Machine Learning*: <Br> a method that uses algorithms to learn from data and make predictions or decisions (sentiment analysis, named entity recognition (NER), and text summarization).
- *Deep Learning*: <Br> a subset of machine learning that uses neural networks with multiple layers to learn from data (language generation, machine translation, and text classification).
- *Rule-Based Systems*: <Br> use pre-defined rules and heuristics to analyze and understand the text (part-of-speech tagging (POS-tagging), syntactic parsing, and named entity recognition).
- *Language Modeling*: <Br> a statistical method used to predict the probability distribution of a sequence of words. It is used to train models that can generate text and improve the performance of speech recognition and machine translation.
- *Transfer Learning*: <Br> a method that involves pre-training a neural network on a large dataset and then fine-tuning it on a smaller dataset for a specific task. This approach can improve task performance such as sentiment analysis, named entity recognition, and language translation.
- *Reinforcement Learning*: <Br> a method in which an agent learns by taking actions in an environment to maximize a reward. It is used for tasks such as dialog systems and text generation. <Br>

Natural Language Processing (NLP) is a critical subset of NLT, providing the core techniques for understanding and generating text and speech.

---
### Natural Language Processing (NLP)
**Natural Language Processing (NLP)** focuses on enabling computers to understand human language in both written and verbal forms. It involves machine learning and deep learning techniques. (1) Tokenization, (2) Embedding, and (3) Model architectures, are the three main components that help machines understand natural language. Tokenization is the first step in natural language processing (NLP) projects. It involves dividing a text into individual units, known as tokens. Tokens can be words or punctuation marks. These tokens are then transformed into numerical vectors representing words. Two main concepts are vectorization and embedding. Text Vectorization is the process of turning words into numerical vectors in a one-dimensional space. Word Embedding (Word Vector) is a type of vectorization through deep learning as dense vectors in a high-dimensional space. Then, these numerical vectors will be fed to models and processed for specific tasks.
#### 1. Natural Language Processing Pipelines:
##### 1.1 Preprocessing:
Includes tokenization, stemming, lemmatization, and stop-word removal.
- Tokenization
  - [Sentence based](https://colab.research.google.com/drive/1CWyJzCyD1pc2u0JXG4C0NFyZtcJjTqK6#scrollTo=Kf2RKlgKzsro)
  - [Word based](https://colab.research.google.com/drive/1CWyJzCyD1pc2u0JXG4C0NFyZtcJjTqK6#scrollTo=Kf2RKlgKzsro)
  - [Character based](https://colab.research.google.com/drive/1CWyJzCyD1pc2u0JXG4C0NFyZtcJjTqK6#scrollTo=Kf2RKlgKzsro)
  - Subwords based
    - [BPE Algorithm – a Frequency-based Model](https://colab.research.google.com/drive/1CWyJzCyD1pc2u0JXG4C0NFyZtcJjTqK6#scrollTo=Kf2RKlgKzsro)
    - [WordPiece Algorithm](https://colab.research.google.com/drive/1CWyJzCyD1pc2u0JXG4C0NFyZtcJjTqK6#scrollTo=Kf2RKlgKzsro)
    - [Unigram Algorithm – a Probability-based Model](https://colab.research.google.com/drive/1CWyJzCyD1pc2u0JXG4C0NFyZtcJjTqK6#scrollTo=Kf2RKlgKzsro)
- [Preprocessing using spaCy](https://colab.research.google.com/drive/1VChFh5pcY-yKzXWCIHUX6DNbMmgT0RoK?usp=sharing)
- [Preprocessing using nltk](https://colab.research.google.com/drive/1aoimloh3KtrrkjrkgG9MlU1U_xA65lwn?usp=sharing)
##### 1.2. Feature Extraction: 
Convert text to numerical representations.
- Text Vectorization:
  - Traditional approach
    - [One-Hot Encoding](https://colab.research.google.com/drive/1oljXnBEnNFsWQBR1ebl-u_C8bhM22-Ik?usp=sharing)
    - [Bag of Words](https://colab.research.google.com/drive/1oljXnBEnNFsWQBR1ebl-u_C8bhM22-Ik?usp=sharing)
    - [CountVectorizer](https://colab.research.google.com/drive/1oljXnBEnNFsWQBR1ebl-u_C8bhM22-Ik?usp=sharing)
    - [TF-IDF](https://colab.research.google.com/drive/1oljXnBEnNFsWQBR1ebl-u_C8bhM22-Ik?usp=sharing)
- Word Embedding (Word Vector):
  - Context-independent
    - Neural Word Embedding
      - [Word2Vec](https://colab.research.google.com/drive/1_YijyM-9mHKujbW5_PoDdvks1k5rMSt_?usp=sharing)
    - Pretrained Word-Embedding
      - [GloVe](https://colab.research.google.com/drive/1_YijyM-9mHKujbW5_PoDdvks1k5rMSt_?usp=sharing)
      - [FastText](https://colab.research.google.com/drive/1_YijyM-9mHKujbW5_PoDdvks1k5rMSt_?usp=sharing)
  - Context-dependent
    - RNN based
      - [ELMO](https://colab.research.google.com/drive/1MkNmcfM6Pfb2yzTUYIcPSyvTFH2Ubpnd?usp=sharing)
      - [CoVe](https://colab.research.google.com/drive/1RkPi17NjwRUOzXAsT5uUFZqBBQQPkP8Z?usp=sharing)
    - Transformer based
      - [BERT](https://colab.research.google.com/drive/1sJ6umr6JsH2VmV2gmmyq7YVMMspktYri?usp=sharing)
      - XML
      - RoBERTa
      - ALBERT
- [Document Embedding](https://colab.research.google.com/drive/1Vq4nFMg58rvDy_pCLJv5VVDHczCrKNOX?usp=sharing)
  - Doc2Vec
    - Distributed Memory (DM)
    - Distributed Bag of Words (DBOW)
##### 1.3. Modeling: 
Use machine learning or deep learning models to process data for specific tasks. The main model architectures include:
- **Recurrent Neural Networks (RNNs)**: Good for sequential data.
  - ***Long Short-Term Memory (LSTM)***
  - ***Gated Recurrent Units (GRUs)***
- **Convolutional Neural Networks (CNNs)**: Effective for capturing local patterns.
- **Transformers**: State-of-the-art models like BERT and GPT that excel at capturing long-range dependencies in text.
##### 1.4. Post-processing: 
Refine and format the model's output for the end application.        
#### 2. Fundamental sub-fields of Natural Language Processing (NLP):
- **Speech Recognition**: Converting spoken language into text.
- **Natural Language Understanding (NLU)**: Understanding the meaning of text.
- **Natural Language Generation (NLG)**: Producing human-like text based on input data.
- **Text-to-Speech (TTS)**: Converting text back into spoken language

---
### Natural Language Understanding (NLU)
**Natural Language Understanding (NLU)** deals with the ability of computers to understand the meaning and context of human language and allows machines to comprehend text in a way similar to how humans do.
#### 1. Core functionalities:
- ***Extracting Meaning***: aims to understand the intent, sentiment, and overall message conveyed in a sentence or passage.
- ***Analyzing Context***: considers the context surrounding the language. For instance, "the bank is closed" can have different meanings depending on the context (financial institution vs. riverbank).
- ***Disambiguation***: Language can be ambiguous at times. NLU tackles situations where a word or phrase can have multiple meanings.

#### 2. How NLU works:
- ***Breaking it down***: break down the input text into smaller components like words, phrases, and sentences.
- ***Understanding Relationships***: analyzes the relationships between these components. This involves part-of-speech tagging (identifying nouns, verbs, etc.) and recognizing grammatical structures.
- ***Deriving Meaning***: Based on the analysis, infer the meaning of the text. This might involve considering the context in which the language is used.

#### 3. Common NLU tasks:
- ***Text Classification***: automatically categorizing a text into predefined categories. Examples: sentiment analysis (positive, negative, neutral), spam filtering (spam, not spam), or topic labeling (sports, politics, entertainment).
  - [Sentence Classifier (simple sentences, compound sentences, complex sentences, passive sentences) using Dependency parsing (Trankit)](https://colab.research.google.com/drive/1T76FLqelbCPs59RjpeiwqO3d7cjaLaAb?usp=sharing)
- ***Named Entity Recognition (NER)***: identifies and classifies named entities within a text, such as people, organizations, locations, dates, monetary values, etc for information extraction or question answering.
- ***Text Summarization***: generate a concise summary of a long text while preserving the key points and meaning. Summarization can be beneficial for quickly grasping the main idea of an article or document.
- ***Machine Translation***: plays a vital role in machine translation, where the system understands the source language (e.g., English) and translates it accurately into the target language (e.g., Spanish) while preserving the meaning and intent.
- ***Question Answering***: answer questions for information retrieval from a knowledge base or open-ended, more challenging questions that require reasoning and inference.
- ***Part-of-Speech Tagging (POS)***: assigns grammatical labels (e.g., noun, verb, adjective) to each word in a sentence. POS tagging is a fundamental step in NLU, providing valuable information about the sentence structure that can be used for various downstream tasks.
  - [Part-Of-SpeechTagging using Hidden MarkovModel](https://colab.research.google.com/drive/1h_DSf0bV5d0kHEpFeWkpCIC3ghSjA3_O?usp=sharing)
- ***Sentiment Analysis***: goes beyond simple classification (positive, negative) and aims to understand a text's emotional tone or opinion. This can involve analyzing the sentiment of product reviews, social media posts, or customer feedback data.
- ***Dialogue Management***: in chatbots or virtual assistants. It allows the system to understand the user's intent within a conversation, track the conversation flow, and generate appropriate responses.
- ***Textual Entailment***: determines whether the meaning of one sentence (hypothesis) is entailed by the definition of another sentence (text). It requires the NLU system to understand the logical relationships between sentences.
- ***Natural Language Inference***: involves reasoning about the relationship between two sentences (Similar to textual entailment). The system determines if the second sentence (hypothesis) can be inferred from the first sentence (premise).
    
---
### Natural Language Generation (NLG)
- ***Natural Language Generation (NLG)***: aim to convert ***structured data/input prompt*** into ***coherent and contextually relevant natural language text*** such as:
    - ***Report Generation***: Creating financial reports, weather summaries, and business insights based on raw data.
    - ***Personalized Communication***: Creating emails, customer service responses, and product descriptions.
    - ***Data Narratives***: Translating complex data sets into easily understandable narratives (business intelligence/analytics).
  - The steps within an NLG system:
    - ***Content analysis***: analyzes the information that will be transformed into text, often focusing on understanding the meaning and themes within existing textual data.
    - ***Data understanding***: a crucial initial step, where the system grasps the meaning and structure of the non-linguistic data it needs to convert to text.
    - ***Document structuring***: involves organizing the information from the data into a logical document structure like sections, paragraphs, or bullet points.
    - ***Sentence aggregation***: combine or condense information from various data points into concise sentences.
    - ***Grammar structuring***: is essential for generating human-like text. NLG systems employ grammar rules to ensure proper sentence structure.
    - ***Language presentation***: tailoring the text formality to the audience or stylistic elements depending on the NLG application.
- ***Text Generation***: involve generating ***human-like text*** based on a given ***input or prompt*** across various domains such as:
  - ***Creative Writing***: Generating poems, stories, and other forms of creative text.
  - ***Dialogue Systems***: Creating responses for chatbots and conversational agents.
  - ***Predictive Text***: Autocomplete features in smartphones and text editors.
  - ***Content Creation***: Assisting in the generation of articles, summaries, and reports.

- ***Text generation models***:
There are two main categories of text generation Models:
  - **1. Rule-based models**: rely on predefined rules and templates, offering control and explainability but limited creativity.
  - **2. Statistical models**: learn patterns from large amounts of text data:
    - **2.1 Traditional statistical models**: Techniques like ***n-grams***, ***Hidden Markov Models (HMMs)***, and ***Conditional Random Fields (CRFs)***.
    - **2.2 Neural statistical models**: Includes Bengio's ***Neural Probabilistic Language Model (NPLM)***
    - **2.3 Deep Learning models**: A ***specific type*** of the statistical model using ***deep learning architectures*** includes ***Transformers*** and ***Large Language Models (LLMs)*** respectively. While transformers share ***some characteristics with statistical models***, they are a ***more advanced architecture***. They use ***deep learning techniques*** with ***artificial neural networks*** to process information. However, they are still trained on ***massive amounts*** of text data, and their outputs are based on the ***statistical patterns*** learned from that data. LLMs are built using Transformers (often) or other deep learning architectures to train specifically for NLG tasks and excel at generating creative and human-like text.

  ##### 1. Rule-based models:
  - ***Predefined Rules***: Linguists and domain experts craft a set of rules that govern how the text is structured and phrased. These rules cover aspects like grammar, sentence structure, and word choice.
  - ***Pattern Matching***: The model analyzes the situation or input and identifies patterns that match its rule set.
  - ***Template Selection***: Based on the matched pattern, the model selects a pre-defined template for the text generation. These templates can be simple phrases or more complex structures with placeholders.
  - ***Filling the Blanks***: The model fills the placeholders in the chosen template with specific words based on the rules and any available data.
    
  ##### 2. Statistical models:
  - **2.1 Traditional statistical models**: <Br>
  Rely on ***statistical analysis*** of data to learn ***patterns*** and make ***predictions***. This category encompasses techniques like ***n-grams***, ***Hidden Markov Models (HMMs)***, and ***Conditional Random Fields (CRFs)*** used for text generation.
    - [N-grams](https://colab.research.google.com/drive/1lgbXAnXiJV26cegl8H5pUnO8dvEQYYbW?usp=sharing)
    - Hidden Markov Models (HMMs)
    - Conditional Random Fields (CRFs)
  - **2.2 Neural statistical models**: This sub-category includes Bengio's NPLM. While NPLM utilizes some neural network concepts, it primarily relies on statistical learning from vast amounts of text data. It represents a stepping stone between traditional statistical models and deep learning approaches.
    - [Neural Probabilistic Language Model (NPLM)](https://colab.research.google.com/drive/16GZscemOE5ecJbE6iutodr412XZmSw7l?usp=sharing)
  - **2.3 Deep Learning models**: <Br>
    - **2.3.1 Transformers**: <Br> can be considered as ***Statistical models in a broader sense*** because they leverage statistical learning from vast amounts of text data. Besides, they incorporate deep neural networks for more complex pattern recognition and improved performance over traditional statistical techniques as ***A sub-category of statistical models with deep learning elements***.
      - **Decoder-Only Transformers** (e.g., GPT-n): focus solely on the decoder part, excelling at text generation tasks like creating stories or poems. They process information sequentially, predicting the next word based on the previous ones. However, they may not be ideal for tasks requiring full context understanding, such as question answering.
      - **Encoder-Decoder Transformers** (e.g., T5): combine encoder and decoder parts. The encoder encodes the input sentence, and the decoder uses that encoded representation to generate the output. These models are often used for machine translation, where the model's mission is to understand the source language (encode) and generate the target language (decode).
    - **2.3.2 Large Language Models (LLMs)**: <Br> A deep learning model specifically trained on massive amounts of text data allows them to learn complex statistical relationships between words and generate human-quality text as a response to a prompt or question.
      - **Pretrained-model**: include Closed-Source Models which are powerful, state-of-the-art models like GPT-3, Gemini, and Claude... available via commercial APIs or specific platforms, and Open-Source Models which are widely accessible and customizable models like BERT, GPT-2, RoBERTa, and T5, available for download and fine-tuning.
      - **Prompt**: plays a crucial role in using prompting engineering techniques to guide the models (LLMs) to generate desired outputs.
      - **Open-source framework**: is designed to facilitate the development of applications powered by large language models (LLMs). It streamlines the process includes integrating and managing LLMs in various applications, enabling developers to build sophisticated AI systems more efficiently. Its benefits include (1) No licensing fees, reduced financial burden, (2) Open source code allows inspection and fosters trust, (3) Enables community contributions and faster advancement, (4) Adaptable to specific needs and allows for fine-tuning, (5) Provides learning opportunities for developers and researchers...
        - **LangChain**: focuses on prompt engineering, multi-step workflows, and abstraction of LLM integration
        - **Hugging Face**: emphasizes model access, fine-tuning, and community collaboration

---
#### Large Language Models (LLMs):
##### 1. Prompt: 
Play a crucial role in using prompting engineering techniques to guide the models (LLMs) to generate desired outputs.
- **Basic prompting engineering techniques**: fundamental strategies used to design prompts
  - ***Task Definition***: Clearly define the task or objective to guide the model's behavior.
  - ***Context Establishment***: Provide relevant background information or context to help the model understand the task.
  - ***Example-Based Prompting***: Include examples or instances to guide the model's understanding.
  - ***Instructional Prompts***: Give clear instructions or directives to guide how the model approaches the task.
  - ***Constraint Setting***: Impose constraints or limitations within the prompt to guide the model's output.
  - ***Feedback Incorporation***: Incorporate feedback mechanisms to improve the model's performance over time.
  - ***Progressive Prompting***: Break down complex tasks into smaller, more manageable steps and provide progressive prompts.
  - ***Relevance Emphasis***: Emphasize key concepts or information to guide the model's attention.
- **Advanced prompting engineering techniques**: enable more complex tasks and enhance the reliability and performance of large language models (LLMs)
  - ***Zero-shot Prompting***: rely solely on the prompt, then generate responses without specific examples or training on a task.
  - ***Few-shot Prompting***: use a small number of examples or prompts to perform a task, facilitating generalization with limited supervision.
  - ***Chain-of-Thought Prompting***: use a logical sequence of steps to solve a problem or answer a question.
  - ***Self-Consistency***: encourages the model to generate consistent responses within a context.
  - ***Generate Knowledge Prompting***: generate new knowledge or information based on the prompt.
  - ***Prompt Chaining***: chains multiple prompts together, where the output of one prompt serves as the input for the next.
  - ***Tree of Thoughts***: organizes prompts and responses hierarchically, facilitating structured interactions.
  - ***Retrieval Augmented Generation***: integrates retrieval-based methods with generative models to enhance response relevance and coherence. Prompting techniques typically focus on how inputs are formulated to guide the models (LLMs) to generate desired outputs. The retrieval part of RAG contributes to creating a meaningful context, then the generation part can be based on the context to generate desired outputs. This makes RAG can be classified as a prompting technique. But with two components retrieval and generation parts, it can more accurately be categorized as a model architecture or approach.
  - ***Automatic Reasoning and Tool-use***: enables to perform logical reasoning and use external tools or resources to solve problems.
  - ***Automatic Prompt Engineer***: automates prompt generation using reinforcement learning or optimization algorithms.
  - ***Active-Prompt***: dynamically adjusts prompts based on model performance and feedback, iteratively improving response quality.
  - ***Directional Stimulus Prompting***: guides the attention towards specific aspects of the input prompt.
  - ***Program-Aided Language Models***:integrates programming constructs within prompts to guide behavior.
  - ***ReAct (Reflective Active Learning)***: uses feedback loops to refine prompts and improve performance over time.
  - ***Reflexion***: Encourages self-awareness and reflection in the model's reasoning process.
  - ***Multimodal CoT***: extends Chain-of-Thought prompting to incorporate multimodal inputs.
  - ***Graph Prompting***: represents prompts and responses as graphs, facilitating efficient processing and manipulation.

##### 2. Pretrained models:
- **Closed-Source Models**:
These models are developed by organizations that do not publicly release the model weights or code. They are usually accessible through APIs or specific platforms.
  - GPT-n (OpenAI): A powerful language model with billions of parameters, available via the OpenAI API.
  - Gemini (Google DeepMind): Advanced models are typically not open-sourced but accessible through Google's platforms.
  - Claude (Anthropic): An advanced language model accessed via specific platforms or APIs.
  - ...
- **Open-Source Models**:
These models are publicly released with their weights and code, allowing anyone to download, fine-tune, and deploy them.
  - BERT (Google): Bidirectional Encoder Representations from Transformers, effective for many NLP tasks.
  - GPT-2 (OpenAI): The predecessor to GPT-3, smaller but still very capable, available for download and customization.
  - RoBERT-a (Facebook AI): A robustly optimized version of BERT, designed to improve performance on NLP tasks.
  - ...

##### 3. Capabilities of LLMs: 
Some of the key capabilities of text generation models include:
- ***Natural Language Understanding***: understand and process natural language input, allowing them to interpret user queries, commands, or prompts effectively.
- ***Contextual Understanding***: captures contextual information from the input text, enabling the generation of responses or outputs that are contextually relevant and coherent.
- ***Creativity and Flexibility***: exhibit creativity and flexibility in generating diverse and novel text outputs (text in different styles, tones, or voices suitable for creative writing tasks).
- ***Multi-Modal Generation***: generate text by combining with other modalities such as images, audio, or video, enabling multi-modal content generation for tasks like image captioning or video summarization.
- ***Multi-Lingual Support***: generate text in multiple languages for translation tasks, multilingual content generation, or cross-lingual applications.
- ***Fine-grained control***: allows users to steer the generation towards desired outcomes by offering fine-grained control over the generated outputs by conditioning the generation process on specific attributes, constraints, or prompts.
- ***Adaptability and Customization***: adapt to different contexts and requirements by fine-tuning or customizing for specific tasks, domains, or applications through transfer learning or fine-tuning.
- ***Long-Range Dependencies***: capture complex linguistic structures and relationships by effectively handling long-range dependencies in the input text.
- ***Semantic Understanding***: understand the semantic meaning of the input text, enabling them to generate semantically meaningful responses aligned with the input.
- ***Interactive Generation***: support interactive generation, where users can provide feedback or guidance in the generation process to steer the outputs toward desired directions.
  
##### 4. Performance-enhanced LLMs in domain-specific knowledge:
While LLMs possess impressive capabilities, fine-tuning remains valuable for several reasons:
- ***Task-Specific Adaptation***: Fine-tuning allows models to adapt to specific tasks or domains, optimizing their performance for particular applications.
- ***Domain-Specific Knowledge***: Fine-tuning enables models to learn domain-specific nuances and vocabulary, enhancing the relevance and accuracy of generated outputs.
- ***Bias Mitigation***: Fine-tuning provides an opportunity to address biases present in pre-trained models, promoting fairness and inclusivity in the generated text.
- ***Improved Performance***: Fine-tuning leads to significant improvements by optimizing models for specific objectives or datasets.
- ***Customization and Control***: Fine-tuning offers greater customization and control over model behavior, allowing users to tailor outputs to meet specific requirements or preferences.

A general process of fine-tuning:
- ***Data Preparation***: Gather and preprocess relevant data.
- ***Choose Model***: Select a pre-trained model.
- ***Initialization***: Load and initialize the model.
- ***Fine-tuning***: Train the model on task-specific data.
- ***Evaluation***: Assess model performance.
- ***Refinement***: Adjust as needed.
- ***Deployment***: Use the fine-tuned model for text generation tasks.
  
##### 5. Applications of LLMs:
There is 1 special point, the LLMs are built using Transformers (often) or other deep learning architectures to train for NLG tasks. However, they are highly versatile and can be fine-tuned for various applications beyond traditional NLG tasks. Therefore some tasks typically fall outside the scope of Natural Language Generation (NLG) tasks but inside the scope of LLMs. For code generation tasks (which fall outside the scope of NLG), LLMs can be trained to understand programming languages and generate code snippets or scripts based on input prompts. This can be particularly useful for automating software development tasks, providing code suggestions, or assisting programmers in writing code more efficiently.
- ***Language translation***
- ***Text summarization***
- ***Content generation***
- ***Chatbots and virtual assistants***
- ***Question answering***
- ***Creative writing***
- ***Language modeling***
- ***Code generation***
- ***Text editing and revision***
- ***Knowledge discovery and exploration***
  
---
### Retrieval-Augmented Generation (RAG)
#### 1. Retrieval-Augmented Generation (RAG)
Instead of only using generating models, Retrieval-augmented generation (RAG)  first understands queries (as inputs). It creates results (as outputs) by combining a model retrieving relevant information and a model generating text from that information. By combining, a model’s strengths compensate for the other’s weaknesses to output the best-suited responses. <Br>
Flow: **Question + Retrieval-Augmented Generation (RAG) => Answer** <Br>
Retrieval-Augmented Generation (RAG) components <Br>
- Retrieval
  - [Query (or question) Translation](https://colab.research.google.com/drive/1gYN-LV3rjpX4YnOQ4I5_qvrB7vP6Rv0_?usp=drive_link)
  - [Routing](https://colab.research.google.com/drive/1GVUJU9ViVF6Pt5eJfPxq2BXBlQn1r0Tl?usp=drive_link)
  - [Query Construction](https://colab.research.google.com/drive/1tPeU7WWlM4Z4DkXXtmiC3S_toQagGBFW?usp=drive_link)
  - [Indexing](https://colab.research.google.com/drive/10t6wsGxWT4Yd8pOLYRgUZ7gbW12njMcd?usp=drive_link)
  - [Retrieval](https://colab.research.google.com/drive/1c8KAuqoc7WRyQvv7hlugyBvT_jYPxKzs?usp=drive_link)
- Generation
  - [Generation](https://colab.research.google.com/drive/1aJYZcQCsVia4Vay7tT53mRthgsmARwUY?usp=drive_link)
    
#### 2. Retrieval Augmented Fine-Tuning (RAFT)
Retrieval Augmented Fine-Tuning (RAFT) is a performance-enhanced RAG in domain-specific knowledge. <Br>
Flow: **RAG + Fine-Tuning => RAFT** <Br>
Retrieval Augmented Fine-Tuning (RAFT) components
- Retrieval-Augmented Generation (RAG)
- [Fine-Tuning](https://colab.research.google.com/drive/1FKaIwp_7ZugNRBAk0MBEuLSXvlbKz_Nb?usp=drive_link)

---
### Vision-Language Models (VLMs)
The Bridge Between Vision and Language
#### 1. Motivation
- **Achieving Human-like Perception**: VLMs aim to mimic the brain connecting what we see (image) with how we describe it (language), thereby allowing machines to process visual data and associated language.
- **Expanding Capabilities of AI**:  by incorporating textual information, VLMs can achieve a richer understanding of the visual world to grasp the context of an image where traditional computer vision models are having difficulty.
- **Revolutionizing Downstream Tasks**:  answer questions about an image's content (VQA), generate captions that describe the scene (Image Captioning), or even search for similar images based on a text description (TBIR) will transform various fields like image retrieval, image understanding, and human-computer interaction.
- **Understanding the Real World**: learn more about the world around us by combining visual and textual data in tasks like autonomous vehicles, robotics, and image-based medical diagnosis.
#### 2. Breaking down
Vision-language models (LLMs) integrate visual and textual information to perform tasks involving both modalities. 
- **Inputs**:
  - ***Images***: Image (Raw or preprocessed versions as resized, and normalized...).
  - ***Text***: Natural language text (a sentence, a phrase, or a question).
- **Feature Extraction**:
  - ***Visual Features Extraction for Images***: using a convolutional neural network (CNN) including ResNet, EfficientNet, or a vision transformer (ViT). 
  - ***Textual Features Extraction for Texts***: as tokenized and embedded form using models like BERT, GPT, or other transformer-based architectures to capture the semantic meaning of the text.
- **Multimodal Fusion**: Combines visual and textual features through concatenation, attention mechanisms, or more sophisticated cross-modal transformers. Example: ViLBERT uses two separate transformers for image and text, with cross-attention layers to allow information exchange between the two modalities.
- **Training**: using large datasets that include paired images and text (e.g., image-caption pairs) and optimizing the model to minimize the difference between predicted outputs and the actual labels in the dataset. Training objectives vary depending on specific tasks (desired outputs), including contrastive learning, masked language modeling, image-text matching, or task-related losses (e.g., classification or regression).
  - ***Loss function in VLMs***: a crucial role to understand and integrate visual and textual data effectively. Depending on the specific task and model architecture, different types of loss functions can be used.
    - Contrastive Loss (Image-Text Matching)
    - Cross-Entropy Loss (VQA, Image Captioning)
    - Masked Language Modeling (MLM) Loss (Multimodal Transformers)
    - Regression Loss (Visual Grounding, Image-Text Retrieval)
    - Combined Loss Functions (multiple tasks simultaneously)
- **Outputs**: Depending on the task, different heads can be used to produce the final output.
  - **Textual Outputs**: 
    - ***Image Captioning***: a textual description of the given image using a sequence generation head (transformer decoder or LSTM).
    - ***Visual Question Answering (VQA)***: answer a question about an image, using the classification head to predict the correct answer.
  - **Numerical or Categorical Outputs**:
      - ***Image-Text Matching***: score or probability indicating the relevance of a textual description to an image, using similarity scoring head to calculate a score or probability.
      - ***Visual Grounding***: coordinates or bounding boxes in the image corresponding to a textual description, using regression head to predict bounding box coordinates
  - **Visual Outputs**:
    - ***Text-to-Image Generation***: image generated based on a textual prompt (like DALL-E), using image generation head (GANs or diffusion models).
  - **Retrieval Outputs**:
    - ***Image Retrieval***: list of images that match a given textual description, using retrieval head using similarity scores or embeddings.
    - ***Text Retrieval***: list of text descriptions that match a given image using a sequence generation head (transformer decoder or LSTM).