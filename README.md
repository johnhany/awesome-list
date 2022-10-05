# Awesome List
A list of useful stuff in Machine Learning, Computer Graphics, Software Development, Mathematics, ...

> For specific paper or algorithm, refer to
> [Paper & Algorithm list](/paper_algorithm_list.md)

> For machine learning datasets, refer to
> [Dataset List](dataset_list.md)

> References:
> [github.com/ml-tooling/best-of-ml-python](https://github.com/ml-tooling/best-of-ml-python)

---

# Table of Contents

- [Machine Learning](#machine-learning)
  - [Deep Learning Framework](#deep-learning-framework)
  - [Machine Learning Framework](#machine-learning-framework)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Graph](#graph)
  - [Time-Series & Financial](#time-series--financial)
  - [Linear Algebra / Statistics Toolkit](#linear-algebra--statistics-toolkit)
  - [Data Processing](#data-processing)
  - [Data Visualization](#data-visualization)
  - [Machine Learning Tutorials](#machine-learning-tutorials)
- [Full-Stack Development](#full-stack-development)
  - [Web Development](#web-development)
  - [Data Management & Processing](#data-management--processing)
- [Academic](#academic)
  - [Mathematics](#mathematics)
  - [Paper Reading](#paper-reading)
- [Useful Tools](#useful-tools)
  - [MacOS](#macos)
  - [Cross-Platform](#cross-platform)

---

# Machine Learning

## Deep Learning Framework

* [PyTorch](https://github.com/pytorch/pytorch) - An open source deep learning framework by Facebook, with GPU and dynamic graph support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS*
  * Language API: *Python, C++, Java*
  * <details open><summary>Related projects (click to expand):</summary>

    * [TorchVision](https://github.com/pytorch/vision) - Datasets, Transforms and Models specific to Computer Vision for PyTorch
    * [TorchText](https://github.com/pytorch/text) - Data loaders and abstractions for text and NLP for PyTorch
    * [TorchAudio](https://github.com/pytorch/audio) - Data manipulation and transformation for audio signal processing for PyTorch
    * [TorchServe](https://github.com/pytorch/serve) - Serve, optimize and scale PyTorch models in production
    * [TorchHub](https://github.com/pytorch/hub) - Model zoo for PyTorch
    * [Ignite](https://github.com/pytorch/ignite) - High-level library to help with training and evaluating neural networks for PyTorch
    * [Captum](https://github.com/pytorch/captum) - A model interpretability and understanding library for PyTorch
    * [Glow](https://github.com/pytorch/glow) - Compiler for Neural Network hardware accelerators
    </details>

* [TensorFlow](https://github.com/tensorflow/tensorflow) - An open source deep learning framework by Google, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS, Raspberry Pi, Web*
  * Language API: *Python, C++, Java, JavaScript*
  * <details open><summary>Related projects (click to expand):</summary>

    * [TensorFlow Text](https://github.com/tensorflow/text) - A collection of text related classes and ops for TensorFlow
    * [TensorBoard](https://github.com/tensorflow/tensorboard) - TensorFlow's Visualization Toolkit
    * [TensorFlow Serving](https://github.com/tensorflow/serving) - A flexible, high-performance serving system for machine learning models based on TensorFlow
    * [TFDS](https://github.com/tensorflow/datasets) - A collection of datasets ready to use with TensorFlow and Jax
    * [TensorFlow Model Garden](https://github.com/tensorflow/models) - Models and examples built with TensorFlow
    * [TensorFlow.js](https://github.com/tensorflow/tfjs) - A WebGL accelerated JavaScript library for training and deploying ML models based on TensorFlow
    * [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) **(no longer maintained)** - Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research
    * [Trax](https://github.com/google/trax) **(successor of Tensor2Tensor)** - Deep Learning with Clear Code and Speed
    * [tf_numpy](https://www.tensorflow.org/guide/tf_numpy) - A subset of the NumPy API implemented in TensorFlow
    </details>

* [MXNet](https://github.com/apache/incubator-mxnet) - An open source deep learning framework by Apache, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Raspberry Pi*
  * Language API: *Python, C++, R, Julia, Scala, Go, Javascript*

* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - An open source deep learning framework by Baidu, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS, Web*
  * Language API: *Python, C++, Java, JavaScript*
  * <details open><summary>Related projects (click to expand):</summary>

    * [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Multilingual OCR toolkits based on PaddlePaddle
    * [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) - Object detection toolkit based on PaddlePaddle
    * [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) - Image segmentation toolkit based on PaddlePaddle
    * [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) - Visual classification and recognition toolkit based on PaddlePaddle
    * [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) - Generative Adversarial Networks toolkit based on PaddlePaddle
    * [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) - Video understanding toolkit based on PaddlePaddle
    * [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) - Recommendation algorithm based on PaddlePaddle
    * [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) - Natural language processing toolkit based on PaddlePaddle
    * [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) - Speech Recognition/Translation toolkit based on PaddlePaddle
    * [PGL](https://github.com/PaddlePaddle/PGL) - An efficient and flexible graph learning framework based on PaddlePaddle
    * [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) - Pre-trained models toolkit based on PaddlePaddle
    * [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) - Multi-platform high performance deep learning inference engine for PaddlePaddle
    * [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) - An open source deep learning framework running in the browser based on PaddlePaddle
    </details>

* [MegEngine](https://github.com/MegEngine/MegEngine) - An open source deep learning framework by MEGVII, with GPU support.
  * Supported platform: *Linux, Windows, MacOS*
  * Language API: *Python, C++*

* [MACE](https://github.com/XiaoMi/mace) - A deep learning inference framework optimized for mobile heterogeneous computing by XiaoMi.
  * Supported platform: *Android, iOS, Linux and Windows*

* [Neural Network Libraries](https://github.com/sony/nnabla) - An open source deep learning framework by Sony, with GPU support.

* [CNTK](https://github.com/microsoft/CNTK) **(not actively updated)** - An open source deep learning framework by Microsoft, with GPU support.
  * Supported platform: *Linux, Windows*
  * Language API: *Python, C++, Java, C#, .Net*

* [DyNet](https://github.com/clab/dynet) **(not actively updated)** - A C++ deep learning library by CMU.
  * Supported platform: *Linux, Windows, MacOS*
  * Language API: *C++, Python*

* [Chainer](https://github.com/chainer/chainer) **(not actively updated)** - A flexible framework of neural networks for deep learning.
  
* [fastai](https://github.com/fastai/fastai) - A high-level deep learning library based on PyTorch.

* [Lightning](https://github.com/Lightning-AI/lightning) - A high-level deep learning library based on PyTorch.

* [skorch](https://github.com/skorch-dev/skorch) **(not actively updated)** - A scikit-learn compatible neural network library based on PyTorch.

* [MMF](https://github.com/facebookresearch/mmf) **(not actively updated)** - A modular framework for vision and language multimodal research by Facebook AI Research, based on PyTorch.

* [ktrain](https://github.com/amaiya/ktrain) - A high-level deep learning library based on TensorFlow.

* [Tensorpack](https://github.com/tensorpack/tensorpack) **(not actively updated)** - A high-level deep learning library based on TensorFlow.

* [Sonnet](https://github.com/deepmind/sonnet) **(not actively updated)** - A high-level deep learning library based on TensorFlow.

* [Thinc](https://github.com/explosion/thinc) - A high-level deep learning library for PyTorch, TensorFlow and MXNet.

* [Ludwig](https://github.com/ludwig-ai/ludwig) - A declarative deep learning framework that allows users to train, evaluate, and deploy models without the need to write code.

* [Ivy](https://github.com/unifyai/ivy) **(not actively updated)** - A high-level deep learning library that unifies NumPy, PyTorch, TensorFlow, MXNet and JAX.

* [Jina](https://github.com/jina-ai/jina) - A high-level deep learning library for serving and deployment.

* [Haiku](https://github.com/deepmind/dm-haiku) - A high-level deep learning library based on JAX.

* [Turi Create](https://github.com/apple/turicreate) **(not actively updated)** - A machine learning library for deployment on MacOS/iOS.

## Machine Learning Framework

* [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine learning toolkit for Python.

* [XGBoost](https://github.com/dmlc/xgboost) - Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library.
  * Supported platform: *Linux, Windows, MacOS*
  * Supported distributed framework: *Hadoop, Spark, Dask, Flink, DataFlow*
  * Language API: *Python, C++, R, Java, Scala, Go*

* [LightGBM](https://github.com/microsoft/LightGBM) - A fast, distributed, high performance gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithms.
  * Supported platform: *Linux, Windows, MacOS*
  * Language API: *Python, C++, R*

* [CatBoost](https://github.com/catboost/catboost) - A fast, scalable, high performance Gradient Boosting on Decision Trees library.
  * Supported platform: *Linux, Windows, MacOS*
  * Language API: *Python, C++, R, Java*

* [JAX](https://github.com/google/jax) - Automatical differentiation for native Python and NumPy functions, with GPU support.

* [Flax](https://github.com/google/flax) - A high-performance neural network library and ecosystem for JAX that is designed for flexibility.

* [mlpack](https://github.com/mlpack/mlpack) **(not actively updated)** - A header-only C++ machine learning library.
  * Language API: *C++, Python, R, Julia, Go*

* [fklearn](https://github.com/nubank/fklearn) - A machine learning library that uses functional programming principles.

* [xLearn](https://github.com/aksnzhy/xlearn) **(not actively updated)** - A C++ machine learning library for linear model (LR), factorization machines (FM), and field-aware factorization machines (FFM).

* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) **(not actively updated)** - Fast GBDTs and Random Forests on GPUs.

* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) **(not actively updated)** - A Fast SVM Library on GPUs and CPUs.

## Computer Vision

* [OpenCV](https://github.com/opencv/opencv) - Open Source Computer Vision Library.
  * <details open><summary>Related projects (click to expand):</summary>

    * [opencv_contrib](https://github.com/opencv/opencv_contrib) - Repository for OpenCV's extra modules.
  </details>
  
* [opencv-python](https://github.com/opencv/opencv-python) - Pre-built CPU-only OpenCV packages for Python.

* [Detectron](https://github.com/facebookresearch/Detectron/) **(no longer maintained)** - A research platform for object detection research, implementing popular algorithms by Facebook, based on Caffe2.

* [Detectron2](https://github.com/facebookresearch/detectron2) **(successor of Detectron)** - A platform for object detection, segmentation and other visual recognition tasks, based on PyTorch.

* [Norfair](https://github.com/tryolabs/norfair) - Lightweight Python library for adding real-time multi-object tracking to any detector.

* [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) - A deep learning library for video understanding research, based on PyTorch.

* [Lightly](https://github.com/lightly-ai/lightly) - A computer vision framework for self-supervised learning, based on PyTorch.

* [ClassyVision](https://github.com/facebookresearch/ClassyVision) - An end-to-end framework for image and video classification, based on PyTorch.

* [pycls](https://github.com/facebookresearch/pycls) - Codebase for Image Classification Research, based on PyTorch.

* [SlowFast](https://github.com/facebookresearch/SlowFast) - Video understanding codebase from FAIR, based on PyTorch.

* [SAHI](https://github.com/obss/sahi) - Platform agnostic sliced/tiled inference + interactive ui + error analysis plots for object detection and instance segmentation.

* [GluonCV](https://github.com/dmlc/gluon-cv) - A high-level computer vision library for PyTorch and MXNet.

* [InsightFace](https://github.com/deepinsight/insightface) - An open source 2D&3D deep face analysis toolbox, based on PyTorch and MXNet.

* [Scenic](https://github.com/google-research/scenic) - A codebase with a focus on research around attention-based models for computer vision, based on JAX and Flax.

* [Deepface](https://github.com/serengil/deepface) - A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python.

* [Kornia](https://github.com/kornia/kornia) - Open source differentiable computer vision library, based on PyTorch.

* [Kubric](https://github.com/google-research/kubric) - A data generation pipeline for creating semi-realistic synthetic multi-object videos with rich annotations such as instance segmentation masks, depth maps, and optical flow.

* [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) - A collection of CV models, scripts, pretrained weights, based on PyTorch.

* [vit-pytorch](https://github.com/lucidrains/vit-pytorch) - A collection of Vision Transformer implementations, based on PyTorch.

* [vit-tensorflow](https://github.com/taki0112/vit-tensorflow) - A collection of Vision Transformer implementations, based on TensorFlow.

* [segmentation_models](https://github.com/qubvel/segmentation_models) **(not actively updated)** - Python library with Neural Networks for Image Segmentation based on Keras and TensorFlow.

* [LayoutParser](https://github.com/Layout-Parser/layout-parser) - A Unified Toolkit for Deep Learning Based Document Image Analysis, based on Detectron2.

* [Face Recognition](https://github.com/ageitgey/face_recognition) **(not actively updated)** - A facial recognition api for Python and the command line.

## Natural Language Processing

* [HuggingFace Transformers](https://github.com/huggingface/transformers) - A high-level machine learning library for text, images and audio data, with support for Pytorch, TensorFlow and JAX.

* [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - A high-performance library for text vocabularies and tokenizers.

* [NLTK](https://github.com/nltk/nltk) - An open source natural language processing library in Python.

* [spaCy](https://github.com/explosion/spaCy) - Industrial-strength Natural Language Processing (NLP) in Python.

* [ScispaCy](https://github.com/allenai/scispacy) - A Python package containing spaCy models for processing biomedical, scientific or clinical text.

* [PyTextRank](https://github.com/DerwenAI/pytextrank) - A Python implementation of TextRank as a spaCy pipeline extension, for graph-based natural language work.

* [textacy](https://github.com/chartbeat-labs/textacy) - a Python library for performing a variety of natural language processing tasks, based on spaCy.

* [spacy-transformers](https://github.com/explosion/spacy-transformers) - Use pretrained transformers in spaCy, based on HuggingFace Transformers.

* [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) - An open source natural language processing library for Apache Spark.

* [Flair](https://github.com/flairNLP/flair) - An open source natural language processing library, based on PyTorch.

* [Fairseq](https://github.com/facebookresearch/fairseq) - A sequence-to-sequence toolkit by Facebook, based on PyTorch.

* [ParlAI](https://github.com/facebookresearch/ParlAI) - A python framework for sharing, training and testing dialogue models from open-domain chitchat, based on PyTorch.

* [Stanza](https://github.com/stanfordnlp/stanza) - An open source natural language processing library by Stanford NLP Group, based on PyTorch.

* [ESPnet](https://github.com/espnet/espnet) - An end-to-end speech processing toolkit covering end-to-end speech recognition, text-to-speech, speech translation, speech enhancement, speaker diarization, spoken language understanding, based on PyTorch.

* [SpeechBrain](https://github.com/speechbrain/speechbrain) - An open-source and all-in-one conversational AI toolkit based on PyTorch.

* [NeMo](https://github.com/NVIDIA/NeMo) - A toolkit for conversational AI, based on PyTorch.

* [Sockeye](https://github.com/awslabs/sockeye) - An open-source sequence-to-sequence framework for Neural Machine Translation, based on PyTorch.

* [DeepPavlov](https://github.com/deeppavlov/DeepPavlov) - An open-source conversational AI library built on TensorFlow, Keras and PyTorch.

* [Spleeter](https://github.com/deezer/spleeter) - A source separation library with pretrained models, based on TensorFlow.

* [TTS](https://github.com/coqui-ai/TTS) - A library for advanced Text-to-Speech generation.

* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) - A Python library for audio feature extraction, classification, segmentation and applications.

* [Porcupine](https://github.com/Picovoice/porcupine) - On-device wake word detection powered by deep learning.

* [SpeechRecognition](https://github.com/Uberi/speech_recognition) **(not actively updated)** - Library for performing speech recognition, with support for several engines and APIs, online and offline.

* [Magenta](https://github.com/magenta/magenta) **(no longer maintained)** - Music and Art Generation with Machine Intelligence.

* [FARM](https://github.com/deepset-ai/FARM) **(not actively updated)** - Fast & easy transfer learning for NLP, which focuses on Question Answering.

* [Haystack](https://github.com/deepset-ai/haystack) **(successor of FARM)** - A high-level natural language processing library for deployment and production, based on PyTorch and HuggingFace Transformers.

* [NLP Architect](https://github.com/IntelLabs/nlp-architect) - A Deep Learning NLP/NLU library by Intel AI Lab, based on PyTorch and TensorFlow.

* [LightSeq](https://github.com/bytedance/lightseq) - A high performance training and inference library for sequence processing and generation implemented in CUDA, for Fairseq and HuggingFace Transformers.

* [fastNLP](https://github.com/fastnlp/fastNLP) - A Modularized and Extensible NLP Framework for PyTorch and PaddleNLP.

* [Rubrix](https://github.com/recognai/rubrix) - A production-ready Python framework for exploring, annotating, and managing data in NLP projects.

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) - The PyTorch version of the OpenNMT project, an open-source neural machine translation framework.

* [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) - The TensorFlow version of the OpenNMT project, an open-source neural machine translation framework.

* [AllenNLP](https://github.com/allenai/allennlp) **(not actively updated)** - An open source natural language processing library, based on PyTorch.

* [GluonNLP](https://github.com/dmlc/gluon-nlp) **(not actively updated)** - A high-level NLP toolkit, based on MXNet.

* [Gensim](https://github.com/RaRe-Technologies/gensim) - A Python library for topic modelling, document indexing and similarity retrieval with large corpora, based on NumPy and SciPy.

* [CLTK](https://github.com/cltk/cltk) - A Python library offering natural language processing for pre-modern languages.

* [Rasa](https://github.com/RasaHQ/rasa) - Open source machine learning framework to automate text- and voice-based conversations.

* [SentencePiece](https://github.com/google/sentencepiece) - Unsupervised text tokenizer for Neural Network-based text generation.

* [subword-nmt](https://github.com/rsennrich/subword-nmt) - Unsupervised Word Segmentation for Neural Machine Translation and Text Generation.

* [OpenNRE](https://github.com/thunlp/OpenNRE) - An open-source and extensible toolkit that provides a unified framework to implement relation extraction models.

* [sumy](https://github.com/miso-belica/sumy) - Module for automatic summarization of text documents and HTML pages.

* [OpenPrompt](https://github.com/thunlp/OpenPrompt) - An Open-Source Framework for Prompt-Learning.

* [jiant](https://github.com/nyu-mll/jiant) **(no longer maintained)** - The multitask and transfer learning toolkit for natural language processing research.

* [fastText](https://github.com/facebookresearch/fastText) **(not actively updated)** - A library for efficient learning of word representations and sentence classification.

* [TextBlob](https://github.com/sloria/TextBlob) **(not actively updated)** - A Python library for processing textual data.

## Reinforcement Learning

* [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) - A fast, flexible, online, and active learning solution for solving complex interactive machine learning problems.

## Graph

* [DGL](https://github.com/dmlc/dgl) - An easy-to-use, high performance and scalable Python package for deep learning on graphs for PyTorch, Apache MXNet or TensorFlow.

* [NetworkX](https://github.com/networkx/networkx) - A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

* [PyG](https://github.com/pyg-team/pytorch_geometric) - A Graph Neural Network Library based on PyTorch.

* [OGB](https://github.com/snap-stanford/ogb) - Benchmark datasets, data loaders, and evaluators for graph machine learning.

* [Spektral](https://github.com/danielegrattarola/spektral) - A Python library for graph deep learning, based on Keras and TensorFlow.

* [Graph4nlp](https://github.com/graph4ai/graph4nlp) - A library for the easy use of Graph Neural Networks for NLP (DLG4NLP).

* [Jraph](https://github.com/deepmind/jraph) - A Graph Neural Network Library in Jax.

* [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding) - Implementation and experiments of graph embedding algorithms.

* [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) **(not actively updated)** - Generate embeddings from large-scale graph-structured data,  based on PyTorch.

* [TensorFlow Graphics](https://github.com/tensorflow/graphics) **(not actively updated)** - Differentiable Graphics Layers for TensorFlow.

* [StellarGraph](https://github.com/stellargraph/stellargraph) **(not actively updated)** - A Python library for machine learning on graphs and networks.

## Time-Series & Financial

* [Prophet](https://github.com/facebook/prophet) - Tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth.

* [darts](https://github.com/unit8co/darts) - A python library for easy manipulation and forecasting of time series.

* [GluonTS](https://github.com/awslabs/gluonts) - Probabilistic time series modeling in Python.

* [tslearn](https://github.com/tslearn-team/tslearn) - A machine learning toolkit dedicated to time-series data.

* [sktime](https://github.com/sktime/sktime) - A unified framework for machine learning with time series.

* [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting) - Time series forecasting with PyTorch.

* [STUMPY](https://github.com/TDAmeritrade/stumpy) - A powerful and scalable Python library for modern time series analysis.

* [StatsForecast](https://github.com/Nixtla/statsforecast) - Offers a collection of widely used univariate time series forecasting models, including automatic ARIMA and ETS modeling optimized for high performance using numba.

* [Orbit](https://github.com/uber/orbit) - A Python package for Bayesian time series forecasting and inference.

* [Pmdarima](https://github.com/alkaline-ml/pmdarima) - A statistical library designed to fill the void in Python's time series analysis capabilities, including the equivalent of R's auto.arima function.

* [pyts](https://github.com/johannfaouzi/pyts) **(not actively updated)** - A Python package for time series classification.

* [Qlib](https://github.com/microsoft/qlib) - An AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

* [IB-insync](https://github.com/erdewit/ib_insync) - Python sync/async framework for Interactive Brokers API.

* [ffn](https://github.com/pmorissette/ffn) - A financial function library for Python.

* [bt](https://github.com/pmorissette/bt) - A flexible backtesting framework for Python used to test quantitative trading strategies, based on ffn.

* [finmarketpy](https://github.com/cuemacro/finmarketpy) - Python library for backtesting trading strategies & analyzing financial markets.

* [TensorTrade](https://github.com/tensortrade-org/tensortrade) - An open source reinforcement learning framework for training, evaluating, and deploying robust trading agents, based on TensorFlow.

* [TF Quant Finance](https://github.com/google/tf-quant-finance) - High-performance TensorFlow library for quantitative finance.

* [CryptoSignal](https://github.com/CryptoSignal/Crypto-Signal) **(not actively updated)** - A command line tool that automates your crypto currency Technical Analysis (TA).

* [Catalyst](https://github.com/scrtlabs/catalyst) **(no longer maintained)** - An algorithmic trading library for crypto-assets written in Python.

## Linear Algebra / Statistics Toolkit

* [NumPy](https://github.com/numpy/numpy) - The fundamental package for scientific computing with Python.

* [SciPy](https://github.com/scipy/scipy) - An open-source software for mathematics, science, and engineering in Python.

* [CuPy](https://github.com/cupy/cupy) - A NumPy/SciPy-compatible array library for GPU-accelerated computing with Python.

* [Statsmodels](https://github.com/statsmodels/statsmodels) - Statistical modeling and econometrics in Python.

* [Theano](https://github.com/Theano/Theano) **(no longer maintained)** - A Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.

* [Aesara](https://github.com/aesara-devs/aesara) **(successor of Theano)** - A Python library that allows one to define, optimize/rewrite, and evaluate mathematical expressions, especially ones involving multi-dimensional arrays.

* [einops](https://github.com/arogozhnikov/einops) - A tensor operation library for NumPy, PyTorch, TensorFlow and JAX.

* [openTSNE](https://github.com/pavlin-policar/openTSNE) - Extensible, parallel Python implementations of t-SNE.

* [UMAP](https://github.com/lmcinnes/umap) - Uniform Manifold Approximation and Projection, a dimension reduction technique that can be used for visualisation similarly to t-SNE.

## Data Processing

* [Pillow](https://github.com/python-pillow/Pillow) - The friendly PIL fork (Python Imaging Library).

* [Imageio](https://github.com/imageio/imageio) - Python library for reading and writing image data.

* [scikit-image](https://github.com/scikit-image/scikit-image) - Image processing in Python

* [MoviePy](https://github.com/Zulko/moviepy) - Video editing with Python.

* [Wand](https://github.com/emcconville/wand) - The ctypes-based simple ImageMagick binding for Python.

* [VidGear](https://github.com/abhiTronix/vidgear) - A High-performance cross-platform Video Processing Python framework powerpacked with unique trailblazing features.

* [Albumentations](https://github.com/albumentations-team/albumentations) - A Python library for image augmentation.

* [Augmentor](https://github.com/mdbloice/Augmentor) - Image augmentation library in Python for machine learning.

* [imutils](https://github.com/PyImageSearch/imutils) - A basic image processing toolkit in Python, based on OpenCV.

* [Towhee](https://github.com/towhee-io/towhee) - Data processing pipelines for neural networks.

* [ffcv](https://github.com/libffcv/ffcv) - A drop-in data loading system that dramatically increases data throughput in model training.

* [ImageHash](https://github.com/JohannesBuchner/imagehash) - An image hashing library written in Python.

* [image-match](https://github.com/ProvenanceLabs/image-match) - a simple package for finding approximate image matches from a corpus.

* [pandas-profiling](https://github.com/ydataai/pandas-profiling) - Create HTML data profiling reports for pandas DataFrame.

* [FiftyOne](https://github.com/voxel51/fiftyone) - An open-source tool for building high-quality datasets and computer vision models.

* [dedupe](https://github.com/dedupeio/dedupe) - A python library that uses machine learning to perform fuzzy matching, deduplication and entity resolution quickly on structured data.

* [Ciphey](https://github.com/Ciphey/Ciphey) - Automatically decrypt encryptions without knowing the key or cipher, decode encodings, and crack hashes.

* [jellyfish](https://github.com/jamesturk/jellyfish) - A library for approximate & phonetic matching of strings.

* [TextDistance](https://github.com/life4/textdistance) - Python library for comparing distance between two or more sequences by many algorithms.

* [NLPAUG](https://github.com/makcedward/nlpaug) - Data augmentation for NLP.

* [Texthero](https://github.com/jbesomi/texthero) - A python toolkit to work with text-based dataset, bases on Pandas.

* [ftfy](https://github.com/rspeer/python-ftfy) - Fixes mojibake and other glitches in Unicode text.

* [librosa](https://github.com/librosa/librosa) - A python package for music and audio analysis.

* [Pydub](https://github.com/jiaaro/pydub) - Manipulate audio with a simple and easy high level interface.

* [Audiomentations](https://github.com/iver56/audiomentations) - A Python library for audio data augmentation.

* [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations) - Fast audio data augmentation in PyTorch, with GPU support.

* [DDSP](https://github.com/magenta/ddsp) - A library of differentiable versions of common DSP functions.

* [TSFRESH](https://github.com/blue-yonder/tsfresh) - Automatic extraction of relevant features from time series.

* [Qdrant](https://github.com/qdrant/qdrant) - A vector similarity search engine for text, image and categorical data in Rust.

* [TA](https://github.com/bukosabino/ta) - A Technical Analysis library useful to do feature engineering from financial time series datasets, based on Pandas and NumPy.

* [stockstats](https://github.com/jealous/stockstats) - Supply a wrapper ``StockDataFrame`` based on the ``pandas.DataFrame`` with inline stock statistics/indicators support.

## Data Visualization

* [Matplotlib](https://github.com/matplotlib/matplotlib) - A comprehensive library for creating static, animated, and interactive visualizations in Python.

* [Seaborn](https://github.com/mwaskom/seaborn) - A high-level interface for drawing statistical graphics, based on Matplotlib.

* [HyperTools](https://github.com/ContextLab/hypertools) - A Python toolbox for gaining geometric insights into high-dimensional data, based on Matplotlib and Seaborn.

* [Bokeh](https://github.com/bokeh/bokeh) - Interactive Data Visualization in the browser, from Python.

* [Plotly.js](https://github.com/plotly/plotly.js) - Open-source JavaScript charting library behind Plotly and Dash.

* [Plotly.py](https://github.com/plotly/plotly.py) - An interactive, open-source, and browser-based graphing library for Python, based on Plotly.js.

* [Dash](https://github.com/plotly/dash) - Analytical Web Apps for Python, R, Julia and Jupyter, based on Plotly.js.

* [Datapane](https://github.com/datapane/datapane) - An open-source framework to create data science reports in Python.

* [mpld3](https://github.com/mpld3/mpld3) - An interactive Matplotlib visualization tool in browser, based on D3.

* [Vega](https://github.com/vega/vega) - A visualization grammar, a declarative format for creating, saving, and sharing interactive visualization designs.

* [Vega-Lite](https://github.com/vega/vega-lite) - Provides a higher-level grammar for visual analysis that generates complete Vega specifications.

* [Vega-Altair](https://github.com/altair-viz/altair) - A declarative statistical visualization library for Python, based on Vega-Lite.

* [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) - Fast data visualization and GUI tools for scientific / engineering applications.

* [VisPy](https://github.com/vispy/vispy) - A high-performance interactive 2D/3D data visualization library, with OpenGL support.

* [PyVista](https://github.com/pyvista/pyvista) - 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK).

* [Holoviews](https://github.com/holoviz/holoviews) - An open-source Python library designed to make data analysis and visualization seamless and simple.

* [Graphviz](https://github.com/xflr6/graphviz) - Python interface for Graphviz to create and render graphs.

* [PyGraphistry](https://github.com/graphistry/pygraphistry) - A Python library to quickly load, shape, embed, and explore big graphs with the GPU-accelerated Graphistry visual graph analyzer.

* [Apache ECharts](https://github.com/apache/echarts) - A powerful, interactive charting and data visualization library for browser.

* [pyecharts](https://github.com/pyecharts/pyecharts) - A Python visualization interface for Apache ECharts.

* [word_cloud](https://github.com/amueller/word_cloud) - A little word cloud generator in Python.

* [Datashader](https://github.com/holoviz/datashader) - A data rasterization pipeline for automating the process of creating meaningful representations of large amounts of data.

* [Perspective](https://github.com/finos/perspective) - A data visualization and analytics component, especially well-suited for large and/or streaming datasets.

* [ggplot2](https://github.com/tidyverse/ggplot2) - An implementation of the Grammar of Graphics in R.

* [plotnine](https://github.com/has2k1/plotnine) - An implementation of the Grammar of Graphics in Python, based on ggplot2.

* [bqplot](https://github.com/bqplot/bqplot) - An implementation of the Grammar of Graphics for IPython/Jupyter notebooks.

* [D-Tale](https://github.com/man-group/dtale) - A visualization tool for Pandas DataFrame, with ipython notebooks support.

* [missingno](https://github.com/ResidentMario/missingno) - A Python visualization tool for missing data.

* [HiPlot](https://github.com/facebookresearch/hiplot) - A lightweight interactive visualization tool to help AI researchers discover correlations and patterns in high-dimensional data.

* [Sweetviz](https://github.com/fbdesignpro/sweetviz) - Visualize and compare datasets, target values and associations, with one line of code.

* [Scattertext](https://github.com/JasonKessler/scattertext) **(not actively updated)** - A tool for finding distinguishing terms in corpora and displaying them in an interactive HTML scatter plot.

## Machine Learning Tutorials

* [labml.ai](https://nn.labml.ai/) - A collection of PyTorch implementations of neural networks and related algorithms, which are documented with explanations and rendered as side-by-side formatted notes.

* [PyTorch official tutorials](https://pytorch.org/tutorials/) - Official tutorials for PyTorch.

* [numpy-100](https://github.com/rougier/numpy-100) - 100 numpy exercises (with solutions).

# Full-Stack Development

## Web Development

* [D3](https://github.com/d3/d3) - A JavaScript library for visualizing data using web standards.

## Data Management & Processing

* [Apache Flink](https://github.com/apache/flink) - An open source stream processing framework with powerful stream- and batch-processing capabilities.

---

# Academic

## Mathematics

## Paper Reading


---

# Useful Tools

## MacOS

* [Scroll-Reverser](https://github.com/pilotmoon/Scroll-Reverser) - Reverses the direction of macOS scrolling, with independent settings for trackpads and mice.

## Cross-Platform

* [LANDrop](https://github.com/LANDrop/LANDrop) - A cross-platform tool that you can use to conveniently transfer photos, videos, and other types of files to other devices on the same local network.
