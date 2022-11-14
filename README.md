# Awesome List

<h1 align="center">
    <br>
    <img width="200" src="awesome.svg" alt="[!awesome](https://raw.githubusercontent.com/sindresorhus/awesome/main/media/logo.svg)">
    <br>
</h1>

A list of useful stuff in Machine Learning, Computer Graphics, Software Development, ...

---

# Table of Contents

- [Machine Learning](#machine-learning)
  - [Deep Learning Framework](#deep-learning-framework)
    - [High-Level DL APIs](#high-level-dl-apis)
    - [Deployment & Distribution](#deployment--distribution)
    - [Auto ML & Hyperparameter Optimization](#auto-ml--hyperparameter-optimization)
    - [Interpretability & Adversarial Training](#interpretability--adversarial-training)
    - [Anomaly Detection & Others](#anomaly-detection--others)
  - [Machine Learning Framework](#machine-learning-framework)
    - [General Purpose Framework](#general-purpose-framework)
    - [Nearest Neighbors & Similarity](#nearest-neighbors--similarity)
    - [Hyperparameter Search & Gradient-Free Optimization](#hyperparameter-search--gradient-free-optimization)
    - [Experiment Management](#experiment-management)
    - [Model Interpretation](#model-interpretation)
    - [Anomaly Detection](#anomaly-detection)
  - [Computer Vision](#computer-vision)
    - [General Purpose CV](#general-purpose-cv)
    - [Classification & Detection & Tracking](#classification--detection--tracking)
    - [OCR](#ocr)
    - [Image / Video Generation](#image--video-generation)
  - [Natural Language Processing](#natural-language-processing)
    - [General Purpose NLP](#general-purpose-nlp)
    - [Conversation & Translation](#conversation--translation)
    - [Speech & Audio](#speech--audio)
    - [Others](#others)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Graph](#graph)
  - [Causal Inference](#causal-inference)
  - [Recommendation, Advertisement & Ranking](#recommendation-advertisement--ranking)
  - [Time-Series & Financial](#time-series--financial)
  - [Other Machine Learning Applications](#other-machine-learning-applications)
  - [Linear Algebra / Statistics Toolkit](#linear-algebra--statistics-toolkit)
    - [General Purpose Tensor Library](#general-purpose-tensor-library)
    - [Tensor Similarity & Dimension Reduction](#tensor-similarity--dimension-reduction)
    - [Statistical Toolkit](#statistical-toolkit)
    - [Others](#others-1)
  - [Data Processing](#data-processing)
    - [Data Representation](#data-representation)
    - [Data Pre-processing & Loading](#data-pre-processing--loading)
    - [Data Similarity](#data-similarity)
    - [Data Management](#data-management)
  - [Data Visualization](#data-visualization)
  - [Machine Learning Tutorials](#machine-learning-tutorials)
- [Computer Graphics](#computer-graphics)
  - [Graphic Libraries & Renderers](#graphic-libraries--renderers)
  - [Game Engines](#game-engines)
  - [CG Tutorials](#cg-tutorials)
- [Full-Stack Development](#full-stack-development)
  - [DevOps](#devops)
  - [Desktop App Development](#desktop-app-development)
    - [Python Toolkit](#python-toolkit)
    - [C++/C Toolkit](#cc-toolkit)
  - [Web Development](#web-development)
  - [Mobile Development](#mobile-development)
  - [Process, Thread & Coroutine](#process-thread--coroutine)
  - [Debugging & Profiling & Tracing](#debugging--profiling--tracing)
    - [For Python](#for-python)
    - [For C++/C](#for-cc)
    - [For Go](#for-go)
  - [Data Management & Processing](#data-management--processing)
    - [Database & Cloud Management](#database--cloud-management)
    - [Streaming Data Management](#streaming-data-management)
  - [Data Format & I/O](#data-format--io)
    - [For Python](#for-python-1)
    - [For C++/C](#for-cc-1)
    - [For Go](#for-go-1)
    - [For Java](#for-java)
  - [Security](#security)
  - [Package Management](#package-management)
    - [For Python](#for-python-2)
    - [For C++/C](#for-cc-2)
    - [For Scala](#for-scala)
    - [For JavaScript](#for-javascript)
  - [Containers & Language Extentions & Linting](#containers--language-extentions--linting)
    - [For Python](#for-python-3)
    - [For C++/C](#for-cc-3)
    - [For Go](#for-go-2)
    - [For Java](#for-java-1)
    - [For Scala](#for-scala-1)
  - [For JavaScript](#for-javascript-1)
  - [Programming Language Tutorials](#programming-language-tutorials)
    - [Python](#python)
    - [C++/C](#cc)
    - [Go](#go)
    - [Java](#java)
    - [Scala](#scala)
    - [Flutter](#flutter)
    - [JavaScript](#javascript)
- [Useful Tools](#useful-tools)
  - [MacOS](#macos)
  - [Windows](#windows)
  - [Linux](#linux)
  - [Cross-Platform](#cross-platform)
- [Other Awesome Lists](#other-awesome-lists)
  - [Machine Learning](#machine-learning-1)
  - [Computer Graphics](#computer-graphics-1)
  - [Programming Language](#programming-language)

---

# Machine Learning

## Deep Learning Framework

### High-Level DL APIs

* [PyTorch](https://github.com/pytorch/pytorch) - An open source deep learning framework by Facebook, with GPU and dynamic graph support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS*
  * Language API: *Python, C++, Java*
  * <details open><summary>Related projects:</summary>

    * [TorchVision](https://github.com/pytorch/vision) - Datasets, Transforms and Models specific to Computer Vision for PyTorch
    * [TorchText](https://github.com/pytorch/text) - Data loaders and abstractions for text and NLP for PyTorch
    * [TorchAudio](https://github.com/pytorch/audio) - Data manipulation and transformation for audio signal processing for PyTorch
    * [TorchRec](https://github.com/pytorch/torchrec) - A PyTorch domain library built to provide common sparsity & parallelism primitives needed for large-scale recommender systems (RecSys).
    * [TorchServe](https://github.com/pytorch/serve) - Serve, optimize and scale PyTorch models in production
    * [TorchHub](https://github.com/pytorch/hub) - Model zoo for PyTorch
    * [Ignite](https://github.com/pytorch/ignite) - High-level library to help with training and evaluating neural networks for PyTorch
    * [Captum](https://github.com/pytorch/captum) - A model interpretability and understanding library for PyTorch
    * [Glow](https://github.com/pytorch/glow) - Compiler for Neural Network hardware accelerators
    * [BoTorch](https://github.com/pytorch/botorch) - Bayesian optimization in PyTorch
    * [TNT](https://github.com/pytorch/tnt) - A library for PyTorch training tools and utilities
    * [TorchArrow](https://github.com/pytorch/torcharrow) - Common and composable data structures built on PyTorch Tensor for efficient batch data representation and processing in PyTorch model authoring
    * [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) - A deep learning library for video understanding research, based on PyTorch
    * [tensorboardX](https://github.com/lanpa/tensorboardX) - Tensorboard for pytorch (and chainer, mxnet, numpy, ...)
    * [TorchMetrics](https://github.com/Lightning-AI/metrics) - Machine learning metrics for distributed, scalable PyTorch applications
    * [Apex](https://github.com/NVIDIA/apex) - Tools for easy mixed precision and distributed training in Pytorch
    * [HuggingFace Accelerate](https://github.com/huggingface/accelerate) - A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision
    * [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning) - The easiest way to use deep metric learning in your application. Modular, flexible, and extensible, written in PyTorch
    * [Auto-PyTorch](https://github.com/automl/Auto-PyTorch) - Automatic architecture search and hyperparameter optimization for PyTorch
    * [torch-optimizer](https://github.com/jettify/pytorch-optimizer) - Collection of optimizers for PyTorch compatible with optim module
    * [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) - PyTorch Extension Library of Optimized Autograd Sparse Matrix Operations
    * [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) - PyTorch Extension Library of Optimized Scatter Operations
    * [Torch-Struct](https://github.com/harvardnlp/pytorch-struct) - A library of tested, GPU implementations of core structured prediction algorithms for deep learning applications
    * [torchinfo](https://github.com/TylerYep/torchinfo) - View model summaries in PyTorch
    * [Torchshow](https://github.com/xwying/torchshow) - Visualize PyTorch tensors with a single line of code
    * [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) - An easy to use PyTorch to TensorRT converter
    * [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) - A PyTorch Library for Accelerating 3D Deep Learning Research
    * [higher](https://github.com/facebookresearch/higher) **(not actively updated)** - A pytorch library allowing users to obtain higher order gradients over losses spanning training loops rather than individual training steps
    </details>

* [TensorFlow](https://github.com/tensorflow/tensorflow) - An open source deep learning framework by Google, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS, Raspberry Pi, Web*
  * Language API: *Python, C++, Java, JavaScript*
  * <details open><summary>Related projects:</summary>

    * [TensorBoard](https://github.com/tensorflow/tensorboard) - TensorFlow's Visualization Toolkit
    * [TensorFlow Text](https://github.com/tensorflow/text) - A collection of text related classes and ops for TensorFlow
    * [TensorFlow Recommenders](https://github.com/tensorflow/recommenders) - A library for building recommender system models using TensorFlow.
    * [TensorFlow Ranking](https://github.com/tensorflow/ranking) - A library for Learning-to-Rank (LTR) techniques on the TensorFlow platform.
    * [TensorFlow Serving](https://github.com/tensorflow/serving) - A flexible, high-performance serving system for machine learning models based on TensorFlow
    * [TFX](https://github.com/tensorflow/tfx) - An end-to-end platform for deploying production ML pipelines.
    * [TFDS](https://github.com/tensorflow/datasets) - A collection of datasets ready to use with TensorFlow and Jax
    * [TensorFlow Addons](https://github.com/tensorflow/addons) - Useful extra functionality for TensorFlow 2.x maintained by SIG-addons
    * [TensorFlow Transform](https://github.com/tensorflow/transform) - A library for preprocessing data with TensorFlow
    * [TensorFlow Model Garden](https://github.com/tensorflow/models) - Models and examples built with TensorFlow
    * [TensorFlow Hub](https://github.com/tensorflow/hub) - A library for transfer learning by reusing parts of TensorFlow models
    * [TensorFlow.js](https://github.com/tensorflow/tfjs) - A WebGL accelerated JavaScript library for training and deploying ML models based on TensorFlow
    * [TensorFlow Probability](https://github.com/tensorflow/probability) - Probabilistic reasoning and statistical analysis in TensorFlow
    * [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization) - A toolkit to optimize ML models for deployment for Keras and TensorFlow, including quantization and pruning
    * [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis) - A library for evaluating TensorFlow models
    * [Trax](https://github.com/google/trax) **(successor of Tensor2Tensor)** - Deep Learning with Clear Code and Speed
    * [Lattice](https://github.com/tensorflow/lattice) - Lattice methods in TensorFlow
    * [tf_numpy](https://www.tensorflow.org/guide/tf_numpy) - A subset of the NumPy API implemented in TensorFlow
    * [TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark) - Brings TensorFlow programs to Apache Spark clusters
    * [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) **(no longer maintained)** - Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research
    </details>

* [MXNet](https://github.com/apache/incubator-mxnet) - An open source deep learning framework by Apache, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Raspberry Pi*
  * Language API: *Python, C++, R, Julia, Scala, Go, Javascript*

* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - An open source deep learning framework by Baidu, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS, Web*
  * Language API: *Python, C++, Java, JavaScript*
  * <details open><summary>Related projects:</summary>

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
    * [PARL](https://github.com/PaddlePaddle/PARL) - A high-performance distributed training framework for Reinforcement Learning based on PaddlePaddle
    * [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) - Pre-trained models toolkit based on PaddlePaddle
    * [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) - Multi-platform high performance deep learning inference engine for PaddlePaddle
    * [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) - An open source deep learning framework running in the browser based on PaddlePaddle
    * [VisualDL](https://github.com/PaddlePaddle/VisualDL) - A visualization analysis tool of PaddlePaddle
    </details>

* [MegEngine](https://github.com/MegEngine/MegEngine) - An open source deep learning framework by MEGVII, with GPU support.
  * Supported platform: *Linux, Windows, MacOS*
  * Language API: *Python, C++*

* [MACE](https://github.com/XiaoMi/mace) - A deep learning inference framework optimized for mobile heterogeneous computing by XiaoMi.
  * Supported platform: *Android, iOS, Linux and Windows*

* [Neural Network Libraries](https://github.com/sony/nnabla) - An open source deep learning framework by Sony, with GPU support.

* [OneFlow](https://github.com/Oneflow-Inc/oneflow) - A deep learning framework designed to be user-friendly, scalable and efficient.

* [fastai](https://github.com/fastai/fastai) - A high-level deep learning library based on PyTorch.

* [Lightning](https://github.com/Lightning-AI/lightning) - A high-level deep learning library based on PyTorch.

* [Lightning Flash](https://github.com/Lightning-AI/lightning-flash) - Your PyTorch AI Factory - Flash enables you to easily configure and run complex AI recipes for over 15 tasks across 7 data domains

* [tinygrad](https://github.com/geohot/tinygrad) - A deep learning framework in between a pytorch and a karpathy/micrograd.

* [Flashlight](https://github.com/flashlight/flashlight) **(successor of wav2letter++)** - A C++ standalone library for machine learning.

* [Avalanche](https://github.com/ContinualAI/avalanche) - An End-to-End Library for Continual Learning, based on PyTorch.

* [ktrain](https://github.com/amaiya/ktrain) - A high-level deep learning library based on TensorFlow.

* [Thinc](https://github.com/explosion/thinc) - A high-level deep learning library for PyTorch, TensorFlow and MXNet.

* [Ludwig](https://github.com/ludwig-ai/ludwig) - A declarative deep learning framework that allows users to train, evaluate, and deploy models without the need to write code.

* [Jina](https://github.com/jina-ai/jina) - A high-level deep learning library for serving and deployment.

* [Haiku](https://github.com/deepmind/dm-haiku) - A high-level deep learning library based on JAX.

* [scarpet-nn](https://github.com/ashutoshbsathe/scarpet-nn) - Tools and libraries to run neural networks in Minecraft.

* [CNTK](https://github.com/microsoft/CNTK) **(not actively updated)** - An open source deep learning framework by Microsoft, with GPU support.
  * Supported platform: *Linux, Windows*
  * Language API: *Python, C++, Java, C#, .Net*

* [DyNet](https://github.com/clab/dynet) **(not actively updated)** - A C++ deep learning library by CMU.
  * Supported platform: *Linux, Windows, MacOS*
  * Language API: *C++, Python*

* [Chainer](https://github.com/chainer/chainer) **(not actively updated)** - A flexible framework of neural networks for deep learning.

* [skorch](https://github.com/skorch-dev/skorch) **(not actively updated)** - A scikit-learn compatible neural network library based on PyTorch.

* [MMF](https://github.com/facebookresearch/mmf) **(not actively updated)** - A modular framework for vision and language multimodal research by Facebook AI Research, based on PyTorch.

* [Tensorpack](https://github.com/tensorpack/tensorpack) **(not actively updated)** - A high-level deep learning library based on TensorFlow.

* [Sonnet](https://github.com/deepmind/sonnet) **(not actively updated)** - A high-level deep learning library based on TensorFlow.

* [Ivy](https://github.com/unifyai/ivy) **(not actively updated)** - A high-level deep learning library that unifies NumPy, PyTorch, TensorFlow, MXNet and JAX.

* [X-DeepLearning](https://github.com/alibaba/x-deeplearning) **(not actively updated)** - An industrial deep learning framework for high-dimension sparse data.

* [HiddenLayer](https://github.com/waleedka/hiddenlayer) **(not actively updated)** - Neural network graphs and training metrics for PyTorch, Tensorflow, and Keras.

* [TensorFX](https://github.com/TensorLab/tensorfx) **(not actively updated)** - TensorFlow framework for training and serving machine learning models.

* [FeatherCNN](https://github.com/Tencent/FeatherCNN) **(not actively updated)** - A high performance inference engine for convolutional neural networks.

* [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) **(not actively updated)** - Header only, dependency-free deep learning framework in C++14.

* [TFLearn](https://github.com/tflearn/tflearn) - Deep learning library featuring a higher-level API for TensorFlow.

### Deployment & Distribution

* [MediaPipe](https://github.com/google/mediapipe) - Cross-platform, customizable ML solutions for live and streaming media.

* [Triton](https://github.com/openai/triton) - A language and compiler for writing highly efficient custom Deep-Learning primitives.

* [Hummingbird](https://github.com/microsoft/hummingbird) - A library for compiling trained traditional ML models into tensor computations.

* [OpenVINO](https://github.com/openvinotoolkit/openvino) - An open-source toolkit for optimizing and deploying AI inference.
  * <details open><summary>Related projects:</summary>

    * [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo) - Pre-trained Deep Learning models and demos (high quality and extremely fast).
    </details>

* [Kubeflow](https://github.com/kubeflow/kubeflow) - Machine Learning Toolkit for Kubernetes.

* [Kubeflow Training Operator](https://github.com/kubeflow/training-operator) - Training operators on Kubernetes.

* [m2cgen](https://github.com/BayesWitnesses/m2cgen) - Transform ML models into a native code (Java, C, Python, Go, JavaScript, Visual Basic, C#, R, PowerShell, PHP, Dart, Haskell, Ruby, F#, Rust) with zero dependencies.

* [DeepSpeed](https://github.com/microsoft/DeepSpeed) - An easy-to-use deep learning optimization software suite that enables unprecedented scale and speed for Deep Learning Training and Inference.

* [Analytics Zoo](https://github.com/intel-analytics/analytics-zoo) **(no longer maintained)** - Distributed Tensorflow, Keras and PyTorch on Apache Spark/Flink & Ray.

* [BigDL](https://github.com/intel-analytics/BigDL) **(successor of Analytics Zoo)** - Building Large-Scale AI Applications for Distributed Big Data.

* [FairScale](https://github.com/facebookresearch/fairscale) - A PyTorch extension library for high performance and large scale training.

* [ColossalAI](https://github.com/hpcaitech/ColossalAI) - Provides a collection of parallel components and user-friendly tools to kickstart distributed training and inference in a few lines.

* [Ray](https://github.com/ray-project/ray) - A unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a toolkit of libraries (Ray AIR) for accelerating ML workloads.

* [BentoML](https://github.com/bentoml/BentoML) - BentoML is compatible across machine learning frameworks and standardizes ML model packaging and management for your team.

* [cortex](https://github.com/cortexlabs/cortex) - Production infrastructure for machine learning at scale.

* [Horovod](https://github.com/horovod/horovod) - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.

* [Angel](https://github.com/Angel-ML/angel) - A Flexible and Powerful Parameter Server for large-scale machine learning.

* [Elephas](https://github.com/maxpumperla/elephas) **(no longer maintained)** - Distributed Deep learning with Keras & Spark.

* [Elephas](https://github.com/danielenricocahall/elephas) **(successor of maxpumperla/elephas)** - Distributed Deep learning with Keras & Spark.

* [MLeap](https://github.com/combust/mleap) - Allows data scientists and engineers to deploy machine learning pipelines from Spark and Scikit-learn to a portable format and execution engine.

* [ZenML](https://github.com/zenml-io/zenml) - Build portable, production-ready MLOps pipelines.

* [Optimus](https://github.com/hi-primus/optimus) - An opinionated python library to easily load, process, plot and create ML models that run over pandas, Dask, cuDF, dask-cuDF, Vaex or Spark.

* [ONNX](https://github.com/onnx/onnx) - Open standard for machine learning interoperability.

* [TensorRT](https://github.com/NVIDIA/TensorRT) - A C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.

* [Compute Library](https://github.com/ARM-software/ComputeLibrary) - A set of computer vision and machine learning functions optimised for both Arm CPUs and GPUs using SIMD technologies.

* [Apache TVM](https://github.com/apache/tvm) - Open deep learning compiler stack for cpu, gpu and specialized accelerators.

* [Triton Inference Server](https://github.com/triton-inference-server/server) - The Triton Inference Server provides an optimized cloud and edge inferencing solution.

* [Core ML Tools](https://github.com/apple/coremltools) - Contains supporting tools for Core ML model conversion, editing, and validation.

* [Petastorm](https://github.com/uber/petastorm) - Enables single machine or distributed training and evaluation of deep learning models from datasets in Apache Parquet format.

* [Hivemind](https://github.com/learning-at-home/hivemind) - Decentralized deep learning in PyTorch. Built to train models on thousands of volunteers across the world.

* [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax) - Model parallel transformers in JAX and Haiku.

* [Nebullvm](https://github.com/nebuly-ai/nebullvm) - An open-source tool designed to speed up AI inference in just a few lines of code.

* [ncnn](https://github.com/Tencent/ncnn) - A high-performance neural network inference framework optimized for the mobile platform.

* [Turi Create](https://github.com/apple/turicreate) **(not actively updated)** - A machine learning library for deployment on MacOS/iOS.

* [Apache SINGA](https://github.com/apache/singa) **(not actively updated)** - A distributed deep learning platform.

* [BytePS](https://github.com/bytedance/byteps) **(not actively updated)** - A high performance and generic framework for distributed DNN training.

* [MMdnn](https://github.com/microsoft/MMdnn) **(not actively updated)** - MMdnn is a set of tools to help users inter-operate among different deep learning frameworks.

### Auto ML & Hyperparameter Optimization

* [NNI](https://github.com/microsoft/nni) - An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning.

* [AutoKeras](https://github.com/keras-team/autokeras) - AutoML library for deep learning.

* [KerasTuner](https://github.com/keras-team/keras-tuner) - An easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search.

* [Talos](https://github.com/autonomio/talos) - Hyperparameter Optimization for TensorFlow, Keras and PyTorch.

* [Distiller](https://github.com/IntelLabs/distiller) - Neural Network Distiller by Intel AI Lab: a Python package for neural network compression research.

* [Hyperas](https://github.com/maxpumperla/hyperas) **(not actively updated)** - A very simple wrapper for convenient hyperparameter optimization for Keras.

* [Model Search](https://github.com/google/model_search) **(not actively updated)** - A framework that implements AutoML algorithms for model architecture search at scale.

### Interpretability & Adversarial Training

* [AI Explainability 360](https://github.com/Trusted-AI/AIX360) - An open-source library that supports interpretability and explainability of datasets and machine learning models.

* [explainerdashboard](https://github.com/oegedijk/explainerdashboard) - Quickly build Explainable AI dashboards that show the inner workings of so-called "blackbox" machine learning models.

* [iNNvestigate](https://github.com/albermax/innvestigate) - A toolbox to innvestigate neural networks' predictions.

* [Foolbox](https://github.com/bethgelab/foolbox) - A Python toolbox to create adversarial examples that fool neural networks in PyTorch, TensorFlow, and JAX.

* [AdvBox](https://github.com/advboxes/AdvBox) - A toolbox to generate adversarial examples that fool neural networks in PaddlePaddle、PyTorch、Caffe2、MxNet、Keras、TensorFlow.

* [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - Python Library for Machine Learning Security - Evasion, Poisoning, Extraction, Inference.

* [CleverHans](https://github.com/cleverhans-lab/cleverhans) - An adversarial example library for constructing attacks, building defenses, and benchmarking both.

### Anomaly Detection & Others

* [Anomalib](https://github.com/openvinotoolkit/anomalib) - An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference.

* [Gradio](https://github.com/gradio-app/gradio) - An open-source Python library that is used to build machine learning and data science demos and web applications.

* [Traingenerator](https://github.com/jrieke/traingenerator) - Generates custom template code for PyTorch & sklearn, using a simple web UI built with streamlit.

* [Fairlearn](https://github.com/fairlearn/fairlearn) - A Python package to assess and improve fairness of machine learning models.

* [AI Fairness 360](https://github.com/Trusted-AI/AIF360) - A comprehensive set of fairness metrics for datasets and machine learning models, explanations for these metrics, and algorithms to mitigate bias in datasets and models.

## Machine Learning Framework

### General Purpose Framework

* [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine learning toolkit for Python.
  * <details open><summary>Related projects:</summary>

    * [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - A python package offering a number of re-sampling techniques commonly used in datasets showing strong between-class imbalance
    * [category_encoders](https://github.com/scikit-learn-contrib/category_encoders) - A set of scikit-learn-style transformers for encoding categorical variables into numeric by means of different techniques
    * [lightning](https://github.com/scikit-learn-contrib/lightning) - Large-scale linear classification, regression and ranking in Python
    * [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) - Pandas integration with sklearn
    * [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) - A high performance implementation of HDBSCAN clustering
    * [metric-learn](https://github.com/scikit-learn-contrib/metric-learn) - Metric learning algorithms in Python
    * [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) - Sequential model-based optimization with a `scipy.optimize` interface
    * [scikit-image](https://github.com/scikit-image/scikit-image) - Image processing in Python
    * [auto-sklearn](https://github.com/automl/auto-sklearn) - An automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.
    * [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) - A Python module capable of performing multi-label learning tasks
    * [scikit-lego](https://github.com/koaning/scikit-lego) - Extra blocks for scikit-learn pipelines.
    * [scikit-opt](https://github.com/guofei9987/scikit-opt) - Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Optimization Algorithm,Immune Algorithm, Artificial Fish Swarm Algorithm, Differential Evolution and TSP(Traveling salesman)
    * [sklearn-porter](https://github.com/nok/sklearn-porter) - Transpile trained scikit-learn estimators to C, Java, JavaScript and others.
  </details>

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

* [Autograd](https://github.com/HIPS/autograd) **(no longer maintained)** - Efficiently computes derivatives of numpy code.

* [JAX](https://github.com/google/jax) **(successor of Autograd)** - Automatical differentiation for native Python and NumPy functions, with GPU support.

* [Flax](https://github.com/google/flax) - A high-performance neural network library and ecosystem for JAX that is designed for flexibility.

* [Equinox](https://github.com/patrick-kidger/equinox) - A JAX library based around a simple idea: represent parameterised functions (such as neural networks) as PyTrees.

* [cuML](https://github.com/rapidsai/cuml) - A suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other RAPIDS projects.

* [Mlxtend](https://github.com/rasbt/mlxtend) - A library of extension and helper modules for Python's data analysis and machine learning libraries.

* [River](https://github.com/online-ml/river) - A Python library for online machine learning.

* [FilterPy](https://github.com/rlabbe/filterpy) - Python Kalman filtering and optimal estimation library.

* [igel](https://github.com/nidhaloff/igel) - A delightful machine learning tool that allows you to train, test, and use models without writing code.

* [fklearn](https://github.com/nubank/fklearn) - A machine learning library that uses functional programming principles.

* [SynapseML](https://github.com/microsoft/SynapseML) - An open-source library that simplifies the creation of massively scalable machine learning pipelines.

* [Dask](https://github.com/dask/dask) - A flexible parallel computing library for NumPy, Pandas and Scikit-Learn.
  * <details open><summary>Related projects:</summary>

    * [Distributed](https://github.com/dask/distributed) - A distributed task scheduler for Dask
  </details>

* [H2O](https://github.com/h2oai/h2o-3) - An in-memory platform for distributed, scalable machine learning.

* [autodiff](https://github.com/autodiff/autodiff) - automatic differentiation made easier for C++

* [GoLearn](https://github.com/sjwhitworth/golearn) - Machine Learning for Go.

* [leaves](https://github.com/dmitryikh/leaves) - Pure Go implementation of prediction part for GBRT (Gradient Boosting Regression Trees) models from popular frameworks.

* [go-xgboost](https://github.com/Unity-Technologies/go-xgboost) - XGBoost bindings for golang.

* [DEAP](https://github.com/DEAP/deap) - Distributed Evolutionary Algorithms in Python.

* [ESTool](https://github.com/hardmaru/estool) - Evolution Strategies Tool.

* [mlpack](https://github.com/mlpack/mlpack) **(not actively updated)** - A header-only C++ machine learning library.
  * Language API: *C++, Python, R, Julia, Go*

* [xLearn](https://github.com/aksnzhy/xlearn) **(not actively updated)** - A C++ machine learning library for linear model (LR), factorization machines (FM), and field-aware factorization machines (FFM).

* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) **(not actively updated)** - Fast GBDTs and Random Forests on GPUs.

* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) **(not actively updated)** - A Fast SVM Library on GPUs and CPUs.

* [PyBrain](https://github.com/pybrain/pybrain) - The Python Machine Learning Library.

### Nearest Neighbors & Similarity

* [Annoy](https://github.com/spotify/annoy) - Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.

* [Hnswlib](https://github.com/nmslib/hnswlib) - Header-only C++/python library for fast approximate nearest neighbors.

* [NMSLIB](https://github.com/nmslib/nmslib) - Non-Metric Space Library (NMSLIB): An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces.

* [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) - Benchmarks of approximate nearest neighbor libraries in Python.

* [kmodes](https://github.com/nicodv/kmodes) - Python implementations of the k-modes and k-prototypes clustering algorithms, for clustering categorical data.

### Hyperparameter Search & Gradient-Free Optimization

* [Optuna](https://github.com/optuna/optuna) - An automatic hyperparameter optimization software framework, particularly designed for machine learning.

* [Ax](https://github.com/facebook/Ax) - An accessible, general-purpose platform for understanding, managing, deploying, and automating adaptive experiments.

* [AutoGluon](https://github.com/awslabs/autogluon) - Automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.

* [Nevergrad](https://github.com/facebookresearch/nevergrad) - A Python toolbox for performing gradient-free optimization.

* [MLJAR](https://github.com/mljar/mljar-supervised) - Python package for AutoML on Tabular Data with Feature Engineering, Hyper-Parameters Tuning, Explanations and Automatic Documentation.

* [gplearn](https://github.com/trevorstephens/gplearn) - Genetic Programming in Python, with a scikit-learn inspired API.

* [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) **(not actively updated)** - A Python implementation of global optimization with gaussian processes.

* [Hyperopt](https://github.com/hyperopt/hyperopt) **(not actively updated)** - Distributed Asynchronous Hyperparameter Optimization in Python.

* [Dragonfly](https://github.com/dragonfly/dragonfly) **(not actively updated)** - An open source python library for scalable Bayesian optimization.

### Experiment Management

* [MLflow](https://github.com/mlflow/mlflow) - A platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

* [PyCaret](https://github.com/pycaret/pycaret) - An open-source, low-code machine learning library in Python that automates machine learning workflows.

* [Aim](https://github.com/aimhubio/aim) - An open-source, self-hosted ML experiment tracking tool.

* [Ax](https://github.com/facebook/Ax) - An accessible, general-purpose platform for understanding, managing, deploying, and automating adaptive experiments.

* [labml](https://github.com/labmlai/labml) - Monitor deep learning model training and hardware usage from your mobile phone.

* [ClearML](https://github.com/allegroai/clearml) - Auto-Magical Suite of tools to streamline your ML workflow Experiment Manager, MLOps and Data-Management.

* [DVC](https://github.com/iterative/dvc) - A command line tool and VS Code Extension for data/model version control.

* [Metaflow](https://github.com/Netflix/metaflow) - A human-friendly Python/R library that helps scientists and engineers build and manage real-life data science projects.

* [Weights&Biases](https://github.com/wandb/wandb) - A tool for visualizing and tracking your machine learning experiments.

* [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visual analysis and diagnostic tools to facilitate machine learning model selection.

### Model Interpretation

* [dtreeviz](https://github.com/parrt/dtreeviz) - A python library for decision tree visualization and model interpretation.

* [InterpretML](https://github.com/interpretml/interpret) - An open-source package that incorporates state-of-the-art machine learning interpretability techniques.

* [Shapash](https://github.com/MAIF/shapash) - A Python library which aims to make machine learning interpretable and understandable by everyone.

* [Alibi](https://github.com/SeldonIO/alibi) - An open source Python library aimed at machine learning model inspection and interpretation.

* [PyCM](https://github.com/sepandhaghighi/pycm) - Multi-class confusion matrix library in Python.

### Anomaly Detection

* [PyOD](https://github.com/yzhao062/pyod) - A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection).

* [Alibi Detect](https://github.com/SeldonIO/alibi-detect) - Algorithms for outlier, adversarial and drift detection.

## Computer Vision

### General Purpose CV

* [OpenCV](https://github.com/opencv/opencv) - Open Source Computer Vision Library.
  * <details open><summary>Related projects:</summary>

    * [opencv-python](https://github.com/opencv/opencv-python) - Pre-built CPU-only OpenCV packages for Python.
    * [opencv_contrib](https://github.com/opencv/opencv_contrib) - Repository for OpenCV's extra modules.
    * [CVAT](https://github.com/opencv/cvat) - Annotate better with CVAT, the industry-leading data engine for machine learning. Used and trusted by teams at any scale, for data of any scale.
  </details>

* [OMMCV](https://github.com/open-mmlab/mmcv) - OpenMMLab Computer Vision Foundation.
  * <details open><summary>Related projects:</summary>

    * [MMClassification](https://github.com/open-mmlab/mmclassification) - OpenMMLab Image Classification Toolbox and Benchmark
    * [MMDetection](https://github.com/open-mmlab/mmdetection) - OpenMMLab Detection Toolbox and Benchmark
    * [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - OpenMMLab's next-generation platform for general 3D object detection
    * [MMOCR](https://github.com/open-mmlab/mmocr) - OpenMMLab Text Detection, Recognition and Understanding Toolbox
    * [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) - OpenMMLab Semantic Segmentation Toolbox and Benchmark.
    * [MMTracking](https://github.com/open-mmlab/mmtracking) - OpenMMLab Video Perception Toolbox
    * [MMPose](https://github.com/open-mmlab/mmpose) - OpenMMLab Pose Estimation Toolbox and Benchmark
    * [MMSkeleton](https://github.com/open-mmlab/mmskeleton) - A OpenMMLAB toolbox for human pose estimation, skeleton-based action recognition, and action synthesis
    * [MMGeneration](https://github.com/open-mmlab/mmgeneration) - MMGeneration is a powerful toolkit for generative models, based on PyTorch and MMCV
    * [MMEditing](https://github.com/open-mmlab/mmediting) - MMEditing is a low-level vision toolbox based on PyTorch, supporting super-resolution, inpainting, matting, video interpolation, etc
    * [MMDeploy](https://github.com/open-mmlab/mmdeploy) - OpenMMLab Model Deployment Framework
    * [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) - OpenPCDet Toolbox for LiDAR-based 3D Object Detection
  </details>

* [Lightly](https://github.com/lightly-ai/lightly) - A computer vision framework for self-supervised learning, based on PyTorch.

* [GluonCV](https://github.com/dmlc/gluon-cv) - A high-level computer vision library for PyTorch and MXNet.

* [Scenic](https://github.com/google-research/scenic) - A codebase with a focus on research around attention-based models for computer vision, based on JAX and Flax.

* [Kornia](https://github.com/kornia/kornia) - Open source differentiable computer vision library, based on PyTorch.

* [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) - A collection of CV models, scripts, pretrained weights, based on PyTorch.

* [vit-pytorch](https://github.com/lucidrains/vit-pytorch) - A collection of Vision Transformer implementations, based on PyTorch.

* [vit-tensorflow](https://github.com/taki0112/vit-tensorflow) - A collection of Vision Transformer implementations, based on TensorFlow.

* [ccv](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library.

* [TorchCV](https://github.com/donnyyou/torchcv) **(not actively updated)** - A PyTorch-Based Framework for Deep Learning in Computer Vision.

### Classification & Detection & Tracking

* [Detectron](https://github.com/facebookresearch/Detectron/) **(no longer maintained)** - A research platform for object detection research, implementing popular algorithms by Facebook, based on Caffe2.

* [Detectron2](https://github.com/facebookresearch/detectron2) **(successor of Detectron)** - A platform for object detection, segmentation and other visual recognition tasks, based on PyTorch.

* [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) - Real-Time and Accurate Full-Body Multi-Person Pose Estimation&Tracking System.

* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - Real-time multi-person keypoint detection library for body, face, hands, and foot estimation.

* [OpenPose Unity Plugin](https://github.com/CMU-Perceptual-Computing-Lab/openpose_unity_plugin) - A wrapper of the OpenPose library for Unity users.

* [Norfair](https://github.com/tryolabs/norfair) - Lightweight Python library for adding real-time multi-object tracking to any detector.

* [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) - YOLOv4 / Scaled-YOLOv4 / YOLO - Neural Networks for Object Detection (Windows and Linux version of Darknet ).

* [pjreddie/darknet](https://github.com/pjreddie/darknet) - Convolutional Neural Networks.

* [ClassyVision](https://github.com/facebookresearch/ClassyVision) - An end-to-end framework for image and video classification, based on PyTorch.

* [pycls](https://github.com/facebookresearch/pycls) - Codebase for Image Classification Research, based on PyTorch.

* [CenterNet](https://github.com/xingyizhou/CenterNet) - Object detection, 3D detection, and pose estimation using center point detection.

* [SlowFast](https://github.com/facebookresearch/SlowFast) - Video understanding codebase from FAIR, based on PyTorch.

* [SAHI](https://github.com/obss/sahi) - Platform agnostic sliced/tiled inference + interactive ui + error analysis plots for object detection and instance segmentation.

* [libfacedetection](https://github.com/ShiqiYu/libfacedetection) - An open source library for face detection in images. The face detection speed can reach 1000FPS.

* [openbr](https://github.com/biometrics/openbr) - Open Source Biometrics, Face Recognition.

* [InsightFace](https://github.com/deepinsight/insightface) - An open source 2D&3D deep face analysis toolbox, based on PyTorch and MXNet.

* [Deepface](https://github.com/serengil/deepface) - A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python.

* [deepfakes_faceswap](https://github.com/deepfakes/faceswap) - A tool that utilizes deep learning to recognize and swap faces in pictures and videos.

* [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) - 1MB lightweight face detection model.

* [face_classification](https://github.com/oarriaga/face_classification) **(no longer maintained)** - Real-time face detection and emotion/gender classification using fer2013/imdb datasets with a keras CNN model and openCV.

* [PAZ](https://github.com/oarriaga/paz) **(successor of face_classification)** - Hierarchical perception library in Python for pose estimation, object detection, instance segmentation, keypoint estimation, face recognition, etc.

* [MenpoBenchmark](https://github.com/jiankangdeng/MenpoBenchmark) - Multi-pose 2D and 3D Face Alignment & Tracking.

* [CaImAn](https://github.com/flatironinstitute/CaImAn) - Computational toolbox for large scale Calcium Imaging Analysis, including movie handling, motion correction, source extraction, spike deconvolution and result visualization.

* [segmentation_models](https://github.com/qubvel/segmentation_models) **(not actively updated)** - Python library with Neural Networks for Image Segmentation based on Keras and TensorFlow.

* [OpenFace](https://github.com/cmusatyalab/openface) **(not actively updated)** - Face recognition with deep neural networks.

* [Face Recognition](https://github.com/ageitgey/face_recognition) **(not actively updated)** - A facial recognition api for Python and the command line.

* [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) **(not actively updated)** - A state-of-the art tool intended for facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation.

* [hgpvision/darknet](https://github.com/hgpvision/darknet) **(not actively updated)** - darknet深度学习框架源码分析：详细中文注释，涵盖框架原理与实现语法分析

### OCR

* [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Ready-to-use OCR with 80+ supported languages and all popular writing scripts.

* [Python-tesseract](https://github.com/madmaze/pytesseract) - A Python wrapper for Google's Tesseract-OCR Engine.

* [tesserocr](https://github.com/sirfz/tesserocr) - A simple, Pillow-friendly, wrapper around the tesseract-ocr API for OCR.

* [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) - Text recognition (optical character recognition) with deep learning methods.

* [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) - Adds an OCR text layer to scanned PDF files, allowing them to be searched.

* [LayoutParser](https://github.com/Layout-Parser/layout-parser) - A Unified Toolkit for Deep Learning Based Document Image Analysis, based on Detectron2.

* [chineseocr](https://github.com/chineseocr/chineseocr) - yolo3+ocr

* [HyperLPR](https://github.com/szad670401/HyperLPR) - 基于深度学习高性能中文车牌识别

* [deep_ocr](https://github.com/JinpengLI/deep_ocr) **(not actively updated)** - make a better chinese character recognition OCR than tesseract

* [chinese_ocr](https://github.com/YCG09/chinese_ocr) **(not actively updated)** - CTPN + DenseNet + CTC based end-to-end Chinese OCR implemented using tensorflow and keras.

* [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract) **(no longer maintained)** - A set of tools for extracting tables from PDF files helping to do data mining on (OCR-processed) scanned documents.

* [CHINESE-OCR](https://github.com/xiaofengShi/CHINESE-OCR) **(not actively updated)** - 运用tf实现自然场景文字检测,keras/pytorch实现ctpn+crnn+ctc实现不定长场景文字OCR识别

* [EasyPR](https://github.com/liuruoze/EasyPR) - 一个开源的中文车牌识别系统

* [License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9) **(not actively updated)** - 中文车牌识别

### Image / Video Generation

* [DALL·E Flow](https://github.com/jina-ai/dalle-flow) - A Human-in-the-Loop workflow for creating HD images from text.

* [DALL·E Mini](https://github.com/borisdayma/dalle-mini) - Generate images from a text prompt.

* [GAN Lab](https://github.com/poloclub/ganlab) - An Interactive, Visual Experimentation Tool for Generative Adversarial Networks.

* [DeepFaceLab](https://github.com/iperov/DeepFaceLab) - DeepFaceLab is the leading software for creating deepfakes.

* [DeOldify](https://github.com/jantic/DeOldify) - A Deep Learning based project for colorizing and restoring old images (and video!)

* [waifu2x](https://github.com/nagadomi/waifu2x) - Image Super-Resolution for Anime-Style Art.

* [Kubric](https://github.com/google-research/kubric) - A data generation pipeline for creating semi-realistic synthetic multi-object videos with rich annotations such as instance segmentation masks, depth maps, and optical flow.

* [benchmark_VAE](https://github.com/clementchadebec/benchmark_VAE) - Implements some of the most common (Variational) Autoencoder models under a unified implementation.

* [FastPhotoStyle](https://github.com/NVIDIA/FastPhotoStyle) **(not actively updated)** - Style transfer, deep learning, feature transform.

* [Real-Time-Person-Removal](https://github.com/jasonmayes/Real-Time-Person-Removal) **(not actively updated)** - Removing people from complex backgrounds in real time using TensorFlow.js in the web browser.

* [MUNIT](https://github.com/NVlabs/MUNIT) **(no longer maintained)** - Multimodal Unsupervised Image-to-Image Translation.

* [pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo) **(not actively updated)** - A mix of GAN implementations including progressive growing.

* [deepcolor](https://github.com/kvfrans/deepcolor) **(not actively updated)** - Automatic coloring and shading of manga-style lineart, using Tensorflow + cGANs.

## Natural Language Processing

### General Purpose NLP

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

* [NLP Architect](https://github.com/IntelLabs/nlp-architect) - A Deep Learning NLP/NLU library by Intel AI Lab, based on PyTorch and TensorFlow.

* [LightSeq](https://github.com/bytedance/lightseq) - A high performance training and inference library for sequence processing and generation implemented in CUDA, for Fairseq and HuggingFace Transformers.

* [FudanNLP](https://github.com/FudanNLP/fnlp) **(no longer maintained)** - Toolkit for Chinese natural language processing.

* [fastNLP](https://github.com/fastnlp/fastNLP) **(successor of FudanNLP)** - A Modularized and Extensible NLP Framework for PyTorch and PaddleNLP.

* [Rubrix](https://github.com/recognai/rubrix) - A production-ready Python framework for exploring, annotating, and managing data in NLP projects.

* [Gensim](https://github.com/RaRe-Technologies/gensim) - A Python library for topic modelling, document indexing and similarity retrieval with large corpora, based on NumPy and SciPy.

* [CLTK](https://github.com/cltk/cltk) - A Python library offering natural language processing for pre-modern languages.

* [OpenNRE](https://github.com/thunlp/OpenNRE) - An open-source and extensible toolkit that provides a unified framework to implement relation extraction models.

* [minGPT](https://github.com/karpathy/minGPT) - A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training.

* [HanLP](https://github.com/hankcs/HanLP) - 中文分词 词性标注 命名实体识别 依存句法分析 成分句法分析 语义依存分析 语义角色标注 指代消解 风格转换 语义相似度 新词发现 关键词短语提取 自动摘要 文本分类聚类 拼音简繁转换 自然语言处理

* [LAC](https://github.com/baidu/lac) - 百度NLP：分词，词性标注，命名实体识别，词重要性

* [AllenNLP](https://github.com/allenai/allennlp) **(not actively updated)** - An open source natural language processing library, based on PyTorch.

* [GluonNLP](https://github.com/dmlc/gluon-nlp) **(not actively updated)** - A high-level NLP toolkit, based on MXNet.

* [jiant](https://github.com/nyu-mll/jiant) **(no longer maintained)** - The multitask and transfer learning toolkit for natural language processing research.

* [fastText](https://github.com/facebookresearch/fastText) **(not actively updated)** - A library for efficient learning of word representations and sentence classification.

* [TextBlob](https://github.com/sloria/TextBlob) **(not actively updated)** - A Python library for processing textual data.

* [jieba](https://github.com/fxsjy/jieba) **(not actively updated)** - 结巴中文分词

* [SnowNLP](https://github.com/isnowfy/snownlp) **(not actively updated)** - Python library for processing Chinese text.

### Conversation & Translation

* [SpeechBrain](https://github.com/speechbrain/speechbrain) - An open-source and all-in-one conversational AI toolkit based on PyTorch.

* [NeMo](https://github.com/NVIDIA/NeMo) - A toolkit for conversational AI, based on PyTorch.

* [Sockeye](https://github.com/awslabs/sockeye) - An open-source sequence-to-sequence framework for Neural Machine Translation, based on PyTorch.

* [DeepPavlov](https://github.com/deeppavlov/DeepPavlov) - An open-source conversational AI library built on TensorFlow, Keras and PyTorch.

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) - The PyTorch version of the OpenNMT project, an open-source neural machine translation framework.

* [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) - The TensorFlow version of the OpenNMT project, an open-source neural machine translation framework.

* [Rasa](https://github.com/RasaHQ/rasa) - Open source machine learning framework to automate text- and voice-based conversations.

* [SentencePiece](https://github.com/google/sentencepiece) - Unsupervised text tokenizer for Neural Network-based text generation.

* [subword-nmt](https://github.com/rsennrich/subword-nmt) - Unsupervised Word Segmentation for Neural Machine Translation and Text Generation.

* [OpenPrompt](https://github.com/thunlp/OpenPrompt) - An Open-Source Framework for Prompt-Learning.

* [sumy](https://github.com/miso-belica/sumy) - Module for automatic summarization of text documents and HTML pages.

* [chatbot](https://github.com/zhaoyingjun/chatbot) - 一个可以自己进行训练的中文聊天机器人， 根据自己的语料训练出自己想要的聊天机器人，可以用于智能客服、在线问答、智能聊天等场景。

* [AI-Writer](https://github.com/BlinkDL/AI-Writer) - AI 写小说，生成玄幻和言情网文等等。中文预训练生成模型。

* [seq2seq-couplet](https://github.com/wb14123/seq2seq-couplet) - 用深度学习对对联。

* [FARM](https://github.com/deepset-ai/FARM) **(not actively updated)** - Fast & easy transfer learning for NLP, which focuses on Question Answering.

* [Haystack](https://github.com/deepset-ai/haystack) **(successor of FARM)** - A high-level natural language processing library for deployment and production, based on PyTorch and HuggingFace Transformers.

* [XLM](https://github.com/facebookresearch/XLM) **(not actively updated)** - PyTorch original implementation of Cross-lingual Language Model Pretraining.

### Speech & Audio

* [TTS](https://github.com/coqui-ai/TTS) - A library for advanced Text-to-Speech generation.

* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) - A Python library for audio feature extraction, classification, segmentation and applications.

* [Porcupine](https://github.com/Picovoice/porcupine) - On-device wake word detection powered by deep learning.

* [MuseGAN](https://github.com/salu133445/musegan) - An AI for Music Generation.

* [wav2letter++](https://github.com/flashlight/wav2letter) **(no longer maintained)** - Facebook AI Research's Automatic Speech Recognition Toolkit.

* [Magenta](https://github.com/magenta/magenta) **(no longer maintained)** - Music and Art Generation with Machine Intelligence.

* [SpeechRecognition](https://github.com/Uberi/speech_recognition) **(not actively updated)** - Library for performing speech recognition, with support for several engines and APIs, online and offline.

### Others

* [Spleeter](https://github.com/deezer/spleeter) - A source separation library with pretrained models, based on TensorFlow.

* [Language Interpretability Tool](https://github.com/PAIR-code/lit) - Interactively analyze NLP models for model understanding in an extensible and framework agnostic interface.

* [TextAttack](https://github.com/QData/TextAttack) - A Python framework for adversarial attacks, data augmentation, and model training in NLP.

* [CheckList](https://github.com/marcotcr/checklist) - Behavioral Testing of NLP models with CheckList.

## Reinforcement Learning

* [OpenAI Gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms by OpenAI.

* [DeepMind Lab](https://github.com/deepmind/lab) - A customisable 3D platform for agent-based AI research.

* [TF-Agents](https://github.com/tensorflow/agents) - A reliable, scalable and easy to use TensorFlow library for Contextual Bandits and Reinforcement Learning.

* [TensorLayer](https://github.com/tensorlayer/TensorLayer) - A novel TensorFlow-based deep learning and reinforcement learning library designed for researchers and engineers.

* [Tensorforce](https://github.com/tensorforce/tensorforce) - A TensorFlow library for applied reinforcement learning.

* [Acme](https://github.com/deepmind/acme) - A research framework for reinforcement learning by DeepMind.

* [RLax](https://github.com/deepmind/rlax) - A library built on top of JAX that exposes useful building blocks for implementing reinforcement learning agents.

* [ReAgent](https://github.com/facebookresearch/ReAgent) - An open source end-to-end platform for applied reinforcement learning by Facebook.

* [Dopamine](https://github.com/google/dopamine) - A research framework for fast prototyping of reinforcement learning algorithms.

* [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) - A fast, flexible, online, and active learning solution for solving complex interactive machine learning problems.

* [PFRL](https://github.com/pfnet/pfrl) - A PyTorch-based deep reinforcement learning library.

* [garage](https://github.com/rlworkgroup/garage) - A toolkit for reproducible reinforcement learning research.

* [PyRobot](https://github.com/facebookresearch/pyrobot) - An Open Source Robotics Research Platform.

* [AirSim](https://github.com/microsoft/AirSim) - Open source simulator for autonomous vehicles built on Unreal Engine / Unity, from Microsoft AI & Research.

* [Self-Driving-Car-in-Video-Games](https://github.com/ikergarcia1996/Self-Driving-Car-in-Video-Games) - A deep neural network that learns to drive in video games.

* [OpenAI Baselines](https://github.com/openai/baselines) **(no longer maintained)** - A set of high-quality implementations of reinforcement learning algorithms.

* [Stable Baselines](https://github.com/hill-a/stable-baselines) **(no longer maintained)** - A fork of OpenAI Baselines, implementations of reinforcement learning algorithms.

* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) **(successor of OpenAI Baselines and Stable Baselines)** - A set of reliable implementations of reinforcement learning algorithms in PyTorch.

* [PySC2](https://github.com/deepmind/pysc2) - StarCraft II Learning Environment.

* [ViZDoom](https://github.com/mwydmuch/ViZDoom) - Doom-based AI Research Platform for Reinforcement Learning from Raw Visual Information.

* [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - The first open-source framework to show the great potential of financial reinforcement learning.

* [AnimalAI-Olympics](https://github.com/beyretb/AnimalAI-Olympics) **(no longer maintained)** - Code repository for the Animal AI Olympics competition.

* [AnimalAI 3](https://github.com/mdcrosby/animal-ai) **(successor of AnimalAI-Olympics)** - AAI supports interdisciplinary research to help better understand human, animal, and artificial cognition.

* [self-driving-car](https://github.com/udacity/self-driving-car) **(no longer maintained)** - The Udacity open source self-driving car project.

## Graph

* [DGL](https://github.com/dmlc/dgl) - An easy-to-use, high performance and scalable Python package for deep learning on graphs for PyTorch, Apache MXNet or TensorFlow.

* [NetworkX](https://github.com/networkx/networkx) - A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

* [igraph](https://github.com/igraph/igraph) - Library for the analysis of networks.

* [python-igraph](https://github.com/igraph/python-igraph) - Python interface for igraph.

* [PyG](https://github.com/pyg-team/pytorch_geometric) - A Graph Neural Network Library based on PyTorch.

* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - Graph Neural Network Library for PyTorch.

* [OGB](https://github.com/snap-stanford/ogb) - Benchmark datasets, data loaders, and evaluators for graph machine learning.

* [Spektral](https://github.com/danielegrattarola/spektral) - A Python library for graph deep learning, based on Keras and TensorFlow.

* [Graph Nets](https://github.com/deepmind/graph_nets) - Build Graph Nets in Tensorflow.

* [Graph4nlp](https://github.com/graph4ai/graph4nlp) - A library for the easy use of Graph Neural Networks for NLP (DLG4NLP).

* [Jraph](https://github.com/deepmind/jraph) - A Graph Neural Network Library in Jax.

* [cuGraph](https://github.com/rapidsai/cugraph) - A collection of GPU accelerated graph algorithms that process data found in GPU DataFrames (cuDF).

* [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding) - Implementation and experiments of graph embedding algorithms.

* [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns) - Repository for benchmarking graph neural networks.

* [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) **(not actively updated)** - Generate embeddings from large-scale graph-structured data,  based on PyTorch.

* [TensorFlow Graphics](https://github.com/tensorflow/graphics) **(not actively updated)** - Differentiable Graphics Layers for TensorFlow.

* [StellarGraph](https://github.com/stellargraph/stellargraph) **(not actively updated)** - A Python library for machine learning on graphs and networks.

## Causal Inference

* [EconML](https://github.com/microsoft/EconML) - A Python package for estimating heterogeneous treatment effects from observational data via machine learning.

* [Causal ML](https://github.com/uber/causalml) - Uplift modeling and causal inference with machine learning algorithms.

* [DoWhy](https://github.com/py-why/dowhy) - A Python library for causal inference that supports explicit modeling and testing of causal assumptions.

* [CausalNex](https://github.com/quantumblacklabs/causalnex) - A Python library that helps data scientists to infer causation rather than observing correlation.

* [causallib](https://github.com/IBM/causallib) - A Python package for modular causal inference analysis and model evaluations.

* [pylift](https://github.com/wayfair/pylift) - Uplift modeling package.

* [grf](https://github.com/grf-labs/grf) - Generalized Random Forests.

* [DoubleML](https://github.com/DoubleML/doubleml-for-py) - Double Machine Learning in Python.

* [Causality](https://github.com/akelleh/causality) - Tools for causal analysis.

* [YLearn](https://github.com/DataCanvasIO/YLearn) - A python package for causal inference.

## Recommendation, Advertisement & Ranking

* [Recommenders](https://github.com/microsoft/recommenders) - Best Practices on Recommendation Systems.

* [Surprise](https://github.com/NicolasHug/Surprise) - A Python scikit for building and analyzing recommender systems.

* [RecLearn](https://github.com/ZiyaoGeng/RecLearn) - Recommender Learning with Tensorflow2.x.

* [Implicit](https://github.com/benfred/implicit) - Fast Python Collaborative Filtering for Implicit Feedback Datasets.

* [LightFM](https://github.com/lyst/lightfm) - A Python implementation of LightFM, a hybrid recommendation algorithm.

* [RecBole](https://github.com/RUCAIBox/RecBole) - A unified, comprehensive and efficient recommendation library for reproducing and developing recommendation algorithms.

* [DeepCTR](https://github.com/shenweichen/DeepCTR) - Easy-to-use,Modular and Extendible package of deep-learning based CTR models.

* [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) - Easy-to-use,Modular and Extendible package of deep-learning based CTR models.

* [deep-ctr-prediction](https://github.com/qiaoguan/deep-ctr-prediction) - CTR prediction models based on deep learning.

* [RecSys](https://github.com/mJackie/RecSys) - 计算广告/推荐系统/机器学习(Machine Learning)/点击率(CTR)/转化率(CVR)预估/点击率预估。

* [AI-RecommenderSystem](https://github.com/zhongqiangwu960812/AI-RecommenderSystem) - 推荐系统领域的一些经典算法模型。

* [Recommend-System-TF2.0](https://github.com/jc-LeeHub/Recommend-System-tf2.0) - 经典推荐算法的原理解析及代码实现。

* [SparkCTR](https://github.com/wzhe06/SparkCTR) **(not actively updated)** - CTR prediction model based on spark(LR, GBDT, DNN).

* [Awesome-RecSystem-Models](https://github.com/JianzhouZhan/Awesome-RecSystem-Models) **(not actively updated)** - Implements of Awesome RecSystem Models with PyTorch/TF2.0.

* [Deep_Rec](https://github.com/Shicoder/Deep_Rec) **(not actively updated)** - 推荐算法相关代码、文档、资料

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

* [Qlib](https://github.com/microsoft/qlib) - An AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

* [IB-insync](https://github.com/erdewit/ib_insync) - Python sync/async framework for Interactive Brokers API.

* [ffn](https://github.com/pmorissette/ffn) - A financial function library for Python.

* [bt](https://github.com/pmorissette/bt) - A flexible backtesting framework for Python used to test quantitative trading strategies, based on ffn.

* [finmarketpy](https://github.com/cuemacro/finmarketpy) - Python library for backtesting trading strategies & analyzing financial markets.

* [TensorTrade](https://github.com/tensortrade-org/tensortrade) - An open source reinforcement learning framework for training, evaluating, and deploying robust trading agents, based on TensorFlow.

* [TF Quant Finance](https://github.com/google/tf-quant-finance) - High-performance TensorFlow library for quantitative finance.

* [Pandas TA](https://github.com/twopirllc/pandas-ta) - An easy to use library that leverages the Pandas package with more than 130 Indicators and Utility functions and more than 60 TA Lib Candlestick Patterns.

* [pyts](https://github.com/johannfaouzi/pyts) **(not actively updated)** - A Python package for time series classification.

* [CryptoSignal](https://github.com/CryptoSignal/Crypto-Signal) **(not actively updated)** - A command line tool that automates your crypto currency Technical Analysis (TA).

* [Catalyst](https://github.com/scrtlabs/catalyst) **(no longer maintained)** - An algorithmic trading library for crypto-assets written in Python.

## Other Machine Learning Applications

* [AlphaFold](https://github.com/deepmind/alphafold) - Open source code for AlphaFold.

* [OpenFold](https://github.com/aqlaboratory/openfold) - Trainable, memory-efficient, and GPU-friendly PyTorch reproduction of AlphaFold 2.

* [DeepChem](https://github.com/deepchem/deepchem) - Democratizing Deep-Learning for Drug Discovery, Quantum Chemistry, Materials Science and Biology.

* [Apollo](https://github.com/ApolloAuto/apollo) - An open autonomous driving platform.

* [OpenCog](https://github.com/opencog/opencog) - A framework for integrated Artificial Intelligence & Artificial General Intelligence (AGI).

* [Screenshot-to-code](https://github.com/emilwallner/Screenshot-to-code) - A neural network that transforms a design mock-up into a static website.

* [PennyLane](https://github.com/PennyLaneAI/pennylane) - A cross-platform Python library for differentiable programming of quantum computers.

* [OR-Tools](https://github.com/google/or-tools) - Google's Operations Research tools.

* [CARLA](https://github.com/carla-simulator/carla) **(not actively updated)** - An open-source simulator for autonomous driving research.

* [convnet-burden](https://github.com/albanie/convnet-burden) **(not actively updated)** - Memory consumption and FLOP count estimates for convnets.

* [gradient-checkpointing](https://github.com/cybertronai/gradient-checkpointing) **(no longer maintained)** - Make huge neural nets fit in memory.

## Linear Algebra / Statistics Toolkit

### General Purpose Tensor Library

* [NumPy](https://github.com/numpy/numpy) - The fundamental package for scientific computing with Python.

* [SciPy](https://github.com/scipy/scipy) - An open-source software for mathematics, science, and engineering in Python.

* [SymPy](https://github.com/sympy/sympy) - A computer algebra system written in pure Python.

* [ArrayFire](https://github.com/arrayfire/arrayfire) - A general-purpose tensor library that simplifies the process of software development for the parallel architectures found in CPUs, GPUs, and other hardware acceleration devices

* [CuPy](https://github.com/cupy/cupy) - A NumPy/SciPy-compatible array library for GPU-accelerated computing with Python.

* [PyCUDA](https://github.com/inducer/pycuda) - Pythonic Access to CUDA, with Arrays and Algorithms.

* [Numba](https://github.com/numba/numba) - NumPy aware dynamic Python compiler using LLVM.

* [xtensor](https://github.com/xtensor-stack/xtensor) - C++ tensors with broadcasting and lazy computing.

* [Halide](https://github.com/halide/Halide) - A language for fast, portable data-parallel computation.

* [NumExpr](https://github.com/pydata/numexpr) - Fast numerical array expression evaluator for Python, NumPy, PyTables, pandas, bcolz and more.

* [OpenBLAS](https://github.com/xianyi/OpenBLAS) - An optimized BLAS library based on GotoBLAS2 1.13 BSD version.

* [Bottleneck](https://github.com/pydata/bottleneck) - Fast NumPy array functions written in C.

* [Enoki](https://github.com/mitsuba-renderer/enoki) - Structured vectorization and differentiation on modern processor architectures.

* [Mars](https://github.com/mars-project/mars) - A tensor-based unified framework for large-scale data computation which scales numpy, pandas, scikit-learn and many other libraries.

* [TensorLy](https://github.com/tensorly/tensorly) - A Python library that aims at making tensor learning simple and accessible.

* [Pythran](https://github.com/serge-sans-paille/pythran) - An ahead of time compiler for a subset of the Python language, with a focus on scientific computing.

* [Patsy](https://github.com/pydata/patsy) **(no longer maintained)** - Describing statistical models in Python using symbolic formulas.

* [Formulaic](https://github.com/matthewwardrop/formulaic) **(successor of Patsy)** - A high-performance implementation of Wilkinson formulas for Python.

* [Theano](https://github.com/Theano/Theano) **(no longer maintained)** - A Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.

* [Aesara](https://github.com/aesara-devs/aesara) **(successor of Theano)** - A Python library that allows one to define, optimize/rewrite, and evaluate mathematical expressions, especially ones involving multi-dimensional arrays.

* [einops](https://github.com/arogozhnikov/einops) - A tensor operation library for NumPy, PyTorch, TensorFlow and JAX.

* [FBGEMM](https://github.com/pytorch/FBGEMM) - A low-precision, high-performance matrix-matrix multiplications and convolution library for server-side inference.

* [taco](https://github.com/tensor-compiler/taco) - A C++ library that computes tensor algebra expressions on sparse and dense tensors.

* [Joblib](https://github.com/joblib/joblib) - Running Python functions as pipeline jobs, with optimizations for numpy.

* [Fastor](https://github.com/romeric/Fastor) - A lightweight high performance tensor algebra framework for modern C++.

* [TiledArray](https://github.com/ValeevGroup/tiledarray) - A massively-parallel, block-sparse tensor framework written in C++.

* [CTF](https://github.com/cyclops-community/ctf) - Cyclops Tensor Framework: parallel arithmetic on multidimensional arrays.

* [Blitz++](https://github.com/blitzpp/blitz) **(not actively updated)** - Multi-Dimensional Array Library for C++.

* [juanjosegarciaripoll/tensor](https://github.com/juanjosegarciaripoll/tensor) **(not actively updated)** - C++ library for numerical arrays and tensor objects and operations with them, designed to allow Matlab-style programming.

* [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) **(not actively updated)** - BLAS extension to xtensor.

### Tensor Similarity & Dimension Reduction

* [Milvus](https://github.com/milvus-io/milvus) - An open-source vector database built to power embedding similarity search and AI applications.

* [Faiss](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors.

* [FLANN](https://github.com/flann-lib/flann) - Fast Library for Approximate Nearest Neighbors

* [openTSNE](https://github.com/pavlin-policar/openTSNE) - Extensible, parallel Python implementations of t-SNE.

* [UMAP](https://github.com/lmcinnes/umap) - Uniform Manifold Approximation and Projection, a dimension reduction technique that can be used for visualisation similarly to t-SNE.

### Statistical Toolkit

* [Statsmodels](https://github.com/statsmodels/statsmodels) - Statistical modeling and econometrics in Python.

* [shap](https://github.com/slundberg/shap) - A game theoretic approach to explain the output of any machine learning model.

* [Pyro](https://github.com/pyro-ppl/pyro) - Deep universal probabilistic programming with Python and PyTorch.

* [GPyTorch](https://github.com/cornellius-gp/gpytorch) - A highly efficient and modular implementation of Gaussian Processes in PyTorch.

* [PyMC](https://github.com/pymc-devs/pymc) - Probabilistic Programming in Python: Bayesian Modeling and Probabilistic Machine Learning with Aesara.

* [hmmlearn](https://github.com/hmmlearn/hmmlearn) - Hidden Markov Models in Python, with scikit-learn like API.

* [emcee](https://github.com/dfm/emcee) - The Python ensemble sampling toolkit for affine-invariant Markov chain Monte Carlo (MCMC).

* [pgmpy](https://github.com/pgmpy/pgmpy) - A python library for working with Probabilistic Graphical Models.

* [pomegranate](https://github.com/jmschrei/pomegranate) - Fast, flexible and easy to use probabilistic modelling in Python.

* [Orbit](https://github.com/uber/orbit) - A Python package for Bayesian forecasting with object-oriented design and probabilistic models under the hood.

* [GPflow](https://github.com/GPflow/GPflow) - Gaussian processes in TensorFlow.

* [ArviZ](https://github.com/arviz-devs/arviz) - A Python package for exploratory analysis of Bayesian models.

* [POT](https://github.com/PythonOT/POT) - Python Optimal Transport.

* [Edward](https://github.com/blei-lab/edward) **(not actively updated)** - A probabilistic programming language in TensorFlow. Deep generative models, variational inference.

### Others

* [torchdiffeq](https://github.com/rtqichen/torchdiffeq) - Differentiable ordinary differential equation (ODE) solvers with full GPU support and O(1)-memory backpropagation.

* [deal.II](https://github.com/dealii/dealii) - A C++ program library targeted at the computational solution of partial differential equations using adaptive finite elements.

* [Neural ODEs](https://github.com/msurtsukov/neural-ode) - Jupyter notebook with Pytorch implementation of Neural Ordinary Differential Equations.

* [Quantum](https://github.com/microsoft/Quantum) - Microsoft Quantum Development Kit Samples.

## Data Processing

### Data Representation

* [pandas](https://github.com/pandas-dev/pandas) - Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R data.frame objects, statistical functions, and much more.

* [cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame Library.

* [Polars](https://github.com/pola-rs/polars) - Fast multi-threaded DataFrame library in Rust, Python and Node.js.

* [Modin](https://github.com/modin-project/modin) - Scale your Pandas workflows by changing a single line of code.

* [Vaex](https://github.com/vaexio/vaex) - Out-of-Core hybrid Apache Arrow/NumPy DataFrame for Python, ML, visualization and exploration of big tabular data at a billion rows per second.

* [PyTables](https://github.com/PyTables/PyTables) - A Python package to manage extremely large amounts of data.

* [Pandaral.lel](https://github.com/nalepae/pandarallel) - A simple and efficient tool to parallelize Pandas operations on all available CPUs.

* [swifter](https://github.com/jmcarpenter2/swifter) - A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner.

* [datatable](https://github.com/h2oai/datatable) - A Python package for manipulating 2-dimensional tabular data structures.

* [xarray](https://github.com/pydata/xarray) - N-D labeled arrays and datasets in Python.

* [Zarr](https://github.com/zarr-developers/zarr-python) - An implementation of chunked, compressed, N-dimensional arrays for Python.

* [Python Sorted Containers](https://github.com/grantjenks/python-sortedcontainers) - Python Sorted Container Types: Sorted List, Sorted Dict, and Sorted Set.

* [Pyrsistent](https://github.com/tobgu/pyrsistent) - Persistent/Immutable/Functional data structures for Python.

* [immutables](https://github.com/MagicStack/immutables) - A high-performance immutable mapping type for Python.

* [DocArray](https://github.com/jina-ai/docarray) - A library for nested, unstructured, multimodal data in transit, including text, image, audio, video, 3D mesh, etc.

* [Texthero](https://github.com/jbesomi/texthero) - A python toolkit to work with text-based dataset, bases on Pandas.

* [ftfy](https://github.com/rspeer/python-ftfy) - Fixes mojibake and other glitches in Unicode text.

* [Box](https://github.com/cdgriffith/Box) - Python dictionaries with advanced dot notation access.

* [bidict](https://github.com/jab/bidict) - The bidirectional mapping library for Python.

* [anytree](https://github.com/c0fec0de/anytree) - Python tree data library.

* [pydantic](https://github.com/pydantic/pydantic) - Data parsing and validation using Python type hints.

* [stockstats](https://github.com/jealous/stockstats) - Supply a wrapper ``StockDataFrame`` based on the ``pandas.DataFrame`` with inline stock statistics/indicators support.

### Data Pre-processing & Loading

* [DALI](https://github.com/NVIDIA/DALI) - A library for data loading and pre-processing to accelerate deep learning applications.

* [Label Studio](https://github.com/heartexlabs/label-studio) - A multi-type data labeling and annotation tool with standardized output format.

* [AugLy](https://github.com/facebookresearch/AugLy) - A data augmentations library for audio, image, text, and video.

* [Albumentations](https://github.com/albumentations-team/albumentations) - A Python library for image augmentation.

* [Augmentor](https://github.com/mdbloice/Augmentor) - Image augmentation library in Python for machine learning.

* [Pillow](https://github.com/python-pillow/Pillow) - The friendly PIL fork (Python Imaging Library).

* [MoviePy](https://github.com/Zulko/moviepy) - Video editing with Python.

* [Open3D](https://github.com/isl-org/Open3D) - A Modern Library for 3D Data Processing.

* [PCL](https://github.com/PointCloudLibrary/pcl) - The Point Cloud Library (PCL) is a standalone, large scale, open project for 2D/3D image and point cloud processing.

* [imutils](https://github.com/PyImageSearch/imutils) - A basic image processing toolkit in Python, based on OpenCV.

* [Towhee](https://github.com/towhee-io/towhee) - Data processing pipelines for neural networks.

* [ffcv](https://github.com/libffcv/ffcv) - A drop-in data loading system that dramatically increases data throughput in model training.

* [NLPAUG](https://github.com/makcedward/nlpaug) - Data augmentation for NLP.

* [Audiomentations](https://github.com/iver56/audiomentations) - A Python library for audio data augmentation.

* [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations) - Fast audio data augmentation in PyTorch, with GPU support.

* [librosa](https://github.com/librosa/librosa) - A python package for music and audio analysis.

* [Pydub](https://github.com/jiaaro/pydub) - Manipulate audio with a simple and easy high level interface.

* [DDSP](https://github.com/magenta/ddsp) - A library of differentiable versions of common DSP functions.

* [TSFRESH](https://github.com/blue-yonder/tsfresh) - Automatic extraction of relevant features from time series.

* [TA](https://github.com/bukosabino/ta) - A Technical Analysis library useful to do feature engineering from financial time series datasets, based on Pandas and NumPy.

* [Featuretools](https://github.com/alteryx/featuretools) - An open source python library for automated feature engineering.

* [Feature-engine](https://github.com/feature-engine/feature_engine) - A Python library with multiple transformers to engineer and select features for use in machine learning models.

* [img2dataset](https://github.com/rom1504/img2dataset) - Easily turn large sets of image urls to an image dataset.

* [Faker](https://github.com/joke2k/faker) - A Python package that generates fake data for you.

* [SDV](https://github.com/sdv-dev/SDV) - Synthetic Data Generation for tabular, relational and time series data.

* [Googletrans](https://github.com/ssut/py-googletrans) - (unofficial) Googletrans: Free and Unlimited Google translate API for Python. Translates totally free of charge.

* [OptBinning](https://github.com/guillermo-navas-palencia/optbinning) - Monotonic binning with constraints. Support batch & stream optimal binning. Scorecard modelling and counterfactual explanations.

* [Scrapy](https://github.com/scrapy/scrapy) - A fast high-level web crawling & scraping framework for Python.

* [pyspider](https://github.com/binux/pyspider) - A Powerful Spider(Web Crawler) System in Python.

* [Instagram Scraper](https://github.com/arc298/instagram-scraper) - Scrapes an instagram user's photos and videos.

* [instaloader](https://github.com/instaloader/instaloader) - Download pictures (or videos) along with their captions and other metadata from Instagram.

* [XueQiuSuperSpider](https://github.com/decaywood/XueQiuSuperSpider) - 雪球股票信息超级爬虫

* [coordtransform](https://github.com/wandergis/coordtransform) - 提供了百度坐标（BD09）、国测局坐标（火星坐标，GCJ02）、和WGS84坐标系之间的转换

* [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus) - 大规模中文自然语言处理语料

* [imgaug](https://github.com/aleju/imgaug) **(not actively updated)** - Image augmentation for machine learning experiments.

* [accimage](https://github.com/pytorch/accimage) **(not actively updated)** - High performance image loading and augmenting routines mimicking PIL.Image interface.

* [Snorkel](https://github.com/snorkel-team/snorkel) **(not actively updated)** - A system for quickly generating training data with weak supervision.

* [fancyimpute](https://github.com/iskandr/fancyimpute) **(not actively updated)** - A variety of matrix completion and imputation algorithms implemented in Python.

* [Requests-HTML](https://github.com/psf/requests-html) **(not actively updated)** - Pythonic HTML Parsing for Humans.

* [lazynlp](https://github.com/chiphuyen/lazynlp) **(not actively updated)** - Library to scrape and clean web pages to create massive datasets.

* [Google Images Download](https://github.com/hardikvasa/google-images-download) **(not actively updated)** - Python Script to download hundreds of images from 'Google Images'.

### Data Similarity

* [image-match](https://github.com/ProvenanceLabs/image-match) - a simple package for finding approximate image matches from a corpus.

* [jellyfish](https://github.com/jamesturk/jellyfish) - A library for approximate & phonetic matching of strings.

* [TextDistance](https://github.com/life4/textdistance) - Python library for comparing distance between two or more sequences by many algorithms.

* [Qdrant](https://github.com/qdrant/qdrant) - A vector similarity search engine for text, image and categorical data in Rust.

### Data Management

* [pandera](https://github.com/unionai-oss/pandera) - A light-weight, flexible, and expressive statistical data testing library.

* [Kedro](https://github.com/kedro-org/kedro) - A Python framework for creating reproducible, maintainable and modular data science code.

* [PyFunctional](https://github.com/EntilZha/PyFunctional) - Python library for creating data pipelines with chain functional programming.

* [ImageHash](https://github.com/JohannesBuchner/imagehash) - An image hashing library written in Python.

* [pandas-profiling](https://github.com/ydataai/pandas-profiling) - Create HTML data profiling reports for pandas DataFrame.

* [FiftyOne](https://github.com/voxel51/fiftyone) - An open-source tool for building high-quality datasets and computer vision models.

* [Datasette](https://github.com/simonw/datasette) - An open source multi-tool for exploring and publishing data.

* [glom](https://github.com/mahmoud/glom) - Python's nested data operator (and CLI), for all your declarative restructuring needs.

* [dedupe](https://github.com/dedupeio/dedupe) - A python library that uses machine learning to perform fuzzy matching, deduplication and entity resolution quickly on structured data.

* [Ciphey](https://github.com/Ciphey/Ciphey) - Automatically decrypt encryptions without knowing the key or cipher, decode encodings, and crack hashes.

* [datasketch](https://github.com/ekzhu/datasketch) - Gives you probabilistic data structures that can process and search very large amount of data super fast, with little loss of accuracy.

## Data Visualization

* [Matplotlib](https://github.com/matplotlib/matplotlib) - A comprehensive library for creating static, animated, and interactive visualizations in Python.

* [Seaborn](https://github.com/mwaskom/seaborn) - A high-level interface for drawing statistical graphics, based on Matplotlib.

* [Bokeh](https://github.com/bokeh/bokeh) - Interactive Data Visualization in the browser, from Python.

* [Plotly.js](https://github.com/plotly/plotly.js) - Open-source JavaScript charting library behind Plotly and Dash.

* [Plotly.py](https://github.com/plotly/plotly.py) - An interactive, open-source, and browser-based graphing library for Python, based on Plotly.js.

* [ggplot2](https://github.com/tidyverse/ggplot2) - An implementation of the Grammar of Graphics in R.

* [ggpy](https://github.com/yhat/ggpy) - ggplot port for python.

* [Datapane](https://github.com/datapane/datapane) - An open-source framework to create data science reports in Python.

* [Visdom](https://github.com/fossasia/visdom) - A flexible tool for creating, organizing, and sharing visualizations of live, rich data. Supports Torch and Numpy.

* [TabPy](https://github.com/tableau/TabPy) - Execute Python code on the fly and display results in Tableau visualizations.

* [Streamlit](https://github.com/streamlit/streamlit) - The fastest way to build data apps in Python.

* [HyperTools](https://github.com/ContextLab/hypertools) - A Python toolbox for gaining geometric insights into high-dimensional data, based on Matplotlib and Seaborn.

* [Dash](https://github.com/plotly/dash) - Analytical Web Apps for Python, R, Julia and Jupyter, based on Plotly.js.

* [mpld3](https://github.com/mpld3/mpld3) - An interactive Matplotlib visualization tool in browser, based on D3.

* [Vega](https://github.com/vega/vega) - A visualization grammar, a declarative format for creating, saving, and sharing interactive visualization designs.

* [Vega-Lite](https://github.com/vega/vega-lite) - Provides a higher-level grammar for visual analysis that generates complete Vega specifications.

* [Vega-Altair](https://github.com/altair-viz/altair) - A declarative statistical visualization library for Python, based on Vega-Lite.

* [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) - Fast data visualization and GUI tools for scientific / engineering applications.

* [VisPy](https://github.com/vispy/vispy) - A high-performance interactive 2D/3D data visualization library, with OpenGL support.

* [PyVista](https://github.com/pyvista/pyvista) - 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK).

* [Potree](https://github.com/potree/potree) - WebGL point cloud viewer for large datasets.

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

* [Netron](https://github.com/lutzroeder/netron) - Visualizer for neural network, deep learning, and machine learning models.

* [livelossplot](https://github.com/stared/livelossplot) - Live training loss plot in Jupyter Notebook for Keras, PyTorch and others.

* [Diagrams](https://github.com/mingrammer/diagrams) - Lets you draw the cloud system architecture in Python code.

* [SandDance](https://github.com/microsoft/SandDance) - Visually explore, understand, and present your data.

* [ML Visuals](https://github.com/dair-ai/ml-visuals) - Contains figures and templates which you can reuse and customize to improve your scientific writing.

* [Scattertext](https://github.com/JasonKessler/scattertext) **(not actively updated)** - A tool for finding distinguishing terms in corpora and displaying them in an interactive HTML scatter plot.

* [TensorSpace.js](https://github.com/tensorspace-team/tensorspace) - Neural network 3D visualization framework, build interactive and intuitive model in browsers, support pre-trained deep learning models from TensorFlow, Keras, TensorFlow.js.

* [Netscope](https://github.com/ethereon/netscope) **(not actively updated)** - Neural network visualizer.

* [draw_convnet](https://github.com/gwding/draw_convnet) **(not actively updated)** - Python script for illustrating Convolutional Neural Network (ConvNet).

* [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) **(not actively updated)** - Latex code for making neural networks diagrams.

## Machine Learning Tutorials

* [PyTorch official tutorials](https://pytorch.org/tutorials/) - Official tutorials for PyTorch.

* [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) - State-of-the-Art Deep Learning examples that are easy to train and deploy, achieving the best reproducible accuracy and performance with NVIDIA CUDA-X software stack running on NVIDIA Volta, Turing and Ampere GPUs.

* [Learn OpenCV](https://github.com/spmallick/learnopencv) - C++ and Python Examples.

* [nlp-with-transformers](https://github.com/nlp-with-transformers/notebooks) - Jupyter notebooks for the Natural Language Processing with Transformers book.

* [labml.ai](https://nn.labml.ai/) - A collection of PyTorch implementations of neural networks and related algorithms, which are documented with explanations and rendered as side-by-side formatted notes.

* [Machine Learning Notebooks](https://github.com/ageron/handson-ml) **(no longer maintained)** - This project aims at teaching you the fundamentals of Machine Learning in python. It contains the example code and solutions to the exercises in my O'Reilly book Hands-on Machine Learning with Scikit-Learn and TensorFlow.

* [Machine Learning Notebooks, 3rd edition](https://github.com/ageron/handson-ml3) **(successor of Machine Learning Notebooks)** - A series of Jupyter notebooks that walk you through the fundamentals of Machine Learning and Deep Learning in Python using Scikit-Learn, Keras and TensorFlow 2.

* [Made With ML](https://github.com/GokuMohandas/Made-With-ML) - Learn how to responsibly develop, deploy and maintain production machine learning applications.

* [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow) - Simple Reinforcement learning tutorials.

* [Jezzamonn/fourier](https://github.com/Jezzamonn/fourier) - An Interactive Introduction to Fourier Transforms.

* [adv-financial-ml-marcos-exercises](https://github.com/fernandodelacalle/adv-financial-ml-marcos-exercises) - Exercises of the book: Advances in Financial Machine Learning by Marcos Lopez de Prado.

* [d2l-zh](https://github.com/d2l-ai/d2l-zh) - 《动手学深度学习》：面向中文读者、能运行、可讨论。中英文版被60个国家的400所大学用于教学。

* [nndl.github.io](https://github.com/nndl/nndl.github.io) - 《神经网络与深度学习》 邱锡鹏著

* [AI-Job-Notes](https://github.com/amusi/AI-Job-Notes) - AI算法岗求职攻略（涵盖准备攻略、刷题指南、内推和AI公司清单等资料）

* [TensorFlow Course](https://github.com/instillai/TensorFlow-Course) **(not actively updated)** - Simple and ready-to-use tutorials for TensorFlow.

* [Tensorflow Cookbook](https://github.com/taki0112/Tensorflow-Cookbook) **(not actively updated)** - Simple Tensorflow Cookbook for easy-to-use.

* [Tensorflow2 Cookbook](https://github.com/taki0112/Tensorflow2-Cookbook) **(not actively updated)** - Simple Tensorflow 2.x Cookbook for easy-to-use

* [TensorFlow Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials) - TensorFlow Tutorials with YouTube Videos.

* [stanford-cs-221-artificial-intelligence](https://github.com/afshinea/stanford-cs-221-artificial-intelligence) **(not actively updated)** - VIP cheatsheets for Stanford's CS 221 Artificial Intelligence.

* [TinyFlow](https://github.com/tqchen/tinyflow) **(no longer maintained)** - Tutorial code on how to build your own Deep Learning System in 2k Lines.

* [Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic) **(not actively updated)** - A technical report on convolution arithmetic in the context of deep learning.

* [tensorflow2_tutorials_chinese](https://github.com/czy36mengfei/tensorflow2_tutorials_chinese) **(not actively updated)** - tensorflow2中文教程

* [yao62995/tensorflow](https://github.com/yao62995/tensorflow) **(not actively updated)** - 图解tensorflow 源码

* [deeplearningbook-chinese](https://github.com/exacity/deeplearningbook-chinese) - Deep Learning 中文翻译

* [lihang-code](https://github.com/fengdu78/lihang-code) **(not actively updated)** - 《统计学习方法》的代码实现

# Computer Graphics

## Graphic Libraries & Renderers

* [NVIDIA Linux Open GPU Kernel Module Source](https://github.com/NVIDIA/open-gpu-kernel-modules) - NVIDIA Linux open GPU kernel module source.

* [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp) - Open-Source Vulkan C++ API.
  * <details open><summary>Related projects:</summary>

    * [Vulkan Guide](https://github.com/KhronosGroup/Vulkan-Guide) - One stop shop for getting started with the Vulkan API.
    * [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) - One stop solution for all Vulkan samples.
    * [VulkanTools](https://github.com/LunarG/VulkanTools) - Tools to aid in Vulkan development.
    * [VulkanTutorial](https://github.com/Overv/VulkanTutorial) - Tutorial for the Vulkan graphics and compute API.
    * [Vulkan C++ examples and demos](https://github.com/SaschaWillems/Vulkan) - Examples and demos for the new Vulkan API.
    </details>

* [GLFW](https://github.com/glfw/glfw) - A multi-platform library for OpenGL, OpenGL ES, Vulkan, window and input.

* [GLEW](https://github.com/nigels-com/glew) - The OpenGL Extension Wrangler Library.

* [WebGL](https://github.com/KhronosGroup/WebGL) - The Official Khronos WebGL Repository.

* [three.js](https://github.com/mrdoob/three.js) - JavaScript 3D Library.

* [CUB](https://github.com/NVIDIA/cub) - Cooperative primitives for CUDA C++.

* [glad](https://github.com/Dav1dde/glad) - Multi-Language Vulkan/GL/GLES/EGL/GLX/WGL Loader-Generator based on the official specs.

* [Shaderc](https://github.com/google/shaderc) - A collection of tools, libraries, and tests for Vulkan shader compilation.

* [3D Game Shaders For Beginners](https://github.com/lettier/3d-game-shaders-for-beginners) - A step-by-step guide to implementing SSAO, depth of field, lighting, normal mapping, and more for your 3D game.

* [Taichi Lang](https://github.com/taichi-dev/taichi) - Productive & portable high-performance programming in Python.

* [Mitsuba 2](https://github.com/mitsuba-renderer/mitsuba2) **(no longer maintained)** - A Retargetable Forward and Inverse Renderer.

* [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) **(successor of Mitsuba 2)** - A Retargetable Forward and Inverse Renderer.

* [OpenVR](https://github.com/ValveSoftware/openvr) - An API and runtime that allows access to VR hardware from multiple vendors without requiring that applications have specific knowledge of the hardware they are targeting.

* [A-Frame](https://github.com/aframevr/aframe) - Web framework for building virtual reality experiences.

* [Skia](https://github.com/google/skia) - A complete 2D graphic library for drawing Text, Geometries, and Images.

* [tiny-renderer](https://github.com/rougier/tiny-renderer) **(not actively updated)** - A tiny sotfware 3D renderer in 100 lines of Python.

## Game Engines

* [Godot](https://github.com/godotengine/godot) - Multi-platform 2D and 3D game engine.
  * <details open><summary>Related projects:</summary>

    * [Awesome Godot](https://github.com/godotengine/awesome-godot) - A curated list of free/libre plugins, scripts and add-ons for Godot
    * [Godot demo projects](https://github.com/godotengine/godot-demo-projects) - Demonstration and Template Projects.
    </details>

* [Stride](https://github.com/stride3d/stride) - An open-source C# game engine for realistic rendering and VR.

* [libGDX](https://github.com/libgdx/libgdx) - Desktop/Android/HTML5/iOS Java game development framework.

* [raylib](https://github.com/raysan5/raylib) - A simple and easy-to-use library to enjoy videogames programming.

* [O3DE](https://github.com/o3de/o3de) - An Apache 2.0-licensed multi-platform 3D engine that enables developers and content creators to build AAA games, cinema-quality 3D worlds, and high-fidelity simulations without any fees or commercial obligations.

* [EnTT](https://github.com/skypjack/entt) - Gaming meets modern C++ - a fast and reliable entity component system (ECS) and much more.

* [Halley](https://github.com/amzeratul/halley) - A lightweight game engine written in modern C++.

* [Panda3D](https://github.com/panda3d/panda3d) - Powerful, mature open-source cross-platform game engine for Python and C++, developed by Disney and CMU.

* [OpenXRay](https://github.com/OpenXRay/xray-16) - Improved version of the X-Ray Engine, the game engine used in the world-famous S.T.A.L.K.E.R. game series by GSC Game World.

* [Spring](https://github.com/spring/spring) - A powerful free cross-platform RTS game engine.

* [olcPixelGameEngine](https://github.com/OneLoneCoder/olcPixelGameEngine) - A tool used in [javidx9](https://github.com/OneLoneCoder/Javidx9)'s YouTube videos and projects.

* [Acid](https://github.com/EQMG/Acid) - A high speed C++17 Vulkan game engine.

* [Crown](https://github.com/crownengine/crown) - The flexible game engine.

* [Corange](https://github.com/orangeduck/Corange) - Pure C Game Engine.

* [KlayGE](https://github.com/gongminmin/KlayGE) - A cross-platform open source game engine with plugin-based architecture.

* [nCine](https://github.com/nCine/nCine) - A cross-platform 2D game engine.

* [SuperTuxKart](https://github.com/supertuxkart/stk-code) - SuperTuxKart is a free kart racing game. It focuses on fun and not on realistic kart physics.

* [Endless Sky](https://github.com/endless-sky/endless-sky) - Space exploration, trading, and combat game.

* [SDLPAL](https://github.com/sdlpal/sdlpal) - SDL-based reimplementation of the classic Chinese-language RPG known as PAL.

* [Game-Programmer-Study-Notes](https://github.com/QianMo/Game-Programmer-Study-Notes) - 涉及游戏开发中的图形学、实时渲染、编程实践、GPU编程、设计模式、软件工程等内容。

* [Cocos2d-x](https://github.com/cocos2d/cocos2d-x) **(not actively updated)** - A suite of open-source, cross-platform, game-development tools used by millions of developers all over the world.

* [WebGL Quake 3 Renderer](https://github.com/toji/webgl-quake3) **(not actively updated)** - WebGL app that renders levels from Quake 3.

* [DOOM-3-BFG](https://github.com/id-Software/DOOM-3-BFG) **(not actively updated)** - Doom 3 BFG Edition

* [toy](https://github.com/hugoam/toy) **(not actively updated)** - The thin c++ game engine.

* [GamePlay](https://github.com/gameplay3d/GamePlay) **(not actively updated)** - Open-source, cross-platform, C++ game engine for creating 2D/3D games.

* [Battle City Remake](https://github.com/shinima/battle-city) **(no longer maintained)** - Battle city remake built with react.

## CG Tutorials

* [tinyrenderer](https://github.com/ssloy/tinyrenderer) - Software rendering in 500 lines of code.

* [tinyraytracer](https://github.com/ssloy/tinyraytracer) - Understandable RayTracing in 256 lines of bare C++.

* [Unity3DTraining](https://github.com/XINCGer/Unity3DTraining) - Unity的练习项目

* [tinyraycaster](https://github.com/ssloy/tinyraycaster) - Build your own 3D shooter in a weekend.

* [tinykaboom](https://github.com/ssloy/tinykaboom) **(not actively updated)** - KABOOM! in 180 lines of bare C++.

* [Godot-24-Hours](https://github.com/vnen/Godot-24-Hours) **(not actively updated)** - Examples and demo projects for the Godot Engine Game Development in 24 Hours book.

# Full-Stack Development

## DevOps

* [Docker Compose](https://github.com/docker/compose) - Define and run multi-container applications with Docker.
  * <details open><summary>Related projects:</summary>

    * [Docker SDK for Python](https://github.com/docker/docker-py) - A Python library for the Docker Engine API
    * [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) - Build and run Docker containers leveraging NVIDIA GPUs
  </details>

* [Kubernetes Python Client](https://github.com/kubernetes-client/python) - Official Python client library for kubernetes.

* [Apache Airflow](https://github.com/apache/airflow) - A platform to programmatically author, schedule, and monitor workflows.

* [Gaia](https://github.com/gaia-pipeline/gaia) - Build powerful pipelines in any programming language.

* [ZooKeeper](https://github.com/apache/zookeeper) - Apache ZooKeeper.

* [Apollo](https://github.com/apolloconfig/apollo) - A reliable configuration management system suitable for microservice configuration management scenarios.

* [Nomad](https://github.com/hashicorp/nomad) - An easy-to-use, flexible, and performant workload orchestrator that can deploy a mix of microservice, batch, containerized, and non-containerized applications.

* [Flask](https://github.com/pallets/flask) - The Python micro framework for building web applications.

* [Buildbot](https://github.com/buildbot/buildbot) - Python-based continuous integration testing framework.

* [Kratos](https://github.com/go-kratos/kratos) - Your ultimate Go microservices framework for the cloud-native era.

* [Celery](https://github.com/celery/celery) - Distributed Task Queue.

* [Prefect 2](https://github.com/PrefectHQ/prefect) - The easiest way to transform any function into a unit of work that can be observed and governed by orchestration rules.

* [Luigi](https://github.com/spotify/luigi) - A Python module that helps you build complex pipelines of batch jobs.

* [RQ](https://github.com/rq/rq) - A simple Python library for queueing jobs and processing them in the background with workers.

* [huey](https://github.com/coleifer/huey) - A little task queue for python.

* [arq](https://github.com/samuelcolvin/arq) - Fast job queuing and RPC in python with asyncio and redis.

* [TaskTiger](https://github.com/closeio/tasktiger) - Python task queue using Redis.

* [Mara Pipelines](https://github.com/mara/mara-pipelines) - A lightweight opinionated ETL framework, halfway between plain scripts and Apache Airflow.

* [Ansible](https://github.com/ansible/ansible) - A radically simple IT automation platform that makes your applications and systems easier to deploy and maintain.

* [Pulumi](https://github.com/pulumi/pulumi) - Infrastructure as Code SDK is the easiest way to create and deploy cloud software that use containers, serverless functions, hosted services, and infrastructure, on any cloud.

* [Fabric](https://github.com/fabric/fabric) - Simple, Pythonic remote execution and deployment.

* [pyinfra](https://github.com/Fizzadar/pyinfra) - Automates infrastructure super fast at massive scale. It can be used for ad-hoc command execution, service deployment, configuration management and more.

* [Nightingale](https://github.com/ccfos/nightingale) - An enterprise-level cloud-native monitoring system, which can be used as drop-in replacement of Prometheus for alerting and Grafana for visualization.

* [Linux kernel](https://github.com/torvalds/linux) - Linux kernel source tree.

* [OSv](https://github.com/cloudius-systems/osv) - A new operating system for the cloud.

* [Netdata](https://github.com/netdata/netdata) - Real-time performance monitoring, done right!

* [whylogs](https://github.com/whylabs/whylogs) - The open standard for data logging.

* [devops-exercises](https://github.com/bregman-arie/devops-exercises) - Linux, Jenkins, AWS, SRE, Prometheus, Docker, Python, Ansible, Git, Kubernetes, Terraform, OpenStack, SQL, NoSQL, Azure, GCP, DNS, Elastic, Network, Virtualization. DevOps Interview Questions.

* [TencentOS-tiny](https://github.com/OpenAtomFoundation/TencentOS-tiny) - 腾讯物联网终端操作系统

* [Codespaces](https://github.com/codespaces-io/codespaces) **(not actively updated)** - Devops Workspaces in a Box.

## Desktop App Development

* [Electron](https://github.com/electron/electron) - Build cross-platform desktop apps with JavaScript, HTML, and CSS.
  * <details open><summary>Related projects:</summary>

    * [electron-quick-start](https://github.com/electron/electron-quick-start) - Clone to try a simple Electron app.
    * [Electron API Demos](https://github.com/electron/electron-api-demos) - Explore the Electron APIs.
    </details>

* [TypeScript](https://github.com/microsoft/TypeScript) - A superset of JavaScript that compiles to clean JavaScript output.

* [React Native](https://github.com/facebook/react-native) - A framework for building native applications using React.

* [Appsmith](https://github.com/appsmithorg/appsmith) - Low code project to build admin panels, internal tools, and dashboards. Integrates with 15+ databases and any API.

* [SCons](https://github.com/SCons/scons) - A software construction tool.

* [Bazel](https://github.com/bazelbuild/bazel) - A fast, scalable, multi-language and extensible build system.

* [xmake](https://github.com/xmake-io/xmake) - A cross-platform build utility based on Lua.

* [Proton Native](https://github.com/kusti8/proton-native) **(not actively updated)** - A React environment for cross platform desktop apps.

### Python Toolkit

* [Kivy](https://github.com/kivy/kivy) - Open source UI framework written in Python, running on Windows, Linux, macOS, Android and iOS.

* [Gooey](https://github.com/chriskiehl/Gooey) - Turn (almost) any Python command line program into a full GUI application with one line.

* [DearPyGui](https://github.com/hoffstadt/DearPyGui) - A fast and powerful Graphical User Interface Toolkit for Python with minimal dependencies.

* [Flexx](https://github.com/flexxui/flexx) - Write desktop and web apps in pure Python.

* [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI) - Transforms the tkinter, Qt, WxPython, and Remi (browser-based) GUI frameworks into a simpler interface.

* [Eel](https://github.com/python-eel/Eel) - A little Python library for making simple Electron-like HTML/JS GUI apps.

* [Toga](https://github.com/beeware/toga) - A Python native, OS native GUI toolkit.

* [schedule](https://github.com/dbader/schedule) - Python job scheduling for humans.

* [Click](https://github.com/pallets/click) - A Python package for creating beautiful command line interfaces in a composable way with as little code as necessary.

* [Rich](https://github.com/Textualize/rich) - A Python library for rich text and beautiful formatting in the terminal.

* [Colorama](https://github.com/tartley/colorama) - Simple cross-platform colored terminal text in Python.

* [colout](https://github.com/nojhan/colout) - Color text streams with a polished command line interface.

* [ASCIIMATICS](https://github.com/peterbrittain/asciimatics) - A cross platform package to do curses-like operations, plus higher level APIs and widgets to create text UIs and ASCII art animations.

* [Emoji](https://github.com/carpedm20/emoji) - emoji terminal output for Python.

* [Python Fire](https://github.com/google/python-fire) - A library for automatically generating command line interfaces (CLIs) from absolutely any Python object.

* [Typer](https://github.com/tiangolo/typer) - A Python library for building CLI applications.

* [powerline-shell](https://github.com/b-ryan/powerline-shell) - A beautiful and useful prompt for your shell.

* [Python Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) - Library for building powerful interactive command line applications in Python.

* [Questionary](https://github.com/tmbo/questionary) - A Python library for effortlessly building pretty command line interfaces.

* [Argcomplete](https://github.com/kislyuk/argcomplete) - Provides easy, extensible command line tab completion of arguments for your Python script.

* [python-dotenv](https://github.com/theskumar/python-dotenv) - Reads key-value pairs from a .env file and can set them as environment variables.

* [Cookiecutter](https://github.com/cookiecutter/cookiecutter) - A cross-platform command-line utility that creates projects from cookiecutters (project templates), e.g. Python package projects, C projects.

* [PyScaffold](https://github.com/pyscaffold/pyscaffold) - A project generator for bootstrapping high quality Python packages, ready to be shared on PyPI and installable via pip.

* [dynaconf](https://github.com/dynaconf/dynaconf) - Configuration Management for Python.

* [Hydra](https://github.com/facebookresearch/hydra) - A framework for elegantly configuring complex applications.

* [Python Decouple](https://github.com/henriquebastos/python-decouple) - Helps you to organize your settings so that you can change parameters without having to redeploy your app.

* [OmegaConf](https://github.com/omry/omegaconf) - A hierarchical configuration system, with support for merging configurations from multiple sources (YAML config files, dataclasses/objects and CLI arguments) providing a consistent API regardless of how the configuration was created.

* [Gin Config](https://github.com/google/gin-config) - Provides a lightweight configuration framework for Python.

* [Py4J](https://github.com/py4j/py4j) - Enables Python programs to dynamically access arbitrary Java objects.

* [keyboard](https://github.com/boppreh/keyboard) - Hook and simulate global keyboard events on Windows and Linux.

### C++/C Toolkit

* [wxWidgets](https://github.com/wxWidgets/wxWidgets) - Cross-Platform C++ GUI Library.

* [Nana](https://github.com/cnjinhao/nana) **(not actively updated)** - A modern C++ GUI library.

## Web Development

* [React](https://github.com/facebook/react) - A declarative, efficient, and flexible JavaScript library for building user interfaces.

* [Django](https://github.com/django/django) - A high-level Python web framework that encourages rapid development and clean, pragmatic design.

* [jQuery](https://github.com/jquery/jquery) - jQuery JavaScript Library.

* [jQuery UI](https://github.com/jquery/jquery-ui) - The official jQuery user interface library.

* [Ant Design](https://github.com/ant-design/ant-design) - An enterprise-class UI design language and React UI library.

* [Hugo](https://github.com/gohugoio/hugo) - The world’s fastest framework for building websites.

* [Hexo](https://github.com/hexojs/hexo) - A fast, simple & powerful blog framework, powered by Node.js.

* [Jekyll](https://github.com/jekyll/jekyll) - A blog-aware static site generator in Ruby.

* [Gutenberg](https://github.com/WordPress/gutenberg) - The Block Editor project for WordPress and beyond.

* [Wasmer](https://github.com/wasmerio/wasmer) - The leading WebAssembly Runtime supporting WASI and Emscripten.

* [Ghost](https://github.com/TryGhost/Ghost) - Turn your audience into a business. Publishing, memberships, subscriptions and newsletters.

* [Mercury](https://github.com/mljar/mercury) - Convert Python notebook to web app and share with non-technical users.

* [Stylus](https://github.com/stylus/stylus) - Expressive, robust, feature-rich CSS language built for nodejs.

* [D3](https://github.com/d3/d3) - A JavaScript library for visualizing data using web standards.

* [Paramiko](https://github.com/paramiko/paramiko) - The leading native Python SSHv2 protocol library.

* [Netmiko](https://github.com/ktbyers/netmiko) - Multi-vendor library to simplify Paramiko SSH connections to network devices.

* [Storybook](https://github.com/storybookjs/storybook) - A frontend workshop for building UI components and pages in isolation. Made for UI development, testing, and documentation.

* [ProjectVisBug](https://github.com/GoogleChromeLabs/ProjectVisBug) - FireBug for designers › Edit any webpage, in any state.

* [readthedocs.org](https://github.com/readthedocs/readthedocs.org) - The source code that powers readthedocs.org

* [reactnative.dev](https://github.com/facebook/react-native-website) - Configuration and documentation powering the React Native website.

* [Clone Wars](https://github.com/GorvGoyl/Clone-Wars) - 100+ open-source clones of popular sites like Airbnb, Amazon, Instagram, Netflix, Tiktok, Spotify, Whatsapp, Youtube etc. See source code, demo links, tech stack, github stars.

* [50projects50days](https://github.com/bradtraversy/50projects50days) - 50+ mini web projects using HTML, CSS & JS.

* [Public APIs](https://github.com/public-apis/public-apis) - A collective list of free APIs

* [WebKit](https://github.com/WebKit/WebKit) - The browser engine used by Safari, Mail, App Store and many other applications on macOS, iOS and Linux.

* [PhantomJS](https://github.com/ariya/phantomjs) - Scriptable Headless Browser.

* [Open-IM-Server](https://github.com/OpenIMSDK/Open-IM-Server) - Open source Instant Messaging Server.

* [progress-bar](https://github.com/fredericojordan/progress-bar) - Flask API for SVG progress badges.

* [ScrollMagic](https://github.com/janpaepke/ScrollMagic) - The javascript library for magical scroll interactions.

* [KaTeX](https://github.com/KaTeX/KaTeX) - Fast math typesetting for the web.

* [Brook](https://github.com/txthinking/brook) - A cross-platform network tool designed for developers.

* [pixelmatch](https://github.com/mapbox/pixelmatch) - The smallest, simplest and fastest JavaScript pixel-level image comparison library.

* [kcptun](https://github.com/xtaci/kcptun) - A Stable & Secure Tunnel based on KCP with N:M multiplexing and FEC. Available for ARM, MIPS, 386 and AMD64

* [mall-swarm](https://github.com/macrozheng/mall-swarm) - 是一套微服务商城系统，采用了 Spring Cloud 2021 & Alibaba、Spring Boot 2.7、Oauth2、MyBatis、Docker、Elasticsearch、Kubernetes等核心技术，同时提供了基于Vue的管理后台方便快速搭建系统。mall-swarm在电商业务的基础集成了注册中心、配置中心、监控中心、网关等系统功能。文档齐全，附带全套Spring Cloud教程。

* [bbs-go](https://github.com/mlogclub/bbs-go) - 基于Golang的开源社区系统。

* [py12306](https://github.com/pjialin/py12306) - 12306 购票助手，支持集群，多账号，多任务购票以及 Web 页面管理

* [heti](https://github.com/sivan/heti) - 赫蹏（hètí）是专为中文内容展示设计的排版样式增强。它基于通行的中文排版规范而来，可以为网站的读者带来更好的文章阅读体验。

* [spring-boot-examples](https://github.com/ityouknow/spring-boot-examples) - Spring Boot 教程、技术栈示例代码，快速简单上手教程。

* [SpringBoot-Learning](https://github.com/dyc87112/SpringBoot-Learning) - Spring Boot基础教程。

* [big-react](https://github.com/BetaSu/big-react) - 从零实现 React v18 的核心功能。

* [visual-drag-demo](https://github.com/woai3c/visual-drag-demo) - 一个低代码（可视化拖拽）教学项目。

* [Waypoints](https://github.com/imakewebthings/waypoints) - A library that makes it easy to execute a function whenever you scroll to an element.

* [flv.js](https://github.com/bilibili/flv.js) **(not actively updated)** - HTML5 FLV Player

* [cim](https://github.com/crossoverJie/cim) **(not actively updated)** - 适用于开发者的分布式即时通讯系统

## Mobile Development

* [Ionic](https://github.com/ionic-team/ionic-framework) - A powerful cross-platform UI toolkit for building native-quality iOS, Android, and Progressive Web Apps with HTML, CSS, and JavaScript.

* [PulltoRefresh.js](https://github.com/BoxFactura/pulltorefresh.js) - A quick and powerful plugin for your pull-to-refresh needs in your webapp.

* [Signal Android](https://github.com/signalapp/Signal-Android) - A private messenger for Android.

* [QMUI_Android](https://github.com/Tencent/QMUI_Android) - 提高 Android UI 开发效率的 UI 库

* [GSYVideoPlayer](https://github.com/CarGuo/GSYVideoPlayer) - 视频播放器（IJKplayer、ExoPlayer、MediaPlayer），HTTPS，支持弹幕，外挂字幕，支持滤镜、水印、gif截图，片头广告、中间广告，多个同时播放，支持基本的拖动，声音、亮度调节，支持边播边缓存，支持视频自带rotation的旋转（90,270之类），重力旋转与手动旋转的同步支持，支持列表播放 ，列表全屏动画，视频加载速度，列表小窗口支持拖动，动画效果，调整比例，多分辨率切换，支持切换播放器，进度条小窗口预览，列表切换详情页面无缝播放，rtsp、concat、mpeg。

* [GSYGithubAppKotlin](https://github.com/CarGuo/GSYGithubAppKotlin) - 超完整的Android Kotlin 项目，功能丰富，适合学习和日常使用。GSYGithubApp系列的优势：目前已经拥有Flutter、Weex、ReactNative、Kotlin四个版本。 功能齐全，项目框架内技术涉及面广，完成度高。

* [MethodTraceMan](https://github.com/zhengcx/MethodTraceMan) - 用于快速找到高耗时方法，定位解决Android App卡顿问题。通过gradle plugin+ASM实现可配置范围的方法插桩来统计所有方法的耗时，并提供友好的界面展示，支持耗时筛选、线程筛选、方法名筛选等。

* [EasyFloat](https://github.com/princekin-f/EasyFloat) - 浮窗从未如此简单（Android可拖拽悬浮窗口，支持页面过滤、自定义动画，可设置单页面浮窗、前台浮窗、全局浮窗，浮窗权限按需自动申请...）

* [Dexposed](https://github.com/alibaba/dexposed) - Dexposed enable 'god' mode for single android application.

* [Epic](https://github.com/tiann/epic) - Dynamic java method AOP hook for Android(continution of Dexposed on ART), Supporting 5.0~11.

* [GPUImage](https://github.com/BradLarson/GPUImage) **(not actively updated)** - An open source iOS framework for GPU-based image and video processing.

* [GPUImage for Android](https://github.com/cats-oss/android-gpuimage) **(not actively updated)** - Android filters based on OpenGL (idea from GPUImage for iOS).

* [ijkplayer](https://github.com/bilibili/ijkplayer) **(not actively updated)** - Android/iOS video player based on FFmpeg n3.4, with MediaCodec, VideoToolbox support.

* [libstreaming](https://github.com/fyhertz/libstreaming) **(not actively updated)** - A solution for streaming H.264, H.263, AMR, AAC using RTP on Android.

* [Stetho](https://github.com/facebook/stetho) **(not actively updated)** - A debug bridge for Android applications, enabling the powerful Chrome Developer Tools and much more.

* [Genius-Android](https://github.com/qiujuer/Genius-Android) - Android Material Design Theme UI and Tool Library.

* [MultiType](https://github.com/drakeet/MultiType) **(not actively updated)** - Flexible multiple types for Android RecyclerView.

* [DanmakuFlameMaster](https://github.com/bilibili/DanmakuFlameMaster) **(not actively updated)** - Android开源弹幕引擎

* [MagicCamera](https://github.com/wuhaoyu1990/MagicCamera) - 包含美颜等40余种实时滤镜相机，可拍照、录像、图片修改

* [LazyRecyclerAdapter](https://github.com/CarGuo/LazyRecyclerAdapter) **(not actively updated)** - 极简通用的RecyclerAdapter，入侵性低，支持一个列表多种Item类型，无需维护和编写Adapter代码，快速集成拥有点击，动画，自定义刷新，自定义加载更多，自定义空页面显示，通用分割线，动态绑定等高复用，你只需要编写维护Holder代码。

## Process, Thread & Coroutine

* [sh](https://github.com/amoffat/sh) - A full-fledged subprocess replacement for Python 2, Python 3, PyPy and PyPy3 that allows you to call any program as if it were a function.

* [oneTBB](https://github.com/oneapi-src/oneTBB) - A flexible C++ library that simplifies the work of adding parallelism to complex applications, even if you are not a threading expert.

* [HPX](https://github.com/STEllAR-GROUP/hpx) - The C++ Standard Library for Parallelism and Concurrency.

* [Muduo](https://github.com/chenshuo/muduo) - Event-driven network library for multi-threaded Linux server in C++11.

* [Supervisor](https://github.com/Supervisor/supervisor) - A client/server system that allows its users to control a number of processes on UNIX-like operating systems.

* [Pexpect](https://github.com/pexpect/pexpect) - A Python module for controlling interactive programs in a pseudo-terminal.

* [Plumbum](https://github.com/tomerfiliba/plumbum) - A small yet feature-rich library for shell script-like programs in Python.

* [Greenlets](https://github.com/python-greenlet/greenlet) - Lightweight in-process concurrent programming.

* [AnyIO](https://github.com/agronholm/anyio) - High level asynchronous concurrency and networking framework that works on top of either trio or asyncio.

* [gevent](https://github.com/gevent/gevent) - Coroutine-based concurrency library for Python.

* [CTPL](https://github.com/vit-vit/CTPL) **(not actively updated)** - Modern and efficient C++ Thread Pool Library.

* [ThreadPool](https://github.com/progschj/ThreadPool) **(not actively updated)** - A simple C++11 Thread Pool implementation.

## Debugging & Profiling & Tracing

### For Python

* [PySnooper](https://github.com/cool-RR/PySnooper) - Never use print for debugging again.

* [py-spy](https://github.com/benfred/py-spy) - A sampling profiler for Python programs.

* [Scalene](https://github.com/plasma-umass/scalene) - A high-performance, high-precision CPU, GPU, and memory profiler for Python.

* [Pyroscope](https://github.com/pyroscope-io/pyroscope) - Pyroscope is an open source continuous profiling platform.

* [pyinstrument](https://github.com/joerick/pyinstrument) - Call stack profiler for Python.

* [vprof](https://github.com/nvdv/vprof) - A Python package providing rich and interactive visualizations for various Python program characteristics such as running time and memory usage.

* [GPUtil](https://github.com/anderskm/gputil) - A Python module for getting the GPU status from NVIDA GPUs using nvidia-smi programmically in Python.

* [Wily](https://github.com/tonybaloney/wily) - A Python application for tracking, reporting on timing and complexity in Python code.

* [Radon](https://github.com/rubik/radon) - Various code metrics for Python code.

* [ps_mem](https://github.com/pixelb/ps_mem) - A utility to accurately report the in core memory usage for a program.

### For C++/C

* [x64dbg](https://github.com/x64dbg/x64dbg) - An open-source x64/x32 debugger for windows.

* [ORBIT](https://github.com/google/orbit) - A standalone C/C++ profiler for Windows and Linux.

* [BCC](https://github.com/iovisor/bcc) - Tools for BPF-based Linux IO analysis, networking, monitoring, and more.

* [osquery](https://github.com/osquery/osquery) - SQL powered operating system instrumentation, monitoring, and analytics.

* [Tracy](https://github.com/wolfpld/tracy) - A real time, nanosecond resolution, remote telemetry, hybrid frame and sampling profiler for games and other applications.

* [Coz](https://github.com/plasma-umass/coz) - Finding Code that Counts with Causal Profiling.

* [timemory](https://github.com/NERSC/timemory) - Modular C++ Toolkit for Performance Analysis and Logging. Profiling API and Tools for C, C++, CUDA, Fortran, and Python.

* [gputop](https://github.com/rib/gputop) **(not actively updated)** - A GPU profiling tool.

### For Go

* [gops](https://github.com/google/gops) - A tool to list and diagnose Go processes currently running on your system.

* [pprof](https://github.com/google/pprof) - A tool for visualization and analysis of profiling data.

* [JD-GUI](https://github.com/java-decompiler/jd-gui) **(not actively updated)** - A standalone Java Decompiler GUI.

## Data Management & Processing

### Database & Cloud Management

* [Redis](https://github.com/redis/redis) - An in-memory database that persists on disk.
  * <details open><summary>Related projects:</summary>

    * [redis-py](https://github.com/redis/redis-py) - Redis Python client
    * [Node-Redis](https://github.com/redis/node-redis) - Redis Node.js client
    * [Jedis](https://github.com/redis/jedis) - Redis Java client
  </details>

* [MongoDB](https://github.com/mongodb/mongo) - The MongoDB Database.
  * <details open><summary>Related projects:</summary>

    * [PyMongo](https://github.com/mongodb/mongo-python-driver) - The Python driver for MongoDB
    * [MongoDB Go Driver](https://github.com/mongodb/mongo-go-driver) - The Go driver for MongoDB
    * [MongoDB NodeJS Driver](https://github.com/mongodb/node-mongodb-native) - The Node.js driver for MongoDB
    * [MongoDB C# Driver](https://github.com/mongodb/mongo-csharp-driver) - The .NET driver for MongoDB
    * [MongoEngine](https://github.com/MongoEngine/mongoengine) - A Python Object-Document-Mapper for working with MongoDB
    * [Motor](https://github.com/mongodb/motor) - The async Python driver for MongoDB and Tornado or asyncio
  </details>

* [Apache Spark](https://github.com/apache/spark) - A unified analytics engine for large-scale data processing.

* [Presto](https://github.com/prestodb/presto) - A distributed SQL query engine for big data.

* [Google Cloud Python Client](https://github.com/googleapis/google-cloud-python) - Google Cloud Client Library for Python.

* [Elasticsearch](https://github.com/elastic/elasticsearch) - Free and Open, Distributed, RESTful Search Engine.
  * <details open><summary>Related projects:</summary>

    * [Kibana](https://github.com/elastic/kibana) - A browser-based analytics and search dashboard for Elasticsearch
    * [Logstash](https://github.com/elastic/logstash) - Transport and process your logs, events, or other data
    * [Beats](https://github.com/elastic/beats) - Lightweight shippers for Elasticsearch & Logstash
    * [Elastic UI Framework](https://github.com/elastic/eui) - A collection of React UI components for quickly building user interfaces at Elastic
    * [Elasticsearch Python Client](https://github.com/elastic/elasticsearch-py) - Official Elasticsearch client library for Python
    * [Elasticsearch DSL](https://github.com/elastic/elasticsearch-dsl-py) - High level Python client for Elasticsearch
    * [Elasticsearch Node.js client](https://github.com/elastic/elasticsearch-js) - Official Elasticsearch client library for Node.js
    * [Elasticsearch PHP client](https://github.com/elastic/elasticsearch-php) - Official PHP client for Elasticsearch
    * [go-elasticsearch](https://github.com/elastic/go-elasticsearch) - The official Go client for Elasticsearch
  </details>

* [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) - The Python SQL Toolkit and Object Relational Mapper.
  * <details open><summary>Related projects:</summary>

    * [Alembic](https://github.com/sqlalchemy/alembic) - A database migrations tool for SQLAlchemy
    * [SQLModel](https://github.com/tiangolo/sqlmodel) - SQL databases in Python, designed for simplicity, compatibility, and robustness
    * [Databases](https://github.com/encode/databases) - Async database support for Python
  </details>

* [Apache Libcloud](https://github.com/apache/libcloud) - A Python library which hides differences between different cloud provider APIs and allows you to manage different cloud resources through a unified and easy to use API.

* [Grafana](https://github.com/grafana/grafana) - The open and composable observability and data visualization platform. Visualize metrics, logs, and traces from multiple sources like Prometheus, Loki, Elasticsearch, InfluxDB, Postgres and many more.

* [Joblib Apache Spark Backend](https://github.com/joblib/joblib-spark) - Provides Apache Spark backend for joblib to distribute tasks on a Spark cluster.

* [PyMySQL](https://github.com/PyMySQL/PyMySQL) - Pure Python MySQL Client.
  * <details open><summary>Related projects:</summary>

    * [mysqlclient](https://github.com/PyMySQL/mysqlclient) - MySQL database connector for Python
  </details>

* [Redigo](https://github.com/gomodule/redigo) - Go client for Redis.

* [Dgraph](https://github.com/dgraph-io/dgraph) - Native GraphQL Database with graph backend.

* [Tortoise ORM](https://github.com/tortoise/tortoise-orm) - Familiar asyncio ORM for python, built with relations in mind.

* [Ibis](https://github.com/ibis-project/ibis) - Expressive analytics in Python at any scale.

* [peewee](https://github.com/coleifer/peewee) - A small, expressive orm -- supports postgresql, mysql and sqlite.

* [DB4S](https://github.com/sqlitebrowser/sqlitebrowser) - DB Browser for SQLite (DB4S) is a high quality, visual, open source tool to create, design, and edit database files compatible with SQLite.

* [TinyDB](https://github.com/msiemens/tinydb) - A lightweight document oriented database written in pure Python and has no external dependencies.

* [MyCAT](https://github.com/MyCATApache/Mycat-Server) - An enforced database which is a replacement for MySQL and supports transaction and ACID.

* [Pony](https://github.com/ponyorm/pony) - An advanced object-relational mapper.

* [dataset](https://github.com/pudo/dataset) - Easy-to-use data handling for SQL data stores with support for implicit table creation, bulk loading, and transactions.

* [Dagster](https://github.com/dagster-io/dagster) - An orchestration platform for the development, production, and observation of data assets.

* [Great Expectations](https://github.com/great-expectations/great_expectations) - Helps data teams eliminate pipeline debt, through data testing, documentation, and profiling.

* [dbt](https://github.com/dbt-labs/dbt-core) - Enables data analysts and engineers to transform their data using the same practices that software engineers use to build applications.

* [Metabase](https://github.com/metabase/metabase) - The simplest, fastest way to get business intelligence and analytics to everyone in your company.

* [Ploomber](https://github.com/ploomber/ploomber) - The fastest way to build data pipelines.

* [PyHive](https://github.com/dropbox/PyHive) - Python interface to Hive and Presto.

* [Pypeln](https://github.com/cgarciae/pypeln) - A simple yet powerful Python library for creating concurrent data pipelines.

* [petl](https://github.com/petl-developers/petl) - A general purpose Python package for extracting, transforming and loading tables of data.

* [PySyft](https://github.com/OpenMined/PySyft) - Data science on data without acquiring a copy.

### Streaming Data Management

* [Apache Beam](https://github.com/apache/beam) - A unified programming model for Batch and Streaming data processing.

* [Apache Kafka](https://github.com/apache/kafka) - Mirror of Apache Kafka.

* [Apache Flink](https://github.com/apache/flink) - An open source stream processing framework with powerful stream- and batch-processing capabilities.

* [kafka-python](https://github.com/dpkp/kafka-python) - Python client for Apache Kafka.

* [confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python) - Confluent's Kafka Python Client.

* [Perspective](https://github.com/finos/perspective) - A data visualization and analytics component, especially well-suited for large and/or streaming datasets.

* [Deep Lake](https://github.com/activeloopai/deeplake) - Data Lake for Deep Learning. Build, manage, query, version, & visualize datasets. Stream data real-time to PyTorch/TensorFlow.

* [Streamparse](https://github.com/Parsely/streamparse) - Lets you run Python code against real-time streams of data via Apache Storm.

* [StreamAlert](https://github.com/airbnb/streamalert) - A serverless, realtime data analysis framework which empowers you to ingest, analyze, and alert on data from any environment, using datasources and alerting logic you define.

* [Prometheus](https://github.com/prometheus/prometheus) - The Prometheus monitoring system and time series database.
  * <details open><summary>Related projects:</summary>

    * [Prometheus Python Client](https://github.com/prometheus/client_python) - Prometheus instrumentation library for Python applications
  </details>

## Data Format & I/O

* [protobuf](https://github.com/protocolbuffers/protobuf) - Google's language-neutral, platform-neutral, extensible mechanism for serializing structured data.

* [FlatBuffers](https://github.com/google/flatbuffers) - A cross platform serialization library architected for maximum memory efficiency.

### For Python

* [Imageio](https://github.com/imageio/imageio) - Python library for reading and writing image data.

* [Wand](https://github.com/emcconville/wand) - The ctypes-based simple ImageMagick binding for Python.

* [VidGear](https://github.com/abhiTronix/vidgear) - A High-performance cross-platform Video Processing Python framework powerpacked with unique trailblazing features.

* [marshmallow](https://github.com/marshmallow-code/marshmallow) - A lightweight library for converting complex objects to and from simple Python datatypes.

* [cloudpickle](https://github.com/cloudpipe/cloudpickle) - Extended pickling support for Python objects.

* [dill](https://github.com/uqfoundation/dill) - Extends python's pickle module for serializing and de-serializing python objects to the majority of the built-in python types.

* [UltraJSON](https://github.com/ultrajson/ultrajson) - Ultra fast JSON decoder and encoder written in C with Python bindings.

* [orjson](https://github.com/ijl/orjson) - Fast, correct Python JSON library supporting dataclasses, datetimes, and numpy

* [simplejson](https://github.com/simplejson/simplejson) - A simple, fast, extensible JSON encoder/decoder for Python.

* [jsonschema](https://github.com/python-jsonschema/jsonschema) - An implementation of the JSON Schema specification for Python.

* [jsonpickle](https://github.com/jsonpickle/jsonpickle) - Python library for serializing any arbitrary object graph into JSON.

* [MessagePack](https://github.com/msgpack/msgpack-python) - An efficient binary serialization format. It lets you exchange data among multiple languages like JSON.

* [PyYAML](https://github.com/yaml/pyyaml) - Canonical source repository for PyYAML.

* [StrictYAML](https://github.com/crdoconnor/strictyaml) - Type-safe YAML parser and validator.

* [xmltodict](https://github.com/martinblech/xmltodict) - Python module that makes working with XML feel like you are working with JSON.

* [csvkit](https://github.com/wireservice/csvkit) - A suite of utilities for converting to and working with CSV, the king of tabular file formats.

* [Tablib](https://github.com/jazzband/tablib) - Python Module for Tabular Datasets in XLS, CSV, JSON, YAML, &c.

* [HDF5 for Python](https://github.com/h5py/h5py) - The h5py package is a Pythonic interface to the HDF5 binary data format.

* [smart_open](https://github.com/RaRe-Technologies/smart_open) - Utils for streaming large files (S3, HDFS, gzip, bz2...).

* [validators](https://github.com/python-validators/validators) - Python Data Validation for Humans.

* [Arrow](https://github.com/arrow-py/arrow) - A Python library that offers a sensible and human-friendly approach to creating, manipulating, formatting and converting dates, times and timestamps.

* [Pendulum](https://github.com/sdispater/pendulum) - Python datetimes made easy.

* [dateutil](https://github.com/dateutil/dateutil) - The dateutil module provides powerful extensions to the standard datetime module, available in Python.

* [dateparser](https://github.com/scrapinghub/dateparser) - Python parser for human readable dates.

* [Watchdog](https://github.com/gorakhargosh/watchdog) - Python library and shell utilities to monitor filesystem events.

* [uvloop](https://github.com/MagicStack/uvloop) - A fast, drop-in replacement of the built-in asyncio event loop.

* [aiofiles](https://github.com/Tinche/aiofiles) - An Apache2 licensed library, written in Python, for handling local disk files in asyncio applications.

* [PyFilesystem2](https://github.com/PyFilesystem/pyfilesystem2) - Python's Filesystem abstraction layer.

* [path](https://github.com/jaraco/path) - Object-oriented file system path manipulation.

* [phonenumbers Python Library](https://github.com/daviddrysdale/python-phonenumbers) - Python port of Google's libphonenumber.

* [Chardet](https://github.com/chardet/chardet) - Python character encoding detector.

* [Python Slugify](https://github.com/un33k/python-slugify) - A Python slugify application that handles unicode.

* [humanize](https://github.com/python-humanize/humanize) - Contains various common humanization utilities, like turning a number into a fuzzy human-readable duration ("3 minutes ago") or into a human-readable size or throughput.

* [XlsxWriter](https://github.com/jmcnamara/XlsxWriter) - A Python module for creating Excel XLSX files.

* [xlwings](https://github.com/xlwings/xlwings) - A Python library that makes it easy to call Python from Excel and vice versa.

* [pygsheets](https://github.com/nithinmurali/pygsheets) - Google Spreadsheets Python API v4

* [gdown](https://github.com/wkentaro/gdown) - Download a large file from Google Drive.

* [schema](https://github.com/keleshev/schema) **(not actively updated)** - A library for validating Python data structures.

### For C++/C

* [glog](https://github.com/google/glog) - C++ implementation of the Google logging module.

* [FFmpeg](https://github.com/FFmpeg/FFmpeg) - A collection of libraries and tools to process multimedia content such as audio, video, subtitles and related metadata.

* [LAV Filters](https://github.com/Nevcairiel/LAVFilters) - Open-Source DirectShow Media Splitter and Decoders.

* [OpenEXR](https://github.com/AcademySoftwareFoundation/openexr) - Provides the specification and reference implementation of the EXR file format, the professional-grade image storage format of the motion picture industry.

* [spdlog](https://github.com/gabime/spdlog) - Fast C++ logging library.

* [glogg](https://github.com/nickbnf/glogg) **(not actively updated)** - A fast, advanced log explorer.

### For Go

* [json-iterator/go](https://github.com/json-iterator/go) - A high-performance 100% compatible drop-in replacement of "encoding/json"

* [json-to-go](https://github.com/mholt/json-to-go) - Translates JSON into a Go type in your browser instantly (original).

### For Java

* [fastjson](https://github.com/alibaba/fastjson) - A Java library that can be used to convert Java Objects into their JSON representation.

* [jackson-core](https://github.com/FasterXML/jackson-core) - Core part of Jackson that defines Streaming API as well as basic shared abstractions.

* [Okio](https://github.com/square/okio) - A modern I/O library for Android, Java, and Kotlin Multiplatform.

## Security

* [Vulhub](https://github.com/vulhub/vulhub) - Pre-Built Vulnerable Environments Based on Docker-Compose.

* [hackingtool](https://github.com/Z4nzu/hackingtool) - ALL IN ONE Hacking Tool For Hackers.

* [sqlmap](https://github.com/sqlmapproject/sqlmap) - Automatic SQL injection and database takeover tool.

* [detect-secrets](https://github.com/Yelp/detect-secrets) - An enterprise friendly way of detecting and preventing secrets in code.

* [Safety](https://github.com/pyupio/safety) - Safety checks Python dependencies for known security vulnerabilities and suggests the proper remediations for vulnerabilities detected.

* [Bandit](https://github.com/PyCQA/bandit) - A tool designed to find common security issues in Python code.

* [Mattermost](https://github.com/mattermost/mattermost-server) - An open source platform for secure collaboration across the entire software development lifecycle.

## Package Management

### For Python

* [Conda](https://github.com/conda/conda) - OS-agnostic, system-level binary package manager and ecosystem.

* [mamba](https://github.com/mamba-org/mamba) - The Fast Cross-Platform Package Manager.

* [pip](https://github.com/pypa/pip) - The Python package installer.

* [Poetry](https://github.com/python-poetry/poetry) - Python packaging and dependency management made easy.

* [pipx](https://github.com/pypa/pipx) - Install and Run Python Applications in Isolated Environments.

* [PDM](https://github.com/pdm-project/pdm) - A modern Python package and dependency manager supporting the latest PEP standards.

* [pip-tools](https://github.com/jazzband/pip-tools) - A set of tools to keep your pinned Python dependencies fresh.

* [pipreqs](https://github.com/bndr/pipreqs) **(not actively updated)** - Generate pip requirements.txt file based on imports of any project.

### For C++/C

* [Vcpkg](https://github.com/microsoft/vcpkg) - C++ Library Manager for Windows, Linux, and MacOS.

### For Scala

* [Coursier](https://github.com/coursier/coursier) - Pure Scala Artifact Fetching.

### For JavaScript

* [NVM for Windows](https://github.com/coreybutler/nvm-windows) - A node.js version management utility for Windows. Ironically written in Go.

* [cnpm](https://github.com/cnpm/cnpm) - npm client for China mirror of npm

## Containers & Language Extentions & Linting

* [Linguist](https://github.com/github/linguist) - This library is used on GitHub.com to detect blob languages, ignore binary or vendored files, suppress generated files in diffs, and generate language breakdown graphs.

* [cloc](https://github.com/AlDanial/cloc) - Counts blank lines, comment lines, and physical lines of source code in many programming languages.

* [ShellCheck](https://github.com/koalaman/shellcheck) - A static analysis tool for shell scripts.

* [Cosmos](https://github.com/OpenGenus/cosmos) - Cosmos is your personal offline collection of every algorithm and data structure one will ever encounter and use in a lifetime.

* [DevDocs](https://github.com/freeCodeCamp/devdocs) - API Documentation Browser.

* [The Silver Searcher](https://github.com/ggreer/the_silver_searcher) **(not actively updated)** - A code-searching tool similar to ack, but faster.

### For Python

* [CPython](https://github.com/python/cpython) - The Python programming language.

* [manylinux](https://github.com/pypa/manylinux) - Python wheels that work on any linux (almost).

* [pytest](https://github.com/pytest-dev/pytest) - The pytest framework makes it easy to write small tests, yet scales to support complex functional testing.

* [tqdm](https://github.com/tqdm/tqdm) - A Fast, Extensible Progress Bar for Python and CLI.

* [transitions](https://github.com/pytransitions/transitions) - A lightweight, object-oriented finite state machine implementation in Python with many extensions.

* [MicroPython](https://github.com/micropython/micropython) - A lean and efficient Python implementation for microcontrollers and constrained systems.

* [Pyston](https://github.com/pyston/pyston) - A faster and highly-compatible implementation of the Python programming language.

* [attrs](https://github.com/python-attrs/attrs) - Python Classes Without Boilerplate.

* [Boltons](https://github.com/mahmoud/boltons) - A set of over 230 BSD-licensed, pure-Python utilities in the same spirit as — and yet conspicuously missing from — the standard library.

* [GRequests](https://github.com/spyoungtech/grequests) - Allows you to use Requests with Gevent to make asynchronous HTTP Requests easily.

* [cachetools](https://github.com/tkem/cachetools) - Provides various memoizing collections and decorators, including variants of the Python Standard Library's @lru_cache function decorator.

* [More Itertools](https://github.com/more-itertools/more-itertools) - More routines for operating on iterables, beyond itertools.

* [Toolz](https://github.com/pytoolz/toolz) - A set of utility functions for iterators, functions, and dictionaries.

* [Funcy](https://github.com/Suor/funcy) - A collection of fancy functional tools focused on practicality.

* [Dependency Injector](https://github.com/ets-labs/python-dependency-injector) - A dependency injection framework for Python.

* [Tenacity](https://github.com/jd/tenacity) - An Apache 2.0 licensed general-purpose retrying library, written in Python, to simplify the task of adding retry behavior to just about anything.

* [returns](https://github.com/dry-python/returns) - Make your functions return something meaningful, typed, and safe.

* [wrapt](https://github.com/GrahamDumpleton/wrapt) - A Python module for decorators, wrappers and monkey patching.

* [Mypy](https://github.com/python/mypy) - A static type checker for Python.

* [Pyright](https://github.com/microsoft/pyright) - A fast type checker meant for large Python source bases.

* [pytype](https://github.com/google/pytype) - A static type analyzer for Python code.

* [Jedi](https://github.com/davidhalter/jedi) - Awesome autocompletion, static analysis and refactoring library for python.

* [Beartype](https://github.com/beartype/beartype) - Unbearably fast near-real-time runtime type-checking in pure Python.

* [Flake8](https://github.com/PyCQA/flake8) - A python tool that glues together pycodestyle, pyflakes, mccabe, and third-party plugins to check the style and quality of some python code.
  * <details open><summary>Related projects:</summary>

    * [wemake-python-styleguide](https://github.com/wemake-services/wemake-python-styleguide) - The strictest and most opinionated python linter ever.
  </details>

* [Pylint](https://github.com/PyCQA/pylint) - A static code analyser for Python 2 or 3.

* [isort](https://github.com/PyCQA/isort) - A Python utility / library to sort imports alphabetically, and automatically separated into sections and by type.

* [prospector](https://github.com/PyCQA/prospector) - Inspects Python source files and provides information about type and location of classes, methods etc.

* [Pyre](https://github.com/facebook/pyre-check) - Performant type-checking for python.

* [YAPF](https://github.com/google/yapf) - A formatter for Python files.

* [Black](https://github.com/psf/black) - The uncompromising Python code formatter.

* [autopep8](https://github.com/hhatto/autopep8) - A tool that automatically formats Python code to conform to the PEP 8 style guide.

* [rope](https://github.com/python-rope/rope) - A python refactoring library.

* [pyupgrade](https://github.com/asottile/pyupgrade) - A tool (and pre-commit hook) to automatically upgrade syntax for newer versions of the language.

* [Vulture](https://github.com/jendrikseipp/vulture) - Finds unused code in Python programs.

* [algorithms](https://github.com/keon/algorithms) - Minimal examples of data structures and algorithms in Python.

* [DeepDiff](https://github.com/seperman/deepdiff) - Deep Difference and search of any Python object/data.

* [Pygments](https://github.com/pygments/pygments) - A generic syntax highlighter written in Python.

* [pybind11](https://github.com/pybind/pybind11) - Seamless operability between C++11 and Python.

* [cppimport](https://github.com/tbenthompson/cppimport) - Import C++ files directly from Python.

* [jupyter_contrib_nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) - A collection of various notebook extensions for Jupyter.

### For C++/C

* [Folly](https://github.com/facebook/folly) - An open-source C++ library developed and used at Facebook.

* [gflags](https://github.com/gflags/gflags) - Contains a C++ library that implements commandline flags processing. It includes built-in support for standard types such as string and the ability to define flags in the source file in which they are used.

* [GoogleTest](https://github.com/google/googletest) - Google Testing and Mocking Framework.

* [Catch2](https://github.com/catchorg/Catch2) - A modern, C++-native, test framework for unit-tests, TDD and BDD - using C++14, C++17 and later (C++11 support is in v2.x branch, and C++03 on the Catch1.x branch).

* [Ninja](https://github.com/ninja-build/ninja) - A small build system with a focus on speed.

* [Coost](https://github.com/idealvin/coost) - A tiny boost library in C++11.

* [AsmJit](https://github.com/asmjit/asmjit) - A lightweight library for machine code generation written in C++ language.

* [fmt](https://github.com/fmtlib/fmt) - A modern formatting library.

* [gperftools](https://github.com/gperftools/gperftools) - a collection of a high-performance multi-threaded malloc() implementation, plus some pretty nifty performance analysis
tools.

* [jemalloc](https://github.com/jemalloc/jemalloc) - A general purpose malloc(3) implementation that emphasizes
fragmentation avoidance and scalable concurrency support.

* [libhv](https://github.com/ithewei/libhv) - A c/c++ network library for developing TCP/UDP/SSL/HTTP/WebSocket/MQTT client/server.

* [cpp-sort](https://github.com/Morwenn/cpp-sort) - Sorting algorithms & related tools for C++14.

* [SimpleGPUHashTable](https://github.com/nosferalatu/SimpleGPUHashTable) - A simple GPU hash table implemented in CUDA using lock free techniques.

* [PJON](https://github.com/gioblu/PJON) - An experimental, arduino-compatible, multi-master, multi-media network protocol.

* [cppman](https://github.com/aitjcize/cppman) - C++ 98/11/14 manual pages for Linux/MacOS.

* [cpp-docs](https://github.com/MicrosoftDocs/cpp-docs) - Visual Studio documentation for Microsoft C++.

* [vscode-leetcode](https://github.com/LeetCode-OpenSource/vscode-leetcode) - Solve LeetCode problems in VS Code.

* [Nano](https://github.com/refnum/Nano) **(not actively updated)** - High-performance C++ for macOS, iOS, tvOS, Android, Linux, and Windows.

* [leetcode-cli](https://github.com/skygragon/leetcode-cli) **(not actively updated)** - A cli tool to enjoy leetcode.

### For Go

* [Realize](https://github.com/oxequa/realize) - Golang Task Runner which enhance your workflow by automating the most common tasks and using the best performing Golang live reloading.

* [GCache](https://github.com/bluele/gcache) - An in-memory cache library for golang. It supports multiple eviction policies: LRU, LFU, ARC.

* [Gonum](https://github.com/gonum/gonum) - A set of numeric libraries for the Go programming language. It contains libraries for matrices, statistics, optimization, and more.

* [sh](https://github.com/mvdan/sh) - A shell parser, formatter, and interpreter with bash support; includes shfmt.

* [gotests](https://github.com/cweill/gotests) - Automatically generate Go test boilerplate from your source code.

* [goproxy](https://github.com/goproxyio/goproxy) - A global proxy for Go modules.

* [go-echarts](https://github.com/go-echarts/go-echarts) - The adorable charts library for Golang.

* [revive](https://github.com/mgechev/revive) - ~6x faster, stricter, configurable, extensible, and beautiful drop-in replacement for golint.

* [depth](https://github.com/KyleBanks/depth) - Visualize Go Dependency Trees.

* [gophernotes](https://github.com/gopherdata/gophernotes) - The Go kernel for Jupyter notebooks and nteract.

### For Java

* [JavaCPP](https://github.com/bytedeco/javacpp) - The missing bridge between Java and native C++.

* [OkHttp](https://github.com/square/okhttp) - Square’s meticulous HTTP client for the JVM, Android, and GraalVM.

### For Scala

* [Ammonite](https://github.com/com-lihaoyi/Ammonite) - Scala Scripting.

* [ammonite-spark](https://github.com/alexarchambault/ammonite-spark) - Run spark calculations from Ammonite.

* [almond](https://github.com/almond-sh/almond) - A Scala kernel for Jupyter.

* [OS-Lib](https://github.com/com-lihaoyi/os-lib) - A simple, flexible, high-performance Scala interface to common OS filesystem and subprocess APIs.

## For JavaScript

* [nan](https://github.com/nodejs/nan) - Native Abstractions for Node.js

## Programming Language Tutorials

* [developer-roadmap](https://github.com/kamranahmedse/developer-roadmap) - Interactive roadmaps, guides and other educational content to help developers grow in their careers.

* [freeCodeCamp.org](https://github.com/freeCodeCamp/freeCodeCamp) - freeCodeCamp.org's open-source codebase and curriculum. Learn to code for free.

* [Coding Interview University](https://github.com/jwasham/coding-interview-university) - A complete computer science study plan to become a software engineer.

* [kdn251/interviews](https://github.com/kdn251/interviews) - Your personal guide to Software Engineering technical interviews.

* [free-programming-books](https://github.com/EbookFoundation/free-programming-books) - Freely available programming books.

* [build-your-own-x](https://github.com/codecrafters-io/build-your-own-x) - Master programming by recreating your favorite technologies from scratch.

* [iHateRegex](https://github.com/geongeorge/i-hate-regex) - The code for iHateregex.io - The Regex Cheat Sheet

* [The System Design Primer](https://github.com/donnemartin/system-design-primer) - Learn how to design large-scale systems. Prep for the system design interview. Includes Anki flashcards.

* [Algorithm Visualizer](https://github.com/algorithm-visualizer/algorithm-visualizer) - Interactive Online Platform that Visualizes Algorithms from Code

* [CMake Examples](https://github.com/ttroy50/cmake-examples) - Useful CMake Examples.

* [SoftwareArchitect](https://github.com/justinamiller/SoftwareArchitect) - Path to a Software Architect.

* [andkret/Cookbook](https://github.com/andkret/Cookbook) - The Data Engineering Cookbook.

* [servehappy-resources](https://github.com/WordPress/servehappy-resources) - Third-party articles and specific tutorials on PHP upgrades.

* [Java and Spring Tutorials](https://github.com/eugenp/tutorials) - A collection of small and focused tutorials - each covering a single and well defined area of development in the Java ecosystem.

* [gitignore](https://github.com/github/gitignore) - A collection of useful .gitignore templates.

* [God-Of-BigData](https://github.com/wangzhiwubigdata/God-Of-BigData) - 专注大数据学习面试，大数据成神之路开启。Flink/Spark/Hadoop/Hbase/Hive...

* [CS-Notes](https://github.com/CyC2018/CS-Notes) - 技术面试必备基础知识、Leetcode、计算机操作系统、计算机网络、系统设计

* [LeetCodeAnimation](https://github.com/MisterBooo/LeetCodeAnimation) - 用动画的形式呈现解LeetCode题目的思路

* [fucking-algorithm](https://github.com/labuladong/fucking-algorithm) - labuladong 的算法小抄。

* [JS-Sorting-Algorithm](https://github.com/hustcc/JS-Sorting-Algorithm) - 一本关于排序算法的 GitBook 在线书籍 《十大经典排序算法》，多语言实现。

* [apachecn-algo-zh](https://github.com/apachecn/apachecn-algo-zh) - ApacheCN 数据结构与算法译文集

* [rust-based-os-comp2022](https://github.com/LearningOS/rust-based-os-comp2022) - 2022开源操作系统训练营。

* [free-programming-books-zh_CN](https://github.com/justjavac/free-programming-books-zh_CN) - 免费的计算机编程类中文书籍

* [technical-books](https://github.com/doocs/technical-books) - 国内外互联网技术大牛们都写了哪些书籍：计算机基础、网络、前端、后端、数据库、架构、大数据、深度学习。

* [Learn-Git-in-30-days](https://github.com/doggy8088/Learn-Git-in-30-days) - 30 天精通 Git 版本控管

* [BAT_interviews](https://github.com/lengyue1024/BAT_interviews) - 分享最新BAT面试题(包含机器学习,Linux,PHP,大数据,Python,Java,前端...)

* [helloworld](https://github.com/Prithvirajbilla/helloworld) **(not actively updated)** - Helloworld programs in different languages.

* [500 Lines or Less](https://github.com/aosabook/500lines) **(not actively updated)** - This is the source for the book 500 Lines or Less, the fourth in the Architecture of Open Source Applications series.

* [Simple Computer](https://github.com/djhworld/simple-computer) **(not actively updated)** - the scott CPU from "But How Do It Know?" by J. Clark Scott

* [How-to-Make-a-Computer-Operating-System](https://github.com/SamyPesse/How-to-Make-a-Computer-Operating-System) **(not actively updated)** - How to Make a Computer Operating System in C++.

* [phodal/github](https://github.com/phodal/github) **(not actively updated)** - GitHub 漫游指南

* [fullstack-data-engineer](https://github.com/Honlan/fullstack-data-engineer) **(not actively updated)** - 全栈数据工程师养成攻略

* [Micro8](https://github.com/Micropoor/Micro8) **(not actively updated)** - 渗透攻击教程

### Python

* [30 Days Of Python](https://github.com/Asabeneh/30-Days-Of-Python) - A step-by-step guide to learn the Python programming language in 30 days.

* [numpy-100](https://github.com/rougier/numpy-100) - 100 numpy exercises (with solutions).

* [python-patterns](https://github.com/faif/python-patterns) - A collection of design patterns/idioms in Python.

* [python_example](https://github.com/pybind/python_example) - Example pybind11 module built with a Python-based build system.

* [pbpython](https://github.com/chris1610/pbpython) - Code, Notebooks and Examples from Practical Business Python.

* [Python-100-Days](https://github.com/jackfrued/Python-100-Days) - Python - 100天从新手到大师

* [walter201230/Python](https://github.com/walter201230/Python) - 最良心的 Python 教程

* [tech-cow/leetcode](https://github.com/tech-cow/leetcode) **(not actively updated)** - leetcode solutions for Humans.

* [qiwsir/algorithm](https://github.com/qiwsir/algorithm) **(not actively updated)** - Python算法题解

* [AlgorithmsByPython](https://github.com/Jack-Lee-Hiter/AlgorithmsByPython) **(not actively updated)** - 算法/数据结构/Python/剑指offer/机器学习/leetcode

### C++/C

* [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines) - A set of tried-and-true guidelines, rules, and best practices about coding in C++.

* [Modern C++ Tutorial](https://github.com/changkun/modern-cpp-tutorial) - Modern C++ Tutorial: C++11/14/17/20 On the Fly.

* [modern-cpp-features](https://github.com/AnthonyCalandra/modern-cpp-features) - A cheatsheet of modern C++ language and library features.

* [design-patterns-cpp](https://github.com/JakubVojvoda/design-patterns-cpp) - C++ Design Patterns.

* [haoel/leetcode](https://github.com/haoel/leetcode) - LeetCode Problems' Solutions.

* [pezy/LeetCode](https://github.com/pezy/LeetCode) - LeetCode solutions in C++ 11 and Python3.

* [CPlusPlusThings](https://github.com/Light-City/CPlusPlusThings) - 《C++ 那些事》。

* [huihut/interview](https://github.com/huihut/interview) - C/C++ 技术面试基础知识总结，包括语言、程序库、数据结构、算法、系统、网络、链接装载库等知识及面试经验、招聘、内推等信息。

* [flash-linux0.11-talk](https://github.com/sunym1993/flash-linux0.11-talk) - 像小说一样品读 Linux 0.11 核心代码。

* [Cplusplus-Concurrency-In-Practice](https://github.com/forhappy/Cplusplus-Concurrency-In-Practice) **(not actively updated)** - C++ 并发编程指南

* [SGI-STL](https://github.com/steveLauwh/SGI-STL) **(not actively updated)** - SGI-STL V3.3 源代码的学习

### Go

* [the-way-to-go_ZH_CN](https://github.com/unknwon/the-way-to-go_ZH_CN) - 《The Way to Go》中文译本，中文正式名《Go 入门指南》

* [GoGuide](https://github.com/coderit666/GoGuide) - 一份涵盖大部分 Golang 程序员所需要掌握的核心知识，拥有 Go语言教程、Go开源书籍、Go语言入门教程、Go语言学习路线。

### Java

* [JavaGuide](https://github.com/Snailclimb/JavaGuide) - 「Java学习+面试指南」一份涵盖大部分 Java 程序员所需要掌握的核心知识。

* [hello-algorithm](https://github.com/geekxh/hello-algorithm) - 针对小白的算法训练，包括四部分：大厂面经，力扣图解，千本开源电子书，百张技术思维导图。

### Scala

* [spark-scala-examples](https://github.com/spark-examples/spark-scala-examples) - Provides Apache Spark SQL, RDD, DataFrame and Dataset examples in Scala language.

### Flutter

* [FlutterExampleApps](https://github.com/iampawan/FlutterExampleApps) - Basic Flutter apps, for flutter devs.

* [awesome-flutter](https://github.com/Solido/awesome-flutter) - An awesome list that curates the best Flutter libraries, tools, tutorials, articles and more.

### JavaScript

* [You Don't Know JS Yet](https://github.com/getify/You-Dont-Know-JS) - A book series on JavaScript.

---

# Useful Tools

* [Shields.io](https://github.com/badges/shields) - Concise, consistent, and legible badges in SVG and raster format.

* [Badges 4 README.md Profile](https://github.com/alexandresanlim/Badges4-README.md-Profile) - Improve your README.md profile with these amazing badges.

* [best-resume-ever](https://github.com/salomonelli/best-resume-ever) - Build fast and easy multiple beautiful resumes and create your best CV ever! Made with Vue and LESS.

* [Deedy-Resume](https://github.com/deedy/Deedy-Resume) - A one page , two asymmetric column resume template in XeTeX that caters to an undergraduate Computer Science student.

* [Public Sans](https://github.com/uswds/public-sans) - A strong, neutral, principles-driven, open source typeface for text or display.

* [paper-tips-and-tricks](https://github.com/Wookai/paper-tips-and-tricks) - Best practice and tips & tricks to write scientific papers in LaTeX, with figures generated in Python or Matlab.

* [arxiv-sanity lite](https://github.com/karpathy/arxiv-sanity-lite) - Tag arxiv papers of interest get recommendations of similar papers in a nice UI using SVMs over tfidf feature vectors based on paper abstracts.

* [arXiv LaTeX Cleaner](https://github.com/google-research/arxiv-latex-cleaner) - Easily clean the LaTeX code of your paper to submit to arXiv.

* [Conference-Acceptance-Rate](https://github.com/lixin4ever/Conference-Acceptance-Rate) - Acceptance rates for the major AI conferences.

* [CODELF](https://github.com/unbug/codelf) - A search tool helps dev to solve the naming things problem.

* [Apollo-11](https://github.com/chrislgarry/Apollo-11) - Original Apollo 11 Guidance Computer (AGC) source code for the command and lunar modules.

* [ChromeAppHeroes](https://github.com/zhaoolee/ChromeAppHeroes) - 谷粒-Chrome插件英雄榜, 为优秀的Chrome插件写一本中文说明书, 让Chrome插件英雄们造福人类

* [Awesome Resume for Chinese](https://github.com/dyweb/awesome-resume-for-chinese) - 适合中文的简历模板收集（LaTeX，HTML/JS and so on）

* [code6](https://github.com/4x99/code6) - 码小六 - GitHub 代码泄露监控系统

* [howto-make-more-money](https://github.com/easychen/howto-make-more-money) - 程序员如何优雅的挣零花钱，2.0版

* [USTC-Course](https://github.com/USTC-Resource/USTC-Course) - 中国科学技术大学课程资源

* [FLY_US](https://github.com/F4bwDP6a6W/FLY_US) - 美国大学备考资料

* [996.ICU](https://github.com/996icu/996.ICU) - 996加班的公司名单

* [955.WLB](https://github.com/formulahendry/955.WLB) - 955不加班的公司名单

* [Badges](https://github.com/boennemann/badges) **(not actively updated)** - A collection of all JavaScript related and free for open-source readme badges out there.

* [Github Monitor](https://github.com/VKSRC/Github-Monitor) **(not actively updated)** - Github信息泄漏监控系统

## MacOS

* [Scroll-Reverser](https://github.com/pilotmoon/Scroll-Reverser) - Reverses the direction of macOS scrolling, with independent settings for trackpads and mice.

* [Hex Fiend](https://github.com/HexFiend/HexFiend) - A fast and clever hex editor for macOS.

* [iterm2-zmodem](https://github.com/aikuyun/iterm2-zmodem) - 在 Mac 下，实现与服务器进行便捷的文件上传和下载操作。

## Windows

* [winget](https://github.com/microsoft/winget-cli) - Windows Package Manager Client.

* [Scoop](https://github.com/ScoopInstaller/Scoop) - A command-line installer for Windows.

* [Windows Terminal](https://github.com/microsoft/terminal) - The new Windows Terminal and the original Windows console host, all in the same place!

* [Windows Calculator](https://github.com/microsoft/calculator) - A simple yet powerful calculator that ships with Windows.

* [WoeUSB](https://github.com/WoeUSB/WoeUSB) - A Microsoft Windows USB installation media preparer for GNU+Linux.

* [ReShade](https://github.com/crosire/reshade) - A generic post-processing injector for games and video software.

* [pygta5](https://github.com/Sentdex/pygta5) - Explorations of Using Python to play Grand Theft Auto 5.

* [Borderless Gaming](https://github.com/Codeusa/Borderless-Gaming) - Play your favorite games in a borderless window; no more time consuming alt-tabs.

* [Revive Compatibility Layer](https://github.com/LibreVR/Revive) - Play Oculus-exclusive games on the HTC Vive or Valve Index.

* [QuickLook](https://github.com/QL-Win/QuickLook) - Bring macOS “Quick Look” feature to Windows.

* [Debloat Windows 10](https://github.com/W4RH4WK/Debloat-Windows-10) - A Collection of Scripts Which Disable / Remove Windows 10 Features and Apps.

* [CleanMyWechat](https://github.com/blackboxo/CleanMyWechat) - 自动删除 PC 端微信缓存数据，包括从所有聊天中自动下载的大量文件、视频、图片等数据内容，解放你的空间。

* [Watt Toolkit](https://github.com/BeyondDimension/SteamTools) - 一个开源跨平台的多功能 Steam 工具箱。

## Linux

* [tmux](https://github.com/tmux/tmux) - A terminal multiplexer: it enables a number of terminals to be created, accessed, and controlled from a single screen. tmux may be detached from a screen and continue running in the background, then later reattached.

* [Proton](https://github.com/ValveSoftware/Proton) - Compatibility tool for Steam Play based on Wine and additional components.

* [Lutris](https://github.com/lutris/lutris) - Lutris helps you install and play video games from all eras and from most gaming systems.

* [GIT quick statistics](https://github.com/arzzen/git-quick-stats) - Git quick statistics is a simple and efficient way to access various statistics in git repository.

* [git-fame](https://github.com/oleander/git-fame-rb) - A command-line tool that helps you summarize and pretty-print collaborators based on contributions.

* [Hercules](https://github.com/src-d/hercules) - Gaining advanced insights from Git repository history.

* [Gitinspector](https://github.com/ejwa/gitinspector) - The statistical analysis tool for git repositories.

* [Persepolis](https://github.com/persepolisdm/persepolis) **(not actively updated)** - A download manager & a GUI for Aria2.

* [doubi](https://github.com/ToyoDAdoubi/doubi) **(not actively updated)** - 一个逗比写的各种逗比脚本

## Cross-Platform

* [Glances](https://github.com/nicolargo/glances) - A top/htop alternative for GNU/Linux, BSD, Mac OS and Windows operating systems.

* [gpustat](https://github.com/wookayin/gpustat) - A simple command-line utility for querying and monitoring GPU status.

* [NVTOP](https://github.com/Syllo/nvtop) - GPUs process monitoring for AMD, Intel and NVIDIA.

* [s-tui](https://github.com/amanusk/s-tui) - Terminal-based CPU stress and monitoring utility.

* [Tabby](https://github.com/Eugeny/tabby) - A terminal for a more modern age.

* [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) - A delightful community-driven (with 2,000+ contributors) framework for managing your zsh configuration.

* [oh-my-posh](https://github.com/jandedobbeleer/oh-my-posh) - A prompt theme engine for any shell.

* [PowerShell](https://github.com/PowerShell/PowerShell) - PowerShell for every system!

* [fish](https://github.com/fish-shell/fish-shell) - The user-friendly command line shell.

* [The Fuck](https://github.com/nvbn/thefuck) - Magnificent app which corrects your previous console command.

* [Nerd Fonts](https://github.com/ryanoasis/nerd-fonts) - Iconic font aggregator, collection, & patcher. 3,600+ icons, 50+ patched fonts: Hack, Source Code Pro, more. Glyph collections: Font Awesome, Material Design Icons, Octicons, & more.

* [LANDrop](https://github.com/LANDrop/LANDrop) - A cross-platform tool that you can use to conveniently transfer photos, videos, and other types of files to other devices on the same local network.

* [ImageMagick 7](https://github.com/ImageMagick/ImageMagick) - Use ImageMagick to create, edit, compose, or convert digital images.

* [MyPaint](https://github.com/mypaint/mypaint) - A simple drawing and painting program that works well with Wacom-style graphics tablets.

* [LosslessCut](https://github.com/mifi/lossless-cut) - The swiss army knife of lossless video/audio editing.

* [LuminanceHDR](https://github.com/LuminanceHDR/LuminanceHDR) - A complete workflow for HDR imaging.

* [Gifcurry](https://github.com/lettier/gifcurry) - The open-source, Haskell-built video editor for GIF makers.

* [GitHub Desktop](https://github.com/desktop/desktop) - Focus on what matters instead of fighting with Git.

* [Refined GitHub](https://github.com/refined-github/refined-github) - Browser extension that simplifies the GitHub interface and adds useful features.

* [Foam](https://github.com/foambubble/foam) - A personal knowledge management and sharing system for VSCode.

* [Notable](https://github.com/notable/notable) - The Markdown-based note-taking app that doesn't suck.

* [Atom](https://github.com/atom/atom) - The hackable text editor.

* [Fusuma](https://github.com/hiroppy/fusuma) - Makes slides with Markdown easily.

* [Kilo](https://github.com/antirez/kilo) - A text editor in less than 1000 LOC with syntax highlight and search.

* [lint-md](https://github.com/lint-md/lint-md) - 检查中文 markdown 编写格式规范的命令行工具，基于 AST，方便集成 CI，写博客 / 文档必备。支持 API 调用

* [Mailspring](https://github.com/Foundry376/Mailspring) - A beautiful, fast and fully open source mail client for Mac, Windows and Linux.

* [Google Earth Enterprise](https://github.com/google/earthenterprise) - The open source release of Google Earth Enterprise, a geospatial application which provides the ability to build and host custom 3D globes and 2D maps.

* [carbon](https://github.com/carbon-app/carbon) - Create and share beautiful images of your source code.

* [vscode-python](https://github.com/microsoft/vscode-python) - Python extension for Visual Studio Code.

* [vscode-cpptools](https://github.com/microsoft/vscode-cpptools) - Official repository for the Microsoft C/C++ extension for VS Code.

* [code-server](https://github.com/coder/code-server) - VS Code in the browser.

* [Gradle](https://github.com/gradle/gradle) - A build tool with a focus on build automation and support for multi-language development.

* [LiteIDE](https://github.com/visualfc/liteide) - A simple, open source, cross-platform Go IDE.

* [YouCompleteMe](https://github.com/ycm-core/YouCompleteMe) - A code-completion engine for Vim.

* [readme-md-generator](https://github.com/kefranabg/readme-md-generator) - CLI that generates beautiful README.md files.

* [pdfdiff](https://github.com/cascremers/pdfdiff) - Command-line tool to inspect the difference between (the text in) two PDF files.

* [Rufus](https://github.com/pbatard/rufus) - The Reliable USB Formatting Utility.

* [projectM](https://github.com/projectM-visualizer/projectm) - Cross-platform music visualization.

* [Syncthing](https://github.com/syncthing/syncthing) - Open Source Continuous File Synchronization.

* [PCSX2](https://github.com/PCSX2/pcsx2) - The Playstation 2 Emulator.

* [PPSSPP](https://github.com/hrydgard/ppsspp) - A PSP emulator for Android, Windows, Mac and Linux, written in C++.

* [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator written in Python.

* [libtorrent](https://github.com/arvidn/libtorrent) - An efficient feature complete C++ bittorrent implementation.

* [qBittorrent-Enhanced-Edition](https://github.com/c0re100/qBittorrent-Enhanced-Edition) - [Unofficial] qBittorrent Enhanced, based on qBittorrent

* [trackerslist](https://github.com/ngosang/trackerslist) - Updated list of public BitTorrent trackers.

* [TrackersListCollection](https://github.com/XIU2/TrackersListCollection) - A list of popular BitTorrent Trackers.

* [bittorrent-tracker](https://github.com/webtorrent/bittorrent-tracker) - Simple, robust, BitTorrent tracker (client & server) implementation.

* [ShareX](https://github.com/ShareX/ShareX) - A free and open source program that lets you capture or record any area of your screen and share it with a single press of a key.

* [Streamlabs Desktop](https://github.com/stream-labs/desktop) - Free and open source streaming software built on OBS and Electron.

* [SwitchHosts](https://github.com/oldj/SwitchHosts) - Switch hosts quickly.

* [Albert](https://github.com/albertlauncher/albert) - A fast and flexible keyboard launcher.

* [Kindle_download_helper](https://github.com/yihong0618/Kindle_download_helper) - Download all your kindle books script.

* [GitHub520](https://github.com/521xueweihan/GitHub520) - 让你“爱”上 GitHub，解决访问时图裂、加载慢的问题。

* [Peek](https://github.com/phw/peek) **(not actively updated)** - Simple animated GIF screen recorder with an easy to use interface.

* [GayHub](https://github.com/jawil/GayHub) **(not actively updated)** - An awesome chrome extension for github.

---

# Other Awesome Lists

* [sindresorhus/awesome](https://github.com/sindresorhus/awesome) - Awesome lists about all kinds of interesting topics.

## Machine Learning

* [ml-tooling/best-of-ml-python](https://github.com/ml-tooling/best-of-ml-python) - A ranked list of awesome machine learning Python libraries.
  
* [josephmisiti/awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning) - A curated list of awesome Machine Learning frameworks, libraries and software.

* [ChristosChristofidis/awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning) - A curated list of awesome Deep Learning tutorials, projects and communities.

* [floodsung/Deep-Learning-Papers-Reading-Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap) - Deep Learning papers reading roadmap for anyone who are eager to learn this amazing tech.

* [sbrugman/deep-learning-papers](https://github.com/sbrugman/deep-learning-papers) - Papers about deep learning ordered by task, date.

* [terryum/awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers) - The most cited deep learning papers.

* [aleju/papers](https://github.com/aleju/papers) - Summaries of machine learning papers.

* [abhshkdz/papers](https://github.com/abhshkdz/papers) - Summaries of papers on deep learning.

* [RedditSota/state-of-the-art-result-for-machine-learning-problems](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems) - This repository provides state of the art (SoTA) results for all machine learning problems.

* [bharathgs/Awesome-pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list) - A comprehensive list of pytorch related content on github,such as different models,implementations,helper libraries,tutorials etc.

* [jbhuang0604/awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision) - A curated list of awesome computer vision resources.

* [xinghaochen/awesome-hand-pose-estimation](https://github.com/xinghaochen/awesome-hand-pose-estimation) - Awesome work on hand pose estimation/tracking

* [cbsudux/awesome-human-pose-estimation](https://github.com/cbsudux/awesome-human-pose-estimation) - A collection of awesome resources in Human Pose estimation.

* [ChaofWang/Awesome-Super-Resolution](https://github.com/ChaofWang/Awesome-Super-Resolution) - Collect super-resolution related papers, data, repositories

* [flyywh/Image-Denoising-State-of-the-art](https://github.com/flyywh/Image-Denoising-State-of-the-art) - A curated list of image denoising resources and a benchmark for image denoising approaches.

* [wenbihan/reproducible-image-denoising-state-of-the-art](https://github.com/wenbihan/reproducible-image-denoising-state-of-the-art) - Collection of popular and reproducible image denoising works.

* [openMVG/awesome_3DReconstruction_list](https://github.com/openMVG/awesome_3DReconstruction_list) - A curated list of papers & resources linked to 3D reconstruction from images.

* [hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo) - A list of all named GANs.

* [savan77/The-GAN-World](https://github.com/savan77/The-GAN-World) - Everything about Generative Adversarial Networks.

* [nashory/gans-awesome-applications](https://github.com/nashory/gans-awesome-applications) - Curated list of awesome GAN applications and demo.

* [wiseodd/generative-models](https://github.com/wiseodd/generative-models) - Collection of generative models, e.g. GAN, VAE in Pytorch and Tensorflow.

* [NVIDIAAICITYCHALLENGE/2020AICITY_Code_From_Top_Teams](https://github.com/NVIDIAAICITYCHALLENGE/2020AICITY_Code_From_Top_Teams) - The code from the top teams in the 2020 AI City Challenge

* [keon/awesome-nlp](https://github.com/keon/awesome-nlp) - A curated list of resources dedicated to Natural Language Processing (NLP).

* [NLP-progress](https://github.com/sebastianruder/NLP-progress) - Repository to track the progress in Natural Language Processing (NLP), including the datasets and the current state-of-the-art for the most common NLP tasks.

* [niderhoff/nlp-datasets](https://github.com/niderhoff/nlp-datasets) - Alphabetical list of free/public domain datasets with text data for use in Natural Language Processing (NLP).

* [wzhe06/Reco-papers](https://github.com/wzhe06/Reco-papers) - Classic papers and resources on recommendation

* [hongleizhang/RSPapers](https://github.com/hongleizhang/RSPapers) - A Curated List of Must-read Papers on Recommender System.

* [wnzhang/rtb-papers](https://github.com/wnzhang/rtb-papers) - A collection of research and survey papers of real-time bidding (RTB) based display advertising techniques.

* [wzhe06/Ad-papers](https://github.com/wzhe06/Ad-papers) - Papers on Computational Advertising.

* [rguo12/awesome-causality-algorithms](https://github.com/rguo12/awesome-causality-algorithms) - An index of algorithms for learning causality with data.

* [thunlp/GNNPapers](https://github.com/thunlp/GNNPapers) - Must-read papers on graph neural networks (GNN).

* [thunlp/NRLPapers](https://github.com/thunlp/NRLPapers) - Must-read papers on network representation learning (NRL) / network embedding (NE).

* [subeeshvasu/Awesome-Learning-with-Label-Noise](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise) - A curated list of resources for Learning with Noisy Labels.

* [grananqvist/Awesome-Quant-Machine-Learning-Trading](https://github.com/grananqvist/Awesome-Quant-Machine-Learning-Trading) - Quant/Algorithm trading resources with an emphasis on Machine Learning.

* [zhangqianhui/AdversarialNetsPapers](https://github.com/zhangqianhui/AdversarialNetsPapers) - Awesome paper list with code about generative adversarial nets.

* [jindongwang/transferlearning](https://github.com/jindongwang/transferlearning) - Transfer learning / domain adaptation / domain generalization / multi-task learning etc. Papers, codes, datasets, applications, tutorials.

* [zhaoxin94/awesome-domain-adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation) - A collection of AWESOME things about domian adaptation.

* [markdtw/awesome-architecture-search](https://github.com/markdtw/awesome-architecture-search) - A curated list of awesome architecture search resources.

* [Yochengliu/awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis) - A list of papers and datasets about point cloud analysis (processing).

* [AgaMiko/data-augmentation-review](https://github.com/AgaMiko/data-augmentation-review) - List of useful data augmentation resources. You will find here some not common techniques, libraries, links to GitHub repos, papers, and others.

* [academic/awesome-datascience](https://github.com/academic/awesome-datascience) - An awesome Data Science repository to learn and apply for real world problems.

* [r0f1/datascience](https://github.com/r0f1/datascience) - Curated list of Python resources for data science.

* [amusi/CVPR2022-Papers-with-Code](https://github.com/amusi/CVPR2022-Papers-with-Code) - CVPR 2022 论文和开源项目合集

* [extreme-assistant/CVPR2022-Paper-Code-Interpretation](https://github.com/extreme-assistant/CVPR2022-Paper-Code-Interpretation) - cvpr2022/cvpr2021/cvpr2020/cvpr2019/cvpr2018/cvpr2017 论文/代码/解读/直播合集，极市团队整理

* [extreme-assistant/ICCV2021-Paper-Code-Interpretation](https://github.com/extreme-assistant/ICCV2021-Paper-Code-Interpretation) - ICCV2021/2019/2017 论文/代码/解读/直播合集，极市团队整理

* [linyiqun/DataMiningAlgorithm](https://github.com/linyiqun/DataMiningAlgorithm) - 数据挖掘18大算法实现以及其他相关经典DM算法

## Computer Graphics

* [waitin2010/awesome-computer-graphics](https://github.com/waitin2010/awesome-computer-graphics) - A curated list of awesome computer graphics.

* [ellisonleao/magictools](https://github.com/ellisonleao/magictools) - A list of Game Development resources to make magic happen.

* [ericjang/awesome-graphics](https://github.com/ericjang/awesome-graphics) - Curated list of computer graphics tutorials and resources

* [luisnts/awesome-computer-graphics](https://github.com/luisnts/awesome-computer-graphics) - A curated list of awesome stuff to learn computer graphics

* [utilForever/game-developer-roadmap](https://github.com/utilForever/game-developer-roadmap) - Roadmap to becoming a game developer in 2022.

## Programming Language

* [MunGell/awesome-for-beginners](https://github.com/MunGell/awesome-for-beginners) - A list of awesome beginners-friendly projects.

* [papers-we-love/papers-we-love](https://github.com/papers-we-love/papers-we-love) - Papers from the computer science community to read and discuss.

* [practical-tutorials/project-based-learning](https://github.com/practical-tutorials/project-based-learning) - Curated list of project-based tutorials.

* [tayllan/awesome-algorithms](https://github.com/tayllan/awesome-algorithms) - A curated list of awesome places to learn and/or practice algorithms.

* [sdmg15/Best-websites-a-programmer-should-visit](https://github.com/sdmg15/Best-websites-a-programmer-should-visit) - Some useful websites for programmers.

* [orsanawwad/awesome-roadmaps](https://github.com/orsanawwad/awesome-roadmaps) - View roadmaps about developer roles to help you learn.

* [ml-tooling/best-of-python](https://github.com/ml-tooling/best-of-python) - A ranked list of awesome Python open-source libraries and tools.

* [vinta/awesome-python](https://github.com/vinta/awesome-python) - A curated list of awesome Python frameworks, libraries, software and resources.

* [lord63/awesome-python-decorator](https://github.com/lord63/awesome-python-decorator) - A curated list of awesome python decorator resources.

* [fffaraz/awesome-cpp](https://github.com/fffaraz/awesome-cpp) - A curated list of awesome C++ (or C) frameworks, libraries, resources, and shiny things.

* [avelino/awesome-go](https://github.com/avelino/awesome-go) - A curated list of awesome Go frameworks, libraries and software.

* [onqtam/awesome-cmake](https://github.com/onqtam/awesome-cmake) - A curated list of awesome CMake resources, scripts, modules and examples.

* [dkhamsing/open-source-ios-apps](https://github.com/dkhamsing/open-source-ios-apps) - Collaborative List of Open-Source iOS Apps.

* [pcqpcq/open-source-android-apps](https://github.com/pcqpcq/open-source-android-apps) - Open-Source Android Apps.

* [Kr1s77/awesome-python-login-model](https://github.com/Kr1s77/awesome-python-login-model) - python模拟登陆一些大型网站，还有一些简单的爬虫

* [jobbole/awesome-python-cn](https://github.com/jobbole/awesome-python-cn) - Python资源大全中文版，包括：Web框架、网络爬虫、模板引擎、数据库、数据可视化、图片处理等

* [jobbole/awesome-c-cn](https://github.com/jobbole/awesome-c-cn) - C 资源大全中文版，包括了：构建系统、编译器、数据库、加密、初中高的教程/指南、书籍、库等。

* [Quorafind/golang-developer-roadmap-cn](https://github.com/Quorafind/golang-developer-roadmap-cn) - 在 2019 成为一名 Go 开发者的路线图。为学习 Go 的人而准备。

* [jobbole/awesome-java-cn](https://github.com/jobbole/awesome-java-cn) - Java资源大全中文版，包括开发库、开发工具、网站、博客、微信、微博等

* [jobbole/awesome-javascript-cn](https://github.com/jobbole/awesome-javascript-cn) - JavaScript 资源大全中文版，内容包括：包管理器、加载器、测试框架、运行器、QA、MVC框架和库、模板引擎等。
