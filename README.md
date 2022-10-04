# Awesome List
A list of useful stuff in Machine Learning, Computer Graphics, Software Development, Mathematics, ...

---

# Table of Contents

- [Machine Learning](#machine-learning)
  - [Deep Learning Framework](#deep-learning-framework)
  - [Machine Learning Framework](#machine-learning-framework)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Reinforcement Learning](#reinforcement-learning)
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
  - [Mac](#mac)
  - [Cross-Platform](#cross-platform)

---

# Machine Learning

## Deep Learning Framework

* [PyTorch](https://github.com/pytorch/pytorch) - An open source deep learning framework by Facebook, with GPU and dynamic graph support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS*
  * Language API: *Python, C++, Java*
  * <details><summary>Related tools (click to expand):</summary>

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
  * <details><summary>Related tools (click to expand):</summary>

    * [TensorBoard](https://github.com/tensorflow/tensorboard) - TensorFlow's Visualization Toolkit
    * [TensorFlow Serving](https://github.com/tensorflow/serving) - A flexible, high-performance serving system for machine learning models based on TensorFlow
    * [TFDS](https://github.com/tensorflow/datasets) - A collection of datasets ready to use with TensorFlow and Jax
    * [TensorFlow Model Garden](https://github.com/tensorflow/models) - Models and examples built with TensorFlow
    * [TensorFlow.js](https://github.com/tensorflow/tfjs) - A WebGL accelerated JavaScript library for training and deploying ML models based on TensorFlow
    * [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) **(no longer maintained)** - Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research
    * [Trax](https://github.com/google/trax) **(successor of Tensor2Tensor)** - Deep Learning with Clear Code and Speed
    * [tf_numpy](https://www.tensorflow.org/guide/tf_numpy) - A subset of the NumPy API implemented in TensorFlow
    </details>

* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - An open source deep learning framework by Baidu, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Android, iOS, Web*
  * Language API: *Python, C++, Java, JavaScript*
  * <details><summary>Related tools (click to expand):</summary>

    * [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Multilingual OCR toolkits based on PaddlePaddle
    * [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) - Object detection toolkit based on PaddlePaddle
    * [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) - Image segmentation toolkit based on PaddlePaddle
    * [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) - Visual classification and recognition toolkit based on PaddlePaddle
    * [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) - Generative Adversarial Networks toolkit based on PaddlePaddle
    * [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) - Video understanding toolkit based on PaddlePaddle
    * [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) - Recommendation algorithm based on PaddlePaddle
    * [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) - Natural language processing toolkit based on PaddlePaddle
    * [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) - Speech Recognition/Translation toolkit based on PaddlePaddle
    * [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) - Pre-trained models toolkit based on PaddlePaddle
    * [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) - Multi-platform high performance deep learning inference engine for PaddlePaddle
    * [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) - An open source deep learning framework running in the browser based on PaddlePaddle
    </details>

* [MXNet](https://github.com/apache/incubator-mxnet) - An open source deep learning framework by Apache, with GPU support.
  * Supported platform: *Linux, Windows, MacOS, Raspberry Pi*
  * Language API: *Python, C++, R, Julia, Scala, Go, Javascript*

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

## Natural Language Processing

* [HuggingFace Transformers](https://github.com/huggingface/transformers) - A high-level machine learning library for text, images and audio data, with support for Pytorch, TensorFlow and JAX.

* [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - A high-performance library for text vocabularies and tokenizers.

* [NLTK](https://github.com/nltk/nltk) - An open source natural language processing library in Python.

* [spaCy](https://github.com/explosion/spaCy) - Industrial-strength Natural Language Processing (NLP) in Python.

* [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) - An open source natural language processing library for Apache Spark.

* [Flair](https://github.com/flairNLP/flair) - An open source natural language processing library, based on PyTorch.

* [Fairseq](https://github.com/facebookresearch/fairseq) - A sequence-to-sequence toolkit by Facebook, based on PyTorch.

* [ParlAI](https://github.com/facebookresearch/ParlAI) - A python framework for sharing, training and testing dialogue models from open-domain chitchat, based on PyTorch.

* [Stanza](https://github.com/stanfordnlp/stanza) - An open source natural language processing library by Stanford NLP Group, based on PyTorch.

* [NeMo](https://github.com/NVIDIA/NeMo) - A toolkit for conversational AI, based on PyTorch.

* [Haystack](https://github.com/deepset-ai/haystack) - A high-level natural language processing library for deployment and production, based on PyTorch and HuggingFace Transformers.

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) - The PyTorch version of the OpenNMT project, an open-source neural machine translation framework.

* [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) - The TensorFlow version of the OpenNMT project, an open-source neural machine translation framework.

* [AllenNLP](https://github.com/allenai/allennlp) **(not actively updated)** - An open source natural language processing library, based on PyTorch.

* [Gensim](https://github.com/RaRe-Technologies/gensim) - A Python library for topic modelling, document indexing and similarity retrieval with large corpora, based on NumPy and SciPy.

* [Rasa](https://github.com/RasaHQ/rasa) - Open source machine learning framework to automate text- and voice-based conversations.

* [SentencePiece](https://github.com/google/sentencepiece) - Unsupervised text tokenizer for Neural Network-based text generation.

* [subword-nmt](https://github.com/rsennrich/subword-nmt) - Unsupervised Word Segmentation for Neural Machine Translation and Text Generation.

* [fastText](https://github.com/facebookresearch/fastText) **(not actively updated)** - A library for efficient learning of word representations and sentence classification.

* [TextBlob](https://github.com/sloria/TextBlob) **(not actively updated)** - A Python library for processing textual data.

## Reinforcement Learning

* [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) - A fast, flexible, online, and active learning solution for solving complex interactive machine learning problems.

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

* [Towhee](https://github.com/towhee-io/towhee) - Data processing pipelines for neural networks.

* [pandas-profiling](https://github.com/ydataai/pandas-profiling) - Create HTML data profiling reports for pandas DataFrame.

* [FiftyOne](https://github.com/voxel51/fiftyone) - An open-source tool for building high-quality datasets and computer vision models.

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

## Machine Learning Tutorials

* [labml.ai](https://nn.labml.ai/) - A collection of PyTorch implementations of neural networks and related algorithms, which are documented with explanations and rendered as side-by-side formatted notes.

  ![labml.ai](/imgs/labml.ai.png)

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

## Mac

* [Scroll-Reverser](https://github.com/pilotmoon/Scroll-Reverser) - Reverses the direction of macOS scrolling, with independent settings for trackpads and mice.

## Cross-Platform

* [LANDrop](https://github.com/LANDrop/LANDrop) - A cross-platform tool that you can use to conveniently transfer photos, videos, and other types of files to other devices on the same local network.
