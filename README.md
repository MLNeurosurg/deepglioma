# DeepGlioma: AI-based molecular classification of diffuse gliomas using rapid, label-free optical imaging

[**arXiv**](https://arxiv.org/abs/2206.08439) / [**interactive website**](https://deepglioma.mlins.org) / [**MLiNS Lab**](https://mlins.org)

Code repository to perform diffuse glioma molecular classification using stimulated Raman histology (SRH) and deep neural networks. 


### Abstract
Molecular classification has transformed the management of brain tumors by enabling more accurate prognostication and personalized treatment. Access to timely molecular diagnostic testing for brain tumor patients is limited, complicating surgical and adjuvant treatment and obstructing clinical trial enrollment. We developed a rapid (<90 seconds), AI-based diagnostic screening system that can provide molecular classification of diffuse gliomas and report its use in a prospective, multicenter, international testing cohort of diffuse glioma patients (N = 153). By combining stimulated Raman histology (SRH), a rapid, label-free, non-consumptive, optical imaging method, and deep learning-based image classification, we can predict the molecular features used by the World Health Organization (WHO) to define the adult-type diffuse glioma taxonomy. We developed a multimodal training strategy that uses both SRH images and large-scale, public diffuse glioma genomic data to achieve optimal image-based molecular classification performance. Using this system, called DeepGlioma, we were able to achieve an average molecular genetic classification accuracy of 93.2% and identify the correct diffuse glioma molecular subgroup with 91.5% accuracy. Our results represent how artificial intelligence and optical histology can be used to provide a rapid and scalable alternative to wet lab methods for the molecular diagnosis of brain tumor patients.

### TL;DR
*Tumor images in >> **DeepGlioma** >> Tumor genetics out* (end-to-end: ~2 mins)

© This code is made available for non-commercial academic purposes. Imaging and clinical information for this project was collected with IRB approval (HUM00083059) and is protected under HIPAA. Representative images and predictions can be found at [**deepglioma.mlins.org**](https://deepglioma.mlins.org).

![DeepGlioma Workflow](/figures/Figure_1_workflow-01.png)

# Installation

1. Clone DeepGlioma github repo
    ```console
    git clone git@github.com:MLNeurosurg/deepglioma.git
    ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment:  
    ```console
    conda create -n deepglioma python=3.8
    ```
4. Activate conda environment:  
    ```console
    conda activate deepglioma
    ```
5. Install package and dependencies  
    ```console
    <cd /path/to/repo/dir>
    pip install -e .
    ```
6. Train full DeepGlioma model or SRH visual encoder using contrastive learning
    ```console
    python main.py -c main_config.yaml
    ```
    ```console
    python contrastive -c contrastive_config.yaml
    ```

# Directory organization
- datasets: PyTorch datasets and utilities for model training and inference.
- embedding: train the genetic embedding layer with curated, public genomic data included. Able to train using both GloVe and Word2Vec embeddings. Can train out-of-the-box:
    ```console
    python embedding/main.py -c embedding/main_embed.yaml
    ```
- models: SRH encoders and transformer modules. Code adapted from [**C-Tran**](https://github.com/QData/C-Tran) repository.
- utils: model evaluation and metric implementations.


## License Information
The code is licensed under the MIT License.
See LICENSE for license information and third party notices.