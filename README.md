# DeepGlioma: AI-based molecular classification of diffuse gliomas using rapid, label-free optical imaging

[**Preprint**](https://www.researchsquare.com/article/rs-1930236/v1) / [**Interactive Website**](https://deepglioma.mlins.org) / [**MLiNS Lab**](https://mlins.org)

Code repository for our paper 'AI-based molecular classification of diffuse gliomas using rapid, label-free, optical imaging'. We use deep learning perform diffuse glioma molecular classification using stimulated Raman histology (SRH) and deep neural networks. 


### Abstract
Molecular classification has transformed the management of brain tumors by enabling more accurate prognostication and personalized treatment. Access to timely molecular diagnostic testing for brain tumor patients is limited, complicating surgical and adjuvant treatment and obstructing clinical trial enrollment. We developed a rapid (<90 seconds), AI-based diagnostic screening system that can provide molecular classification of diffuse gliomas and report its use in a prospective, multicenter, international testing cohort of diffuse glioma patients (N = 153). By combining stimulated Raman histology (SRH), a rapid, label-free, non-consumptive, optical imaging method, and deep learning-based image classification, we can predict the molecular features used by the World Health Organization (WHO) to define the adult-type diffuse glioma taxonomy. We developed a multimodal training strategy that uses both SRH images and large-scale, public diffuse glioma genomic data to achieve optimal image-based molecular classification performance. Using this system, called DeepGlioma, we were able to achieve an average molecular genetic classification accuracy of 93.2% and identify the correct diffuse glioma molecular subgroup with 91.5% accuracy. Our results represent how artificial intelligence and optical histology can be used to provide a rapid and scalable alternative to wet lab methods for the molecular diagnosis of brain tumor patients.

### TL;DR
*Image tumor with **SRH** >> tumor images in >> **DeepGlioma** >> Tumor genetics out* (end-to-end: ~2 mins)


# Workflow
![DeepGlioma Workflow](/figures/Figure_1_workflow-01.png)
**Bedside SRH and DeepGlioma workflow**. **a**, A patient with a suspected diffuse
glioma undergoes surgery for tumor biopsy or surgical resection. The SRH imaging system
is portable and imaging takes place in the operating room, performed by a single technician
using simple touch screen instructions. A freshly excised tissue specimen is loaded directly
into a premade microscope slide and inserted into the SRH imager without the need for
tissue processing. Raw SRH images are acquired at two Raman
shifts, 2,845cm-1 and 2,930cm-1, as strips. The time to acquire a 3×3mm2 SRH image is
approximately 90 seconds. Raw optical images can then be colored using a custom hematox-
lyin and eosin (HE) virtual staining method for clinician review. **b**, DeepGlioma is trained
using a multi-modal dataset. First, SRH images are used to train an CNN encoder using
weakly supervised, multi-label contrastive learning for image feature embedding. Second, public diffuse glioma genomic data from TCGA, CGGA, and others are used to train a genetic encoder to learn a genetic embedding
that represents known co-occurrence relationships between genetic mutations. **c**, The SRH and genetic encoders are integrated into a single architecture
using a transformer encoder for multi-label prediction of diffuse glioma molecular diagnos-
tic mutations. We use masked label training to train the transformer encoder. Because our system uses patch-level predictions, spatial heatmaps can be generated for both molecular genetic and molecular subgroup predictions to improve model
interpretability, identify regions of variable confidence, and associate SRH image features
with DeepGlioma predictions. 

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
    python contrastive.py -c contrastive_config.yaml
    ```

# Directory organization
- datasets: PyTorch datasets and utilities for model training and inference.
- embedding: train the genetic embedding layer with curated, public genomic data included. Able to train using both GloVe and Word2Vec embeddings. Can train out-of-the-box:
    ```console
    python embedding/main.py -c embedding/main_embed.yaml
    ```
- models: SRH encoders and transformer modules. Code adapted from [**C-Tran**](https://github.com/QData/C-Tran) repository.
- utils: model evaluation and metric implementations.



# Results
![DeepGlioma Workflow](/figures/Figure_2_results-01.png)
**DeepGlioma molecular classification performance**  **a**, Results from our
prospective multicenter testing cohort of diffuse glioma patients are shown. DeepGlioma
was trained using UM data only (n = 373) and tested on our external medical centers (n
= 153). All results are presented as patient-level predictions. Individual ROC curves for
IDH-1/2 (AUROC 95.9%), 1p9q-codeletion (AUROC 97.7%), and ATRX (AUROC 85.7%)
classification are shown. Our AUROC values were highest for IDH-1/2 and 1p19q-codeletion
prediction. Bar plot inset shows the accuracy, F1 score, and AUROC classification metrics
for each of the mutations. Similar to our cross-validation experiments, ATRX mutation pre-
diction was the most challenging as demonstrated by comparatively lower metric scores.
Individual patient-level molecular genetic prediction probabilities are ordered and displayed.
**b**, Results from the LIOCV experiments. Mean (solid line) and standard deviation (fill color)
ROC curves are shown. Metrics are averaged over external testing centers to determine
the stability of DeepGlioma classification results given different patient populations, clinical
workflows, and SRH imagers. Including additional training data resulted in an increase in
DeepGlioma performance, especially for 1p19q and ATRX classification. **c**, Primary testing
endpoint: comparison of IDH1-R132H IHC versus DeepGlioma for IDH mutational status
detection. DeepGlioma achieved a 94.2% balanced accuracy for the prospective cohort and a
97.0% balanced accuracy for patients 55 years or less. The major performance boost was due
to the +10% increase in prediction sensitivity over IDH1-R132H IHC due to DeepGlioma’s
detection of both canonical and non-canonical IDH mutations. **d**, Secondary testing end-
point: DeepGlioma results for molecular subgrouping according to WHO CNS5 adult-type
diffuse glioma taxonomy. Multiclass classification accuracy for all patients and patients 55
years or less are shown. **e**, UMAP visualization of SRH representations from DeepGlioma.
Small, semi-transparent points are SRH patch representations and large, solid points are
patient representations (i.e. average patch location) from the prospective clinical cohort.
Representations are labeled according to their IDH subgroup and diffuse glioma molecular
subgroup. Our patch contrastive learning encourages the SRH encoder to learn representa-
tions that are uniformily distributed on the unit hypersphere.


© This code is made available for academic purposes. Imaging and clinical information for this project was collected with IRB approval (HUM00083059) and is protected under HIPAA. Representative images and predictions can be found at [**deepglioma.mlins.org**](https://deepglioma.mlins.org).


## License Information
The code is licensed under the MIT License.
See LICENSE for license information and third party notices.