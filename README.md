# AI System for Breast Cancer Diagnosis

## Overview

This university project, conducted in 2024, is a collaboration between our team and the Anti-Cancer Center of Sidi Bel Abbes (CAC). The objective of this project was to develop an AI system capable of diagnosing breast cancer through the classification of gigapixel hematoxylin and eosin histopathological images (whole slide images). We used the [BRCAS](https://www.bracs.icar.cnr.it/) dataset for this purpose. The process mainly consists of two stages:

- **Feature Extraction**: This involves processing the high-resolution whole slide images to generate a more compact representation that can be processed by deep learning models.

- **Models Training**: This step involves training the actual models to classify the WSIs, using the compressed WSIs generated from the previous step as input.

The final model was deployed in a desktop application where a WSI can be selected, then the feature extraction step is applied before passing the result to the model for inference and outputting the probabilities of each class, for more details please refere to : [Report]().


## Members

- Abdelnour Fellah: [ab.fellah@esi-sba.dz](mailto:ab.fellah@esi-sba.dz)
- Abderrahmane Benounene: [a.benounene@esi-sba.dz](mailto:a.benounene@esi-sba.dz)
- Adel Abdelkader Mokadem: [aa.mokadem@esi-sba.dz](mailto:aa.mokadem@esi-sba.dz)
- Meriem Mekki: [me.mekki@esi-sba.dz](mailto:me.mekki@esi-sba.dz)
- Yacine Lazreg Benyamina: [yl.benyamina@esi-sba.dz](mailto:yl.benyamina@esi-sba.dz)