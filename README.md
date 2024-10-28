# Human Protein Multi-Label Image Classification - Transfer Learning & Regularization

## Introduction

This report explores the multi-label image classification task of identifying human protein structures using a Convolutional Neural Network (CNN) model with transfer learning and regularization. Human proteins exhibit diverse structures with unique biological functions, and accurately classifying them is critical for biological research and medical applications. Utilizing the Human Protein Atlas dataset, the goal is to implement a deep learning model capable of classifying images with multiple protein labels.

Transfer learning is employed using ResNet, a pre-trained CNN architecture known for its effective feature extraction and transferability across image classification tasks. Regularization techniques, data augmentations, and a custom dataset loader are used to enhance model performance, addressing the challenges posed by multi-label classification and imbalanced dataset distributions.

## Aim

The primary objective is to develop a multi-label image classification model that can accurately detect various human protein substructures. By leveraging a ResNet-34 model, pre-trained on the ImageNet dataset, the project aims to fine-tune the network using transfer learning and optimize performance through regularization and data augmentation techniques.

## Results

The model training proceeded in two stages:

Freezing Layers and Initial Training: Initially, all layers except the final fully connected layer were frozen to allow the model to learn specific features relevant to protein structures. The model was trained over 5 epochs with a maximum learning rate of 0.01. The F1 score, a standard measure in multi-label classification, improved from 0.2930 to 0.6387, indicating a substantial gain in classification performance.

Key metrics after 5 epochs:

Final Training Loss: 0.2352
Validation Loss: 0.2226
Validation F1 Score: 0.6387
Unfreezing Layers and Further Fine-Tuning: After the initial training, all layers were unfrozen, and the model was fine-tuned for additional epochs with a lower learning rate of 0.001 to avoid overfitting. This stage allowed the entire network to adjust and optimize its weights for the specific task of human protein classification.

Performance improved further with reductions in validation loss and increases in the validation F1 score, demonstrating that the model effectively learned the intricate patterns necessary for multi-label protein classification.

## Conclusion

The Human Protein Multi-Label Image Classification task demonstrated the efficacy of transfer learning and regularization in adapting pre-trained models to new, complex datasets. By fine-tuning a ResNet-34 model, we achieved significant improvements in classifying multiple protein structures within images. The model's F1 score, which started low, increased considerably through successive stages of layer freezing and unfreezing, affirming the benefits of a two-stage training process.

Future work may explore alternative CNN architectures or additional augmentations for further improvement. This work underscores the potential of transfer learning in biological image classification, where labeled data may be limited but task complexity is high.
