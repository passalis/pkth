# Heterogeneous Knowledge Distillation using Information Flow Modeling

In this repository we provide an implementation of the Heterogeneous Knowledge Distillation approach using Information Flow Modeling, as described in [our paper](TODO), which is capable of transferring the knowledge between DL models by matching the information flow between multiple layers.

To reproduce the results reported in our paper:
1. Train and evaluate the baselines model ([exp0_train_base.py](cifar10/exp0_train_base.py))
2. Train and evaluate the auxiliary model ([exp1_train_aux.py](cifar10/exp1_train_aux.py))
3. Use the proposed method to transfer the knowledge to the student ([exp2_proposed.py](cifar10/exp2_proposed.py))
4. Print the evaluation results ([exp9_print_results.py](cifar/exp9_print_results.py))

Note that a pretrained Resnet-18 teacher model is also provided, along with the trained students. So you can directly use/evaluate the trained models and/or print the evaluation results.

If you use this code in your work please cite the following paper:

<pre>
@InProceedings{pkt_eccv,
author = {Passalis, Nikolaos and Tzelepi, Maria and Tefas, Anastasios},
title = {Heterogeneous Knowledge Distillation using Information Flow Modeling},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year = {2020}
}
</pre>

  This work was supported by the European Union's Horizon 2020 Research and Innovation Program ([https://opendr.eu](OpenDR Project)) under Grant 871449. This publication reflects the authors' views only. The European Commission is not responsible for any use that may be made of the information it contains.
  
<center>
<img src="https://opendr.csd.auth.gr/wp-content/uploads/2019/12/Flag_of_Europe-300x200.png" height="50px" />
</center>

