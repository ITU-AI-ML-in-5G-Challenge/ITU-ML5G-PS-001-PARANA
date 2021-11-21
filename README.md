<h1> PARANA-GNNChallenge: 1st place solution of Graph Neural Networking Challenge 2021 </h1>

<h3>Team:</h3> PaRaNA (<b>Pa</b>ttern <b>R</b>ecognition <b>a</b>nd <b>N</b>etwork <b>A</b>nalysis) <br/>
--> This is the PaRaNA research group of the Institute of Science and Technology of the <b>Federal University of São Paulo</b>, led by Dr. Lilian Berton.

<h3>Members:</h3> Bruno Klaus de Aquino Afonso </h2><br/>
--> A solo effort, it was pretty useful as I needed some more experience with GNN implementation for my PhD. 


# Awards Ceremony:
https://youtu.be/RmiLVl8yBZs?t=634
[![IMAGE ALT TEXT HERE](http://i3.ytimg.com/vi/RmiLVl8yBZs/maxresdefault.jpg)](https://youtu.be/RmiLVl8yBZs?t=634)

# The problem, summarized
Graph Neural Networks (GNN) have produced groundbreaking applications in many fields where data is fundamentally structured as graphs (e.g., chemistry, physics, biology, recommender systems). In the field of computer networks, this new type of neural networks is being rapidly adopted for a wide variety of use cases [1], particularly for those involving complex graphs (e.g., performance modeling, routing optimization, resource allocation in wireless networks).

The Graph Neural Networking challenge 2021 brings a fundamental limitation of existing GNNs: their lack of generalization capability to larger graphs. In order to achieve production-ready GNN-based solutions, we need models that can be trained in network testbeds of limited size (e.g., at the vendor’s networking lab), and then be directly ready to operate with guarantees in real customer networks, which are often much larger in number of nodes. In this challenge, participants are asked to design GNN-based models that can be trained on small network scenarios (up to 50 nodes), and after that scale successfully to larger networks not seen before, up to 300 nodes. Solutions with better scalability properties will be the winners.


# Our approach, summarized


|                            | **Fast Enough?**   | **Has top tier performance?**| **Generalizes to larger graphs?**
|----------------------------|--------------------|------------------------------------|----------------------------------------|
| *Analytical*       | :heavy_check_mark: | :x:                           | :heavy_check_mark:   (~12% MAPE)                         |
| *Packet simulators* | :x: (prohibited)   | :heavy_check_mark:            | :heavy_check_mark:                                 |
| *RouteNet*          | :heavy_check_mark: | :heavy_check_mark:            | :x:    (>300% MAPE)                              |
| *Proposed solution* | :heavy_check_mark: | :heavy_check_mark:            | :heavy_check_mark:     (1.27% MAPE)                           |

To understand the solution detailed in this report, it is helpful to look at the previous approaches:
- Packet simulators were not allowed in the competition in principle due to excessive running times
- Analytical approaches generalize well and run fast, but they do not offer competitive performance
- RouteNet is still fast and more performant than analytical approaches, but fails to generalize to larger graphs. 

For our proposed solution, we **extract invariant features from the analytical approach**, and feed them to a GNN. This way, we can **maintain generalization while outperforming the purely analytical** approach. This is done  using a **modified RouteNet architecture**, making use of baseline predictions as features and incorporating Graph Attention (GAT) and Graph Convolutional Gated Recurrent Unit (GconvGRU) layers. 

Initially, we constructed a big model using the most available resources (*model 1*). However, later experiments showed  that we can have a much, much smaller model (*model 2*) and still achieve the same result. The final prediction of the challenge was the average of both models, which yielded a slight improvement. 


|                               | **Val. 1** | **Val. 2** | **Val. 3** | **Test** |
|-------------------------------|-----------------|-----------------|-----------------|---------------|
| **RouteNet** | ---             | ---             | ---             | >300.0        |
| **Baseline**             | 12.10           | 9.18            | 9.51             | ?              |
| **Model 1 w/o baseline** | ---             | ---             | ---             | 22.58         |
| **Model 1 (Sep 22nd)**   | **2.71**            | 1.33            | **1.65**            | 1.45          |
| **Model 2 (Sep 29th)**   | 3.61            | **1.17**            | 1.55            | 1.45          |
| **(Model 1+ Model 2)/2** | ---             | ---             | ---             | **1.27**          |

'---' means did not evaluate on (yet).


<h2> This repository includes </h2>

* A report detailing the solution: https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-PARANA/blob/668daa429e46928801affe4cc1a4a3136280e32f/GNNET_2021_report.pdf

* The presentation slides used for the Graph Neural Networking award ceremony:  https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-PARANA/blob/668daa429e46928801affe4cc1a4a3136280e32f/Slides-PS-001-ML5G-PARANA.pdf 
* Scripts for generating a converted dataset with <it>.pt</it> files.
* 3 Jupyter notebooks: one that imports the scripts and generates the dataset, and one for each model used.
* Model weights (<it>22_setembro_modelo.pt,29_setembro_modelo.pt</it>) that are loaded inside the notebooks.

<h2> Requirements and instructions </h2>
Optionally, you may want to first read the PDF file that explains our model. The code requires a GPU, we can only guarantee that 10 GB VRAM (such as an RTX 3080) and 16GB of RAM is sufficient to run everything, though in practice it may be a bit less than that. Our framework uses Pytorch 1.8.1. w/ Pytorch Geometric 1.7.0. (+ dependencies, see here how to install https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

Full requirements are found in the <b>requirements.txt</b> files. With those installed, you should be able to run the 3 jupyter notebooks. On top of those, you need to install <b> jupyter notebook </b> with <b>ipywidgets</b>  

`pip install jupyterlab`
`pip install ipywidgets`

The 1st notebook is `1) Download dataset, create .pt files.ipynb`. If you have ipywidgets, you should run the cell and see the checkboxes with the options to download the dataset, extract tarfiles, and create the converted version of this dataset for training, validation and test sets. You can set the number of processes when creating files (more processes multiplies the RAM used and reduces the time it takes to complete). After it finishes (may take an hour or two), you should now have the corresponding folders on `./dataset` <b>(OBS: see the specification on top of the 1st notebook and count the files to ensure nothing's gone wrong)</b>

Next, we have the 2 notebooks for running our models. They are `Model - September 22.ipynb`, `Model - September 29th-Copy1.ipynb`. From there you can use the notebook cells to train or test the pre-trained model with saved weights. Training the Sep 22nd model takes about 8x longer than the Sep 29th, and they yielded almost the same performance on the test set. Due to problems on the final day of the competition, we opted to average Model 1 and Model 2 instead of trying out the average of multiple runs of the more efficient Model 2.

Lastly, you can use the <b>last cell</b> of `Model - September 29th-Copy1.ipynb` to create the average prediction.
