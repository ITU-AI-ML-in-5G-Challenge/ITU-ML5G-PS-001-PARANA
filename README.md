<h1> PARANA-GNNChallenge: 1st place solution of Graph Neural Networking Challenge 2021 </h1>

<h3>Team:</h3> PaRaNA (<b>Pa</b>ttern <b>R</b>ecognition <b>a</b>nd <b>N</b>etwork <b>A</b>nalysis) <br/>
--> This is the PaRaNA research group of the Institute of Science and Technology of the <b>Federal University of São Paulo</b>, led by Dr. Lilian Berton.

<h3>Members:</h3> Bruno Klaus de Aquino Afonso </h2><br/>
--> A solo effort, it was pretty useful as I needed some more experience with GNN implementation for my PhD. 

<h2> This repository includes </h2>

* A report <b>GNNET_2021_report.pdf</b> detailing the solution
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
