# Getting Started

Initialize your environment with the uv package:

```
uv sync
```

Next, you need to generate some trees. This basically hard codes some sampling
parameters and we're doing a simple birth death process.

```
python ./generation/generate_trees.py
```

Once that's done, check the `simulate.ipynb` notebook for how to actually
encode the trees into a vector representation. We'll clean that up later.