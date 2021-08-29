EEH_G2T

An Edge-Enhanced Hierarchical Graph-to-Tree Network for Math Word Problem Solving

## Requirement

- Python 3.6
- Pytorch 1.8.0
- numpy
- nltk
- stanfordcorenlp
- matplotlib

# Train the model.
python run_seq2tree.py

# Evaluate the model.
python evaluate.py

#Structure

├── README.md                   // help

├── data                        // datasets

│   ├── mawps					// MAWPS dataset

│   │   └── MAWPS.json 			// MAWPS dataset

│   └── Math_23K.json           // Math23K dataset	

├── hownet						// external knowledge base HowNet

│   └── cilin.txt           	// external knowledge base cilin

├── models                      // Saved Models

├── output                      // Test data output

│ 
├── pre_data.py 				// data process

├── masked_cross_entropy.py		// cross_entropy function

├── expressions_transfer.py		// expression process

├── models.py					// EEH_G2T's main model structure

├── run_seq2tree.py				// train the model (Math23K default) 

├── parameter.py				// parameters setting (change dataset="mawps" for MAWPS training) 

├── evaluate.py 				// evaluate the model

└── dependency_generate.py 		// stanford dependency tree
