## Submission Examples

These commands can be used to produce the submission files for DeepTrust classifier.

Download the training dataset, the Track 1 datasets and their pre-extracted features from the [ELSA benchmarks website](https://benchmarks.elsa-ai.eu/?ch=6&com=downloads) inside the `data` directory.

It is recommended to create a new environment. In this example we use conda (it might be required to append `android-detectors/src` directory to the python path before launching the script).

**Reproduce results with trained model**

The files required for it are in:
https://drive.google.com/drive/folders/1MzppCM60UBRjTAZ5jBm32Pfo0if21YmX?usp=sharing. Download them and place them in the `pretrained` directory. 
These are:
- `deeptrust_classifier.pkl`: InspectRF pickle file
- `deeptrust_classifier.pth`: TrustNet and GuardNet weights
- `deeptrust_vectorizer.pkl`: DeepTrust vectorizer
- `guardnet_vectorizer.pkl`: GuardNet vectorizer
- `trustnet_vectorizer.pkl`: TrustNet vectorizer
```bash
conda create -n android python=3.9
conda activate android
pip install -r android-detectors/requirements.txt
pip install -r track_1/attack_requirements.txt
pip install -r track_1/deeptrust_requirements.txt
export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 1 --method_name deeptrust
```

**Reproduce results training from scratch**
```bash
conda create -n android python=3.9
conda activate android
pip install -r android-detectors/requirements.txt
pip install -r track_1/attack_requirements.txt
pip install -r track_1/deeptrust_requirements.txt
export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
python main.py --clf_loader_path android-detectors/src/loaders/trust_mlp_loader.py --track 1 --method_name trust_mlp
python main.py --clf_loader_path android-detectors/src/loaders/guard_mlp_loader.py --track 1 --method_name guard_mlp
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 1 --method_name deeptrust
```

Once the deeptrust is trained, you can delete the `trustnet_classifier.pkl` and `guardnet_classifier.pkl` 
files from the `pretrained` directory as their weights are already saved in the `deeptrust_classifier.pth` file.
