General Idea. 

Use a model trained on either the spectogram of an EMG in a self supervised manner as a foundational model.
This will embed the spectogram into a lower dimensional space. 

Then for the downstream task of keystroke prediction, we can freeze the 
weights of the foundational model and train smaller MLP or some form of classifier on top of it. Hopefully
this will allow us to train the downstream task with less data. Furthermore for the paper, we can sell this as a 
foundational model that can be used for other tasks as well.

TODOs and potential research directions:
- [x] Move the loss fns and spectogram transforms to seperate classes 
- [ ] Finish two trainers, one for a foundational model and one for a downstream task, these will be based on the pytorch lightning trainer
- [ ] Use Hydra for config management (potentially)
- [ ] Experiment with different foundational architectures models, for example:
    - [ ] Autencoder (CNN)
    - [ ] VAE (CNN) (Daniel)
    - [ ] ViTMAE (Lawrence)
    - [ ] VLLM (Manik)
- [ ] Experiment with different loss functions and training stuff, for example:
    - [ ] Channel wise foundational model (compress each channel separately)
    - [ ] Handwise foundational model (compress each hand separately)
    - [ ] Moving beyond MSE for Spectogram reconstruction loss, for example VGG loss
- [ ] Agree on a format between the foundational model and the downstream task
- [ ] Incoporate more data, for example emg2pose dataset and more downstream tasks (This can be saved for post paper)
