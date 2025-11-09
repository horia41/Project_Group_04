# For now:
Data used can be found on the Google Drive. Download the zipped file you find there. <br>
<br>After unzipping it, you should have 2 folders : `2019` and `2022`. Take those 2 and put them in the `current_data` folder. <br>
<br>Afterward, simply run `prepare_dataset.py` and this will split the data into `train`, `val`, `test`. Then, `train` and `val` will be with an 80/20 split for the `2019` dataset and `test` is `2022` entirely.
<br><br>All code we'll write should be in the `Scripts` folder. The progress we have done so far can be found in files `train_shallow.py` and `train_deep.py`.
<br><br> **Some important things I've just seen in this implementation**:
- We need to change the batch size to `128`. I did on `20`, but in order to compare with the paper, we need batch size of `128`.
- They use an initial learning rate of `0.001`. which I also used, however they also use it in a combination with Adam + L2 regularization of `0.0001`. This is the case for ResNet. For VGG11, they used an initial learning rate of `0.0001`, I assume again with Adam and the L2, which I didn't. So, we need to change these.
- There are 2 types of comparisons we can do directly with the paper.
  - Test and train on 2019 with split `75/25` + Test and train on 2022 with split `62/38`. For these, only one run with `5-fold` cross validation, `100` epochs.
  - Train entirely on 2019, test entirely on 2022 and vice versa. Final results to be reported are F1 and Accuracy over `10` runs.

In this current implementation, I didn't really pay attention to these details, and simply trained in both Shallow and Deep Transfer Learning, both ResNet50 and VGG11 with just learning rate of `0.001`, batch size of `20`, epochs = `15` and `100`, only a single run. However, as you can tell already, we can't do any comparisons with these.

<br><br> All the abovementioned things indicate that we need to retrain. So, the next tasks until the meeting we have on Wednesday are the following:
1. Modify data splits like I mentioned above ; 
2. Change the batch size to `128` ;
3. Modify the learning rates like I mentioned above ;
4. Add 5-fold cross validation in the case when using `2019` as train/test and also for `2022` as train/test. Run ResNet50 and VGG11 once in both `train_shallow.py` and `train_deep.py`, `100` epochs;
5. Train entirely on `2019`, test entirely on `2022` and vice versa. Here for each case, `100` epochs, `10` runs. Results (Accuracy, F1) averaged over the `10` runs ;
6. Only after we have all these and check if we are near their results, we start implementing the hybrid configuration + train/test on same configuration `100 ` epochs, `10` runs, averaged results. 

**Please keep in mind, whenever you push things, don't push anything in the data folders, as all of us will have the data locally. Same goes for model saves. Simply just code files.** 
