# For now:
Data used can be found on the Google Drive. Download the zipped file you find there. <br>
<br>After unzipping it, you should have 2 folders : `2019` and `2022`. Take those 2 and put them in the `current_data` folder. <br>
<br>Afterward, go to `prepare_datasets` folder and run everything from there except `get_mean_std_for_normalization.py`. These will split the data as needed for all our cases to make direct comparisons with the paper.
<br><br>All code we'll write should be in the `Scripts` folder. The progress we have done so far can be found in files `train_shallow.py` and `train_deep.py`.
<br><br> **Some important things I've just seen in this implementation**:
- We need to change the batch size to `128`. I did on `20`, but in order to compare with the paper, we need batch size of `128`.
- They use an initial learning rate of `0.001`. which I also used, however they also use it in a combination with Adam + L2 regularization of `0.0001`. This is the case for ResNet. For VGG11, they used an initial learning rate of `0.0001`, I assume again with Adam and the L2, which I didn't. So, we need to change these.
- There are 2 types of comparisons we can do directly with the paper.
  - Test and train on 2019 with split `75/25` + Test and train on 2022 with split `62/38`. For these, only one run with `5-fold` cross validation, `100` epochs. Final results to be reported are F1 and Accuracy.
  - Train entirely on 2019, test entirely on 2022 and vice versa. Final results to be reported are F1 and Accuracy over `10` runs, `100` epochs each. Here without `5-fold` cross validation. 

In this current implementation, I didn't really pay attention to these details, and simply trained in both Shallow and Deep Transfer Learning, both ResNet50 and VGG11 with just learning rate of `0.001`, batch size of `20`, epochs = `15` and `100`, only a single run. However, as you can tell already, we can't do any comparisons with these.

<br> So, the next tasks until the meeting we have on Wednesday are the following:
1. Modify data splits like I mentioned above ; 
2. Change the batch size to `128` ;
3. Modify the learning rates like I mentioned above ;
4. Add 5-fold cross validation in the case when using `2019` as train/test and also for `2022` as train/test. Run ResNet50 and VGG11 once in both `train_shallow.py` and `train_deep.py`, `100` epochs;
5. Train entirely on `2019`, test entirely on `2022` and vice versa. `100` epochs, `10` runs. Results (Accuracy, F1) averaged over the `10` runs. We'll do it only on `train_deep.py` as they also state in the paper that this gives better results ;
6. Only after we have all these and check if we are near their results, we start implementing the hybrid configuration + train/test on same configuration `100 ` epochs, `10` runs, averaged results. 

**Please keep in mind, whenever you push things, don't push anything in the data folders, as all of us will have the data locally. Same goes for model saves.** 

# Updates
`1.` is done. All the data splits files can be found under `prepare_datasets` folder. More precisely:
- `2019train_2022test.py` gets all 2019 data for training, all 2022 data for testing and puts them in the `DeepLearning_PlantDiseases-master/Scripts/PlantVillage_1_2019train_2022test` folder ;
- `2022train_2019test.py` gets all 2022 data for training, all 2019 data for testing and puts them in the `DeepLearning_PlantDiseases-master/Scripts/PlantVillage_2_2022train_2019test` folder ;
- `2019train_and_test.py` splits the 2019 data into training and testing `75/25` like in the paper and puts them in the `DeepLearning_PlantDiseases-master/Scripts/PlantVillage_2019` folder ;
- `2022train_and_test.py` splits the 2022 data into training and testing `62/38` like in the paper and puts them in the `DeepLearning_PlantDiseases-master/Scripts/PlantVillage_2022` folder.

`2.` is also done. Just changed `batch_size` from `20` to `128`. <br>
`3.` also done. Can be seen in the `train_shallow.py` and `train_deep.py` under the `fine_tune_model` method. <br>
`5.` code is done. What's left to do here is to actually run them for 10 runs, each with 100 epochs, both models. `train_deep.py` is already set up for this and ready to run. <br>

If anyone could start working on `4.` and add the 5-fold cross validation based on the codes from `train_shallow.py` and `train_deep.py`. As I mentioned above, here we'll need to run only once for 100 epochs, both networks in both shallow and deep configurations. On the `PlantVillage_2019` and `PlantVillage_2022` data folders.<br>
Also, a thing I probably forgot to mention, data augmentation is applied with the same techniques they use in the paper. One of them is normalization and these have to have specific values. In `get_mean_std_for_normalization.py` I've computed them for all cases, simply remember to put the right ones with accord to the dataset you are training in the `train_shallow.py` and `train_deep.py` under `load_data()` method:
* 2019 train entirely:
    * mean: `0.7553`, `0.3109`, `0.1059`
    * std: `0.1774`, `0.1262`, `0.0863`
* 2022 train entirely:
    * mean: `0.7083`, `0.2776`, `0.0762`
    * std: `0.1704`, `0.1296`, `0.0815`
* 2019 split:
    * mean: `0.7551`, `0.3113`, `0.1063`
    * std: `0.1780`, `0.1264`, `0.0868`
* 2022 split:
    * mean: `0.7074`, `0.2772`, `0.0759`
    * std: `0.1713`, `0.1298`, `0.0812`
