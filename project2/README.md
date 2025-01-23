# Individual Project 2
# Classification with Deep Learning
#### Due Date
* Updated to Tuesday Mar 18, 2025 (23:59), 


#### Total Points
* 100 (One Hundred)

## Goal
In this project, you will be asked to finish a sequence classification task using deep learning. A trajectory data set with five taxi drivers' daily driving trajectories in 6 months will be provided. The primary objective is to predict which driver each 100-step sub-trajectory, extracted from the daily trajectories, belongs to. To evaluate your model, it will be tested on a separate set of data for five additional days (5 CSV files, same format as the training data), using the same preprocessing steps to ensure consistent data handling. This approach ensures consistency in data preparation across training and testing phases, allowing the model to accurately attribute each sub-trajectory to the correct driver. 

<img src="https://github.com/UrbanIntelligence/CS586-DS504-Spring2024/blob/master/project2/pic/classification.png" width="100%">

## Guidelines
* Implement required functions in model.py, train.py, and test.py.
* The file extract_feature.py provides a way to preprocess the data before feeding the data to the neural network, you can customize it to get more features but maintain 100 steps as standard for fair comparison.
* This project should be completed in Python 3. Pytorch is highly recommended.

## How to run :
training model:
* `$ python main.py train`

testing model:
* `$ python main.py test --test_dir = `

## Current Leaderboard
| rank | Name | Accuracy |
|---|---|---|
|**1**   | --- | --- |
|**2**   | --- | --- |
|**3**   | --- | --- |


## Deliverables & Grading

### Submission Required Files (Code Files and Trained Model, please don't include the training data)

#### Code Files
Include all necessary Python scripts for your project:
- `main.py`: Entry point for the project
- `train.py`: Training logic
- `test.py`: Testing logic
- `extract_feature.py`: Data preprocessing logic
- `model.py`: Model definition

#### Trained Model
Include the trained model file (e.g., `best_model.pt`).  
- **Note**: The filename must match the reference in your `test.py` script.

#### Requirements
Include a requirements.txt file that lists any additional Python libraries and dependencies your code requires beyond those already provided.

#### Output Format

- Your `test.py` script must print the **accuracy** in the following format:
    ```plaintext
    Accuracy=0.6123
    ```
  Do not multiply the accuracy by 100; percentages are not allowed.

- **Important Notes**:
  1. **Ensure this is the only critical information printed** for evaluation purposes.
  2. **Remove or redirect debug messages** to avoid interfering with the autograder’s ability to parse the output.
  3. The accuracy value must be a floating-point number formatted to **4 decimal places**.
  4. The autograder will evaluate your submission on a CPU-only machine. Please handle it in your code.
     
#### Grading

* PDF Report (50%) [template](https://www.acm.org/binaries/content/assets/publications/taps/acm_submission_template.docx)
    * proposal
    * methodology
    * empirical results and evaluation
    * conclusion
    
* Python Code (50%)
    * Code is required to avoid plagiarism.
    * Implement functions in model.py, train.py, and test.py.

    * Evaluation criteria.
      | Percentage | Accuracy |
      |---|---|
      | 100 | 0.65 |
      | 90 | 0.60 |
      | 80 | 0.55 |
      | 70 | 0.50|
      | 60 | 0.45 |
* Grading:
  * Total (100):
    * Code (50) + Report (50)

  * Code (50):
    * accuracy >= 0.65: 50
    * accuracy >= 0.60: 45
    * accuracy >= 0.55: 40
    * accuracy >= 0.50: 35
    * accuracy >= 0.45: 30

  * Report (50):
    1. Introduction & Proposal (5)
    2. Methodology (20):
        a. Data processing (5)
        b. Feature generation (5)
        c. Network structure (5)
        d. Training & validation process (5)
    3. Evaluation & Results (20):
        a. Training & validation results (10)
        b. Performance comparing to your baselines (maybe different network structure) (5)
        c. Hyperparameter (learning rate, dropout, activation) (5)
    4. Conclusion (5)

  * Bonus (5):
   
     5 bonus points for the top 3 on the leaderboard.

## Project Guidelines

#### Dataset Description
| plate | longitute | latitude | time | status |
|---|---|---|---|---|
|4    |114.10437    |22.573433    |2016-07-02 0:08:45    |1|
|1    |114.179665    |22.558701    |2016-07-02 0:08:52    |1|
|0    |114.120682    |22.543751    |2016-07-02 0:08:51    |0|
|3    |113.93055    |22.545834    |2016-07-02 0:08:55    |0|
|4    |114.102051    |22.571966    |2016-07-02 0:09:01    |1|
|0    |114.12072    |22.543716    |2016-07-02 0:09:01    |0|


Above is an example of what the data looks like. In the data/ folder, each .csv file is trajectories for 5 drivers on the same day. Each trajectory step is detailed with features such as longitude, latitude, time, and status. Data can be found at [Google Drive](https://drive.google.com/open?id=1xfyxupoE1C5z7w1Bn5oPRcLgtfon6xeT)
#### Feature Description 
* **Plate**: Plate means the taxi's plate. In this project, we change them to 0~5 to keep anonymity. The same plate means the same driver, so this is the target label for the classification. 
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Time**: Timestamp of the record.
* **Status**: 1 means the taxi is occupied and 0 means a vacant taxi.

#### Problem Definition
Given a full-day trajectory of a taxi, you need to extract the sub-trajectories of each 100 steps and predict which taxi driver it belongs to. 

#### Evaluation 
Your model will be evaluated using 100-step sub-trajectories extracted from five days of comprehensive daily trajectories. These test trajectories are not included in the data/ folder provided for training. This ensures a fair assessment of your model's ability to generalize to unseen data.

## Some Tips

* From project 2, you will need some GPU resources to train deep learning models. There are three resources below you can use:
* Open source GPU
   * [WPI Turing](https://arc.wpi.edu/cluster-documentation/build/html/index.html)
   * now you all have access to WPI Turing GPU cluster with your WPI accounts. Follow the [instructions](https://github.com/UrbanIntelligence/CS586-DS504-Spring2025/blob/main/project2/Turing_Setup_Instructions.pdf) to set up the environments.
   * [Google Cloud](https://cloud.google.com/free)
   * [Google CoLab](https://colab.research.google.com/)
* Anaconda and virtual environment set tup
   * [Download and install anaconda](https://www.anaconda.com/download)
   * [Create a virtual environment with commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
* Deep learning package
   * install pytorch: See [install pytorch](https://pytorch.org/get-started/locally/), pip install torch torchvision torchaudio
* **Keywords**. 
   * If you are wondering where to start, you can try to search "sequence classification", "sequence to sequence" or "sequence embedding" in Google or Github, this might provide you with some insights.
   

