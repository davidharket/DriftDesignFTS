class Config:
    def __init__(self):
        self.output_dir = './output'
        self.experiment_name = 'finetune-codellama'
        self.model_name = 'allenai/code-davinci-002'
        self.learning_rate = 2e-5
        self.num_train_epochs = 5
        self.per_device_train_batch_size = 8
        self.per_device_eval_batch_size = 8
        self.logging_steps = 10
        self.save_steps = 500
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.use_wandb = False
        self.evaluation_metric = 'accuracy'
        self.eval_threshold = 0.5

        self.train_csv_file = './data/train.csv'
        self.dev_csv_file = './data/dev.csv'
        self.test_csv_file = './data/test.csv'