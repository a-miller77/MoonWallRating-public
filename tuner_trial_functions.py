import pandas as pd
import numpy as np
import json

def get_num_trials(base_dir):
    dir = base_dir + '/oracle.json'
    with open(dir) as f:
       data = json.load(f)
    return int(data['end_order'][-1])+1

def create_trials_np(num_trials, base_dir):
    trials_np = np.zeros(0)
    for trial in range(0, num_trials):
        dir = base_dir + '/trial_'
        trials_np = np.append(trials_np, create_np_row(dir, trial)).reshape(-1, 21)
    return trials_np

def create_np_row(dir, trial):
    trial = f'{trial:02d}'
    dir = dir + trial + '/trial.json'
    #print(dir)
    df = pd.read_json(dir)
    return df
#print(create_trials_np(9, 'tuners/tuner4/'))

# +
def create_df_row(trials_np, i):
    index = i
    
    hyperparameters = trials_np[i][8]
    
    num_layers = hyperparameters['num_layers']
    d_model = hyperparameters['d_model']
    dff = hyperparameters['dff']
    num_heads = hyperparameters["num_heads"]
    dropout_rate = hyperparameters['dropout_rate']
    warmup_steps = hyperparameters['warmup_steps']
    beta_1 = hyperparameters['beta_1']
    beta_2 = hyperparameters['beta_2']
    epsilon = hyperparameters['epsilon']
    
    batch_size = hyperparameters['batch_size']
    activation = hyperparameters['activation']
    sequential = hyperparameters['sequential']
    
    metrics = trials_np[i][16]
    
    #loss = metrics['loss']['observations'][0]['value'][0]
    #accuracy = metrics['accuracy']['observations'][0]['value'][0]
    #val_loss = metrics['val_loss']['observations'][0]['value'][0]
    val_accuracy = metrics['val_accuracy']['observations'][0]['value'][0]
    #score = trials_np[i][17]
    #best_step = metrics['loss']['observations'][0]['step']
    
    return pd.DataFrame({
        #'trial': index,
        'num_layers': num_layers,
        'd_model': d_model,
        'dff': dff,
        "num_heads": num_heads,
        'dropout_rate': dropout_rate,
        'warmup_steps': warmup_steps,
        'batch_size': batch_size,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'epsilon': epsilon,
        'activation': activation,
        'sequential': sequential,
        #'loss': loss,
        #'accuracy' : accuracy,
        #'val_loss' : val_loss,
        'val_accuracy' : val_accuracy,
        #'score': score,
        #'best_step': best_step},
        },
        index = [i])
        
#create_df_row(create_trials_np(num_trials=5, base_dir='tuner2'), 0)


# -

def create_tuner_df(num_trials, trials_np):
    df = create_df_row(trials_np, 0)
    for trial in range(1, num_trials):
        df = pd.concat((df, create_df_row(trials_np, trial)), axis=0)
    return df

def best_trials(tuner_df, num_trials=5):
    df = tuner_df.sort_values(by=['val_accuracy']).tail(num_trials)
    return df






