import torch
from guardian import train, config

def resume_training():
    checkpoint = torch.load('checkpoints/model_exp1_best.pth')

    # Update config with saved values
    saved_config = checkpoint['config']
    config.update(saved_config)

    # Initialize model and load state
    model = EfficientNetB7().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Update starting epoch
    config['epochs'] = config['epochs'] - checkpoint['epoch']

    # Resume training
    model = train(config)

if __name__ == "__main__":
    resume_training()