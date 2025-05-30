import torch

def load_model_with_key_mapping(model, checkpoint_path):
    """
    Load model state dict with automatic key mapping to handle architecture mismatches.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the state dict (handle both direct state_dict and nested checkpoint formats)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Create mapping from saved keys to model keys
    model_keys = set(model.state_dict().keys())
    saved_keys = set(state_dict.keys())
    
    # Check if we need to map between 'layers.X' and 'X' formats
    if any('layers.' in key for key in saved_keys) and not any('layers.' in key for key in model_keys):
        # Map from 'layers.X' to 'X'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('layers.'):
                new_key = key.replace('layers.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    elif any('layers.' in key for key in model_keys) and not any('layers.' in key for key in saved_keys):
        # Map from 'X' to 'layers.X'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.replace('.', '').replace('weight', '').replace('bias', '').isdigit():
                new_key = f'layers.{key}'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Load the mapped state dict
    model.load_state_dict(state_dict)
    return model
